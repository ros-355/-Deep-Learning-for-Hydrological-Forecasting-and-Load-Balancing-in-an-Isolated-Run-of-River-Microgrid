"""Leakage-safe preprocessing for real hourly microgrid measurements.

The module preserves the model contract used by the digital twin:

* one feature window has shape ``(72, n_features)``;
* one target has shape ``(24,)``;
* a single Keras input batch is therefore ``x[i:i + 1]`` with shape
  ``(1, 72, n_features)``.

Only training-period observations are used to fit imputers and scalers. Windows
are built independently inside the train, validation, and test partitions, so
neither their input history nor their forecast target crosses a split boundary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CSVSchema:
    """Map source CSV column names to the digital-twin signal roles."""

    timestamp: str = "timestamp"
    river_flow: str = "river_flow_m3s"
    critical_load: str = "critical_load_kw"
    noncritical_load: str = "noncritical_load_kw"
    precipitation: str = "precipitation_mm"
    temperature: str = "temperature_c"
    snow_water_equivalent: str = "snow_water_equivalent_mm"

    @property
    def required_signal_columns(self) -> tuple[str, ...]:
        return (self.river_flow, self.critical_load, self.noncritical_load)

    @property
    def optional_signal_columns(self) -> tuple[str, ...]:
        return (self.precipitation, self.temperature, self.snow_water_equivalent)


@dataclass(frozen=True)
class RealDataConfig:
    """Cleaning, split, and sequence parameters for real measurements."""

    site_name: str = "Langtang Khola"
    latitude: float = 28.21081
    longitude: float = 85.56169
    coordinates_are_provisional: bool = False
    timezone: str = "Asia/Kathmandu"
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    lookback_hours: int = 72
    forecast_hours: int = 24
    short_forward_fill_hours: int = 3
    stale_after_hours: int = 6
    load_stale_after_hours: int = 24
    meteorology_stale_after_hours: int = 24
    stale_tolerance: float = 1.0e-9
    max_imputed_fraction: float = 0.25
    duplicate_policy: str = "mean"
    physical_bounds: Mapping[str, tuple[float | None, float | None]] = field(
        default_factory=lambda: {
            "river_flow_m3s": (0.0, None),
            "critical_load_kw": (0.0, None),
            "noncritical_load_kw": (0.0, None),
            "precipitation_mm": (0.0, None),
            "temperature_c": (-60.0, 60.0),
            "snow_water_equivalent_mm": (0.0, None),
        }
    )

    def validate(self) -> None:
        if not self.site_name.strip():
            raise ValueError("site_name must not be empty.")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must lie between -90 and 90 degrees.")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must lie between -180 and 180 degrees.")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie between 0 and 1.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie between 0 and 1.")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("Train and validation fractions must leave a test partition.")
        if self.lookback_hours <= 0 or self.forecast_hours <= 0:
            raise ValueError("lookback_hours and forecast_hours must be positive.")
        if (
            self.short_forward_fill_hours < 0
            or self.stale_after_hours < 1
            or self.load_stale_after_hours < 1
            or self.meteorology_stale_after_hours < 1
        ):
            raise ValueError("Imputation and stale-sensor limits are invalid.")
        if not 0.0 <= self.max_imputed_fraction <= 1.0:
            raise ValueError("max_imputed_fraction must lie between 0 and 1.")
        if self.duplicate_policy not in {"mean", "last", "raise"}:
            raise ValueError("duplicate_policy must be 'mean', 'last', or 'raise'.")


@dataclass(frozen=True)
class ImputationProfile:
    """Training-only seasonal statistics used for causal gap filling."""

    global_median: Mapping[str, float]
    hour_of_day_median: Mapping[str, pd.Series]
    hour_of_week_median: Mapping[str, pd.Series]


@dataclass(frozen=True)
class WindowedSplit:
    """One chronological partition after scaling and window construction."""

    x: np.ndarray
    y: np.ndarray
    input_start_timestamps: np.ndarray
    target_start_timestamps: np.ndarray
    target_end_timestamps: np.ndarray


@dataclass(frozen=True)
class RealSequenceData:
    """Complete preprocessing output ready for a Keras/TensorFlow model."""

    train: WindowedSplit
    validation: WindowedSplit
    test: WindowedSplit
    feature_names: tuple[str, ...]
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    imputation_profile: ImputationProfile
    cleaned_frame: pd.DataFrame
    split_boundaries: Mapping[str, int]
    quality_report: Mapping[str, object]

    @property
    def x_train(self) -> np.ndarray:
        return self.train.x

    @property
    def y_train(self) -> np.ndarray:
        return self.train.y

    @property
    def x_validation(self) -> np.ndarray:
        return self.validation.x

    @property
    def y_validation(self) -> np.ndarray:
        return self.validation.y

    @property
    def x_test(self) -> np.ndarray:
        return self.test.x

    @property
    def y_test(self) -> np.ndarray:
        return self.test.y


class RealMicrogridDataPipeline:
    """Ingest, clean, engineer, scale, and window real hourly sensor data."""

    def __init__(
        self,
        config: RealDataConfig | None = None,
        schema: CSVSchema | None = None,
    ) -> None:
        self.config = config or RealDataConfig()
        self.schema = schema or CSVSchema()
        self.config.validate()

    def prepare(self, csv_path: str | Path) -> RealSequenceData:
        """Run the complete preprocessing pipeline for one CSV file."""

        raw, ingestion_report = self._load_and_regularize(Path(csv_path))
        marked, sensor_report = self._mark_invalid_and_stale(raw)
        train_end, validation_end = self._split_boundaries(len(marked))

        minimum_partition = self.config.lookback_hours + self.config.forecast_hours
        partition_lengths = (
            train_end,
            validation_end - train_end,
            len(marked) - validation_end,
        )
        if min(partition_lengths) < minimum_partition:
            raise ValueError(
                "Each chronological partition must contain at least "
                f"{minimum_partition} hourly rows; observed lengths are "
                f"train={partition_lengths[0]}, validation={partition_lengths[1]}, "
                f"test={partition_lengths[2]}. Supply more history or adjust fractions."
            )

        signal_columns = self._signal_columns(marked)
        profile = self._fit_imputation_profile(marked.iloc[:train_end], signal_columns)
        cleaned, imputation_report = self._impute_partitions(
            marked,
            signal_columns,
            train_end,
            validation_end,
            profile,
        )
        featured, feature_names = self._engineer_features(cleaned, signal_columns)

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        feature_scaler.fit(featured.iloc[:train_end][list(feature_names)])
        target_scaler.fit(featured.iloc[:train_end][[self.schema.river_flow]])

        scaled_features = feature_scaler.transform(featured[list(feature_names)]).astype(
            np.float32
        )
        scaled_target = target_scaler.transform(featured[[self.schema.river_flow]]).astype(
            np.float32
        )[:, 0]

        train = self._window_partition(
            scaled_features[:train_end],
            scaled_target[:train_end],
            featured.index[:train_end],
            "training",
        )
        validation = self._window_partition(
            scaled_features[train_end:validation_end],
            scaled_target[train_end:validation_end],
            featured.index[train_end:validation_end],
            "validation",
        )
        test = self._window_partition(
            scaled_features[validation_end:],
            scaled_target[validation_end:],
            featured.index[validation_end:],
            "test",
        )

        quality_report: dict[str, object] = {
            "source": str(Path(csv_path).resolve()),
            "site": {
                "name": self.config.site_name,
                "latitude": self.config.latitude,
                "longitude": self.config.longitude,
                "coordinates_status": (
                    "provisional basin-scale retrieval default"
                    if self.config.coordinates_are_provisional
                    else "user-confirmed"
                ),
                "field_validation_status": (
                    "Measured CSV ingested; provenance and station identity must be "
                    "verified independently."
                ),
            },
            "rows_after_hourly_regularization": len(featured),
            "timezone": self.config.timezone,
            "feature_names": list(feature_names),
            "split_rows": {
                "train": partition_lengths[0],
                "validation": partition_lengths[1],
                "test": partition_lengths[2],
            },
            "window_counts": {
                "train": len(train.x),
                "validation": len(validation.x),
                "test": len(test.x),
            },
            "ingestion": ingestion_report,
            "sensor_quality": sensor_report,
            "imputation": imputation_report,
        }
        return RealSequenceData(
            train=train,
            validation=validation,
            test=test,
            feature_names=feature_names,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            imputation_profile=profile,
            cleaned_frame=featured,
            split_boundaries={"train_end": train_end, "validation_end": validation_end},
            quality_report=quality_report,
        )

    def _load_and_regularize(self, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = (self.schema.timestamp, *self.schema.required_signal_columns)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        available_signals = [
            column
            for column in (*self.schema.required_signal_columns, *self.schema.optional_signal_columns)
            if column in frame.columns
        ]
        timestamps = pd.to_datetime(frame[self.schema.timestamp], errors="coerce")
        invalid_timestamps = int(timestamps.isna().sum())
        if invalid_timestamps:
            frame = frame.loc[timestamps.notna()].copy()
            timestamps = timestamps.loc[timestamps.notna()]
        if frame.empty:
            raise ValueError("No valid timestamped rows remain after parsing the CSV.")

        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(
                self.config.timezone, ambiguous="raise", nonexistent="raise"
            )
        else:
            timestamps = timestamps.dt.tz_convert(self.config.timezone)
        frame = frame[available_signals].copy()
        for column in available_signals:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.index = pd.DatetimeIndex(timestamps).floor("h")
        frame.index.name = "timestamp"
        frame = frame.sort_index()

        duplicate_rows = int(frame.index.duplicated(keep=False).sum())
        if duplicate_rows and self.config.duplicate_policy == "raise":
            raise ValueError("Duplicate or sub-hourly timestamps collapse to the same hour.")
        if duplicate_rows:
            if self.config.duplicate_policy == "mean":
                frame = frame.groupby(level=0).mean()
            else:
                frame = frame.groupby(level=0).last()

        full_index = pd.date_range(
            frame.index.min(), frame.index.max(), freq="h", tz=self.config.timezone
        )
        inserted_hours = len(full_index) - len(frame)
        frame = frame.reindex(full_index)
        frame.index.name = "timestamp"
        return frame, {
            "source_rows": int(len(timestamps)),
            "invalid_timestamp_rows_dropped": invalid_timestamps,
            "duplicate_rows_detected": duplicate_rows,
            "missing_hour_rows_inserted": int(inserted_hours),
        }

    def _signal_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        return tuple(
            column
            for column in (*self.schema.required_signal_columns, *self.schema.optional_signal_columns)
            if column in frame.columns
        )

    def _bounds_for(self, column: str) -> tuple[float | None, float | None]:
        role_bounds = {
            self.schema.river_flow: self.config.physical_bounds.get("river_flow_m3s"),
            self.schema.critical_load: self.config.physical_bounds.get("critical_load_kw"),
            self.schema.noncritical_load: self.config.physical_bounds.get(
                "noncritical_load_kw"
            ),
            self.schema.precipitation: self.config.physical_bounds.get("precipitation_mm"),
            self.schema.temperature: self.config.physical_bounds.get("temperature_c"),
            self.schema.snow_water_equivalent: self.config.physical_bounds.get(
                "snow_water_equivalent_mm"
            ),
        }
        return role_bounds.get(column) or (None, None)

    def _mark_invalid_and_stale(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        marked = frame.copy()
        report: dict[str, object] = {}
        stale_candidates = {
            self.schema.river_flow,
            self.schema.critical_load,
            self.schema.noncritical_load,
            self.schema.temperature,
            self.schema.snow_water_equivalent,
        }
        for column in self._signal_columns(marked):
            values = marked[column]
            lower, upper = self._bounds_for(column)
            invalid = pd.Series(False, index=marked.index)
            if lower is not None:
                invalid |= values < lower
            if upper is not None:
                invalid |= values > upper
            marked.loc[invalid, column] = np.nan

            stale = pd.Series(False, index=marked.index)
            if column in stale_candidates:
                stale = self._stale_mask(marked[column], self._stale_threshold(column))
                marked.loc[stale, column] = np.nan
            marked[f"was_invalid_{column}"] = invalid
            marked[f"was_stale_{column}"] = stale
            report[column] = {
                "missing_before_imputation": int(marked[column].isna().sum()),
                "physically_invalid_marked_missing": int(invalid.sum()),
                "stale_marked_missing": int(stale.sum()),
            }
        return marked, report

    def _stale_threshold(self, column: str) -> int:
        if column in {self.schema.critical_load, self.schema.noncritical_load}:
            return self.config.load_stale_after_hours
        if column in {self.schema.temperature, self.schema.snow_water_equivalent}:
            return self.config.meteorology_stale_after_hours
        return self.config.stale_after_hours

    def _stale_mask(self, values: pd.Series, stale_after: int) -> pd.Series:
        previous = values.shift(1)
        new_run = (
            values.isna()
            | previous.isna()
            | (values - previous).abs().gt(self.config.stale_tolerance)
        )
        if len(new_run):
            new_run.iloc[0] = True
        run_id = new_run.cumsum()
        position = values.groupby(run_id).cumcount()
        return values.notna() & position.ge(stale_after)

    def _split_boundaries(self, length: int) -> tuple[int, int]:
        train_end = int(length * self.config.train_fraction)
        validation_end = int(
            length * (self.config.train_fraction + self.config.validation_fraction)
        )
        return train_end, validation_end

    @staticmethod
    def _seasonal_key(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(index.hour), np.asarray(index.dayofweek * 24 + index.hour)

    def _fit_imputation_profile(
        self, train: pd.DataFrame, columns: tuple[str, ...]
    ) -> ImputationProfile:
        hour, hour_of_week = self._seasonal_key(train.index)
        global_median: dict[str, float] = {}
        hourly: dict[str, pd.Series] = {}
        weekly: dict[str, pd.Series] = {}
        for column in columns:
            observed = train[column].dropna()
            if observed.empty:
                raise ValueError(
                    f"Training partition has no usable observations for {column!r}."
                )
            global_median[column] = float(observed.median())
            temporary = pd.DataFrame(
                {
                    "value": train[column].to_numpy(),
                    "hour": hour,
                    "hour_of_week": hour_of_week,
                },
                index=train.index,
            )
            hourly[column] = temporary.groupby("hour")["value"].median()
            weekly[column] = temporary.groupby("hour_of_week")["value"].median()
        return ImputationProfile(global_median, hourly, weekly)

    def _impute_partitions(
        self,
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        train_end: int,
        validation_end: int,
        profile: ImputationProfile,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        result = frame.copy()
        for column in columns:
            result[f"was_imputed_{column}"] = False
        partitions = ((0, train_end), (train_end, validation_end), (validation_end, len(frame)))
        for start, end in partitions:
            part = result.iloc[start:end].copy()
            hours, hour_of_week = self._seasonal_key(part.index)
            for column in columns:
                original_missing = part[column].isna()
                values = part[column].copy()
                # Precipitation is an accumulation, so carrying the previous amount
                # forward would fabricate rainfall. Other physical sensors receive a
                # short causal hold before seasonal fallback.
                if (
                    column != self.schema.precipitation
                    and self.config.short_forward_fill_hours > 0
                ):
                    values = values.ffill(limit=self.config.short_forward_fill_hours)
                missing = values.isna()
                if missing.any():
                    weekly_values = pd.Series(hour_of_week, index=part.index).map(
                        profile.hour_of_week_median[column]
                    )
                    values = values.fillna(weekly_values)
                missing = values.isna()
                if missing.any():
                    hourly_values = pd.Series(hours, index=part.index).map(
                        profile.hour_of_day_median[column]
                    )
                    values = values.fillna(hourly_values)
                values = values.fillna(profile.global_median[column])
                result.loc[part.index, column] = values.to_numpy()
                result.loc[part.index, f"was_imputed_{column}"] = (
                    original_missing.to_numpy(dtype=bool)
                )

        report: dict[str, object] = {}
        for column in columns:
            flag = result[f"was_imputed_{column}"].astype(bool)
            fraction = float(flag.mean())
            report[column] = {
                "imputed_rows": int(flag.sum()),
                "imputed_fraction": fraction,
                "training_global_median": profile.global_median[column],
            }
            if fraction > self.config.max_imputed_fraction:
                raise ValueError(
                    f"{column!r} requires {fraction:.1%} imputation, exceeding the "
                    f"configured {self.config.max_imputed_fraction:.1%} quality limit."
                )
            if result[column].isna().any():
                raise RuntimeError(f"Imputation failed to fill all values in {column!r}.")
        return result, report

    def _engineer_features(
        self, frame: pd.DataFrame, signal_columns: tuple[str, ...]
    ) -> tuple[pd.DataFrame, tuple[str, ...]]:
        result = frame.copy()
        hour = result.index.hour.to_numpy(dtype=np.float64)
        # Use the actual year length so leap-year data remains continuous.
        day_zero_based = result.index.dayofyear.to_numpy(dtype=np.float64) - 1.0
        days_in_year = np.where(result.index.is_leap_year, 366.0, 365.0)
        result["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        result["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        result["day_sin"] = np.sin(2.0 * np.pi * day_zero_based / days_in_year)
        result["day_cos"] = np.cos(2.0 * np.pi * day_zero_based / days_in_year)
        feature_names = (
            *signal_columns,
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
        )
        return result, tuple(feature_names)

    def _window_partition(
        self,
        features: np.ndarray,
        target: np.ndarray,
        timestamps: pd.DatetimeIndex,
        partition_name: str,
    ) -> WindowedSplit:
        lookback = self.config.lookback_hours
        horizon = self.config.forecast_hours
        sample_count = len(features) - lookback - horizon + 1
        if sample_count <= 0:
            raise ValueError(f"The {partition_name} partition is too short for windowing.")
        feature_count = features.shape[1]
        x = np.empty((sample_count, lookback, feature_count), dtype=np.float32)
        y = np.empty((sample_count, horizon), dtype=np.float32)
        input_start = np.empty(sample_count, dtype="datetime64[ns]")
        target_start = np.empty(sample_count, dtype="datetime64[ns]")
        target_end = np.empty(sample_count, dtype="datetime64[ns]")
        naive_utc = timestamps.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
        for sample in range(sample_count):
            forecast_start = sample + lookback
            forecast_end = forecast_start + horizon
            x[sample] = features[sample:forecast_start]
            y[sample] = target[forecast_start:forecast_end]
            input_start[sample] = naive_utc[sample]
            target_start[sample] = naive_utc[forecast_start]
            target_end[sample] = naive_utc[forecast_end - 1]
        return WindowedSplit(x, y, input_start, target_start, target_end)


def save_preprocessed_npz(data: RealSequenceData, output_path: str | Path) -> None:
    """Persist arrays and scaler parameters without pickled Python objects."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_train=data.x_train,
        y_train=data.y_train,
        x_validation=data.x_validation,
        y_validation=data.y_validation,
        x_test=data.x_test,
        y_test=data.y_test,
        feature_names=np.asarray(data.feature_names, dtype=np.str_),
        feature_mean=data.feature_scaler.mean_.astype(np.float64),
        feature_scale=data.feature_scaler.scale_.astype(np.float64),
        target_mean=data.target_scaler.mean_.astype(np.float64),
        target_scale=data.target_scaler.scale_.astype(np.float64),
        test_target_start_utc=data.test.target_start_timestamps,
        test_target_end_utc=data.test.target_end_timestamps,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Hourly historical microgrid CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/real_microgrid_sequences.npz"),
        help="Compressed NumPy output file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/real_microgrid_quality_report.json"),
        help="Data-quality JSON report",
    )
    parser.add_argument("--timezone", default="Asia/Kathmandu")
    parser.add_argument("--site-name", default="Langtang Khola")
    parser.add_argument("--latitude", type=float, default=28.21081)
    parser.add_argument("--longitude", type=float, default=85.56169)
    parser.add_argument(
        "--coordinates-provisional",
        action="store_true",
        help="Flag supplied coordinates as provisional rather than source-confirmed.",
    )
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--flow-column", default="river_flow_m3s")
    parser.add_argument("--critical-load-column", default="critical_load_kw")
    parser.add_argument("--noncritical-load-column", default="noncritical_load_kw")
    parser.add_argument("--precipitation-column", default="precipitation_mm")
    parser.add_argument("--temperature-column", default="temperature_c")
    parser.add_argument("--snow-water-equivalent-column", default="snow_water_equivalent_mm")
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    schema = CSVSchema(
        timestamp=args.timestamp_column,
        river_flow=args.flow_column,
        critical_load=args.critical_load_column,
        noncritical_load=args.noncritical_load_column,
        precipitation=args.precipitation_column,
        temperature=args.temperature_column,
        snow_water_equivalent=args.snow_water_equivalent_column,
    )
    config = RealDataConfig(
        site_name=args.site_name,
        latitude=args.latitude,
        longitude=args.longitude,
        coordinates_are_provisional=args.coordinates_provisional,
        timezone=args.timezone,
    )
    result = RealMicrogridDataPipeline(config=config, schema=schema).prepare(args.csv)
    save_preprocessed_npz(result, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result.quality_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(args.report),
                "features": result.feature_names,
                "x_train": result.x_train.shape,
                "y_train": result.y_train.shape,
                "x_validation": result.x_validation.shape,
                "x_test": result.x_test.shape,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
