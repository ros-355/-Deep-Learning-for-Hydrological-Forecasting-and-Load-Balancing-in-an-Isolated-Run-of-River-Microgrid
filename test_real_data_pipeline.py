from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from code.real_data_pipeline import (
    CSVSchema,
    RealDataConfig,
    RealMicrogridDataPipeline,
    save_preprocessed_npz,
)


def _historical_frame(hours: int = 1200) -> pd.DataFrame:
    timestamp = pd.date_range("2022-01-01", periods=hours, freq="h")
    t = np.arange(hours, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "river_flow_m3s": 0.65 + 0.18 * np.sin(2.0 * np.pi * t / (24.0 * 90.0)),
            "critical_load_kw": 22.0 + 2.5 * np.sin(2.0 * np.pi * t / 24.0),
            "noncritical_load_kw": 18.0 + 4.0 * np.maximum(
                0.0, np.sin(2.0 * np.pi * (t - 6.0) / 24.0)
            ),
            "precipitation_mm": np.maximum(0.0, 2.0 * np.sin(2.0 * np.pi * t / 168.0)),
            "temperature_c": 11.0 + 6.0 * np.sin(2.0 * np.pi * (t - 8.0) / 24.0),
            "snow_water_equivalent_mm": 120.0 - 0.02 * t,
        }
    )


def test_real_pipeline_imputes_optional_weather_and_builds_split_safe_windows(tmp_path):
    frame = _historical_frame()
    frame.loc[20:25, "river_flow_m3s"] = np.nan
    frame.loc[100:134, "critical_load_kw"] = 23.123456  # exact stale plateau
    frame.loc[700:710, "temperature_c"] = np.nan
    # Large test-only values must not influence training-fitted scalers.
    frame.loc[1020:, "river_flow_m3s"] += 100.0
    path = tmp_path / "historical.csv"
    frame.to_csv(path, index=False)

    result = RealMicrogridDataPipeline().prepare(path)

    assert result.quality_report["site"] == {
        "name": "Langtang Khola",
        "latitude": 28.21081,
        "longitude": 85.56169,
        "coordinates_status": "user-confirmed",
        "field_validation_status": (
            "Measured CSV ingested; provenance and station identity must be "
            "verified independently."
        ),
    }
    assert result.feature_names == (
        "river_flow_m3s",
        "critical_load_kw",
        "noncritical_load_kw",
        "precipitation_mm",
        "temperature_c",
        "snow_water_equivalent_mm",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    )
    assert result.x_train.shape[1:] == (72, 10)
    assert result.y_train.shape[1:] == (24,)
    assert result.x_validation.shape[1:] == (72, 10)
    assert result.x_test.shape[1:] == (72, 10)
    assert np.isfinite(result.x_train).all()
    assert np.isfinite(result.x_validation).all()
    assert np.isfinite(result.x_test).all()
    assert np.isfinite(result.y_train).all()

    train_end = result.split_boundaries["train_end"]
    validation_end = result.split_boundaries["validation_end"]
    train_features = result.cleaned_frame.iloc[:train_end][list(result.feature_names)]
    scaled_train = result.feature_scaler.transform(train_features)
    np.testing.assert_allclose(scaled_train.mean(axis=0), 0.0, atol=2.0e-6)
    assert result.target_scaler.mean_[0] < 2.0  # test-only flow spike did not leak

    assert result.cleaned_frame["was_imputed_river_flow_m3s"].sum() >= 6
    assert result.cleaned_frame["was_stale_critical_load_kw"].sum() > 0
    assert result.cleaned_frame["was_imputed_temperature_c"].sum() >= 11

    # Targets remain strictly inside their chronological partitions.
    train_boundary_utc = result.cleaned_frame.index[train_end].tz_convert("UTC").tz_localize(None)
    validation_boundary_utc = (
        result.cleaned_frame.index[validation_end].tz_convert("UTC").tz_localize(None)
    )
    assert result.train.target_end_timestamps[-1] < np.datetime64(train_boundary_utc)
    assert result.validation.target_start_timestamps[0] >= np.datetime64(train_boundary_utc)
    assert result.validation.target_end_timestamps[-1] < np.datetime64(
        validation_boundary_utc
    )
    assert result.test.target_start_timestamps[0] >= np.datetime64(validation_boundary_utc)

    output = tmp_path / "sequences.npz"
    save_preprocessed_npz(result, output)
    with np.load(output) as saved:
        assert saved["x_train"].shape == result.x_train.shape
        assert saved["feature_names"].tolist() == list(result.feature_names)


def test_real_pipeline_accepts_custom_required_column_names_without_weather(tmp_path):
    frame = _historical_frame().iloc[:1000].rename(
        columns={
            "timestamp": "DateTime",
            "river_flow_m3s": "Flow",
            "critical_load_kw": "Clinic_kW",
            "noncritical_load_kw": "Flexible_kW",
        }
    )
    frame = frame[["DateTime", "Flow", "Clinic_kW", "Flexible_kW"]]
    path = tmp_path / "custom.csv"
    frame.to_csv(path, index=False)
    schema = CSVSchema(
        timestamp="DateTime",
        river_flow="Flow",
        critical_load="Clinic_kW",
        noncritical_load="Flexible_kW",
    )

    result = RealMicrogridDataPipeline(schema=schema).prepare(path)

    assert result.feature_names == (
        "Flow",
        "Clinic_kW",
        "Flexible_kW",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    )
    assert result.x_train.shape[2] == 7


def test_real_pipeline_rejects_missing_columns_and_excessive_gaps(tmp_path):
    incomplete = _historical_frame().drop(columns="critical_load_kw")
    missing_path = tmp_path / "missing.csv"
    incomplete.to_csv(missing_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        RealMicrogridDataPipeline().prepare(missing_path)

    gappy = _historical_frame()
    gappy.loc[:500, "river_flow_m3s"] = np.nan
    gappy_path = tmp_path / "gappy.csv"
    gappy.to_csv(gappy_path, index=False)
    config = RealDataConfig(max_imputed_fraction=0.10)
    with pytest.raises(ValueError, match="exceeding"):
        RealMicrogridDataPipeline(config=config).prepare(gappy_path)
