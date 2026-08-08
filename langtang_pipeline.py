"""Real-flow Langtang Khola forecasting and hybrid microgrid experiment.

The hydrology in this pipeline comes from the ICIMOD/Utrecht University
Langtang Khola discharge archive (DOI: 10.26066/RDS.1972407). Because the
archive contains no electricity demand, the two load tiers remain explicitly
labelled literature-calibrated synthetic signals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code.config import GridConfig, ModelConfig
from code.data import hydraulic_power_kw
from code.grid import LookAheadMPC, MicroHydroGrid, proportional_no_control_baseline
from code.metrics import regression_metrics, resilience_metrics
from code.model import (
    build_fixed_batch_qat_clone,
    build_lstm_model,
    build_qat_model,
    export_full_int8_tflite,
    fine_tune_qat,
    predict_tflite_int8,
    print_memory_report,
    tflite_memory_report,
    train_float_model,
)
from code.real_data_pipeline import RealDataConfig, RealMicrogridDataPipeline


SOURCE_DOI = "10.26066/RDS.1972407"
SOURCE_CITATION = (
    "ICIMOD & Utrecht University. (2020). Discharge: Langtang Khola "
    "[Data set]. ICIMOD. https://doi.org/10.26066/RDS.1972407"
)
SOURCE_RELATIVE = Path(
    "data/raw/icimod_langtang_khola/data/LangtangKholaDischarge_Kyanjing.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_literature_calibrated_loads(
    timestamps: pd.DatetimeIndex,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible tiered loads without mislabelling them as metered data."""

    rng = np.random.default_rng(seed)
    hour = timestamps.hour.to_numpy(dtype=np.float64)
    day = timestamps.dayofyear.to_numpy(dtype=np.float64)

    critical = (
        19.0
        + 5.5 * np.exp(-0.5 * ((hour - 8.0) / 2.2) ** 2)
        + 8.0 * np.exp(-0.5 * ((hour - 19.0) / 2.8) ** 2)
        + 3.2 * (1.0 + np.cos(2.0 * np.pi * (day - 10.0) / 365.25)) / 2.0
        + rng.normal(0.0, 0.8, len(timestamps))
    )
    critical = np.clip(critical, 15.0, None)

    daylight = np.maximum(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0)
    monsoon = np.exp(-0.5 * ((day - 225.0) / 51.0) ** 2)
    dry_season = np.clip(1.0 - monsoon, 0.15, 1.0)
    noncritical = (
        7.0
        + 36.0 * daylight * dry_season
        + 16.0 * np.exp(-0.5 * ((hour - 6.5) / 1.5) ** 2)
        + 19.0 * np.exp(-0.5 * ((hour - 20.5) / 1.9) ** 2)
        + 11.0 * ((hour >= 9.0) & (hour <= 17.0))
        + rng.normal(0.0, 1.8, len(timestamps))
    )
    return critical.astype(np.float32), np.clip(noncritical, 3.0, None).astype(
        np.float32
    )


def _longest_complete_block(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame["river_flow_m3s"].notna()
    group = complete.ne(complete.shift()).cumsum()
    lengths = complete.groupby(group).sum()
    valid_groups = lengths[lengths > 0]
    if valid_groups.empty:
        raise ValueError("The ICIMOD archive contains no usable discharge values.")
    selected_group = int(valid_groups.idxmax())
    return frame.loc[group.eq(selected_group)].copy()


def prepare_langtang_hybrid_dataset(
    source_path: Path,
    output_csv: Path,
    provenance_path: Path,
    max_linear_gap_hours: int = 6,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate the archive, fill only short gaps, and retain the longest clean block."""

    raw = pd.read_csv(source_path, na_values=["NA"])
    required = {"DATE", "TIME", "Q [m3/s]"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"ICIMOD discharge CSV is missing columns: {missing}")

    timestamp = pd.to_datetime(
        raw["DATE"].astype(str) + " " + raw["TIME"].astype(str),
        errors="coerce",
    )
    if timestamp.isna().any():
        raise ValueError(f"Invalid ICIMOD timestamps: {int(timestamp.isna().sum())}")
    if timestamp.duplicated().any():
        raise ValueError(
            f"Duplicate ICIMOD timestamps: {int(timestamp.duplicated().sum())}"
        )

    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "river_flow_m3s": pd.to_numeric(raw["Q [m3/s]"], errors="coerce"),
            "stage_height_m": pd.to_numeric(raw.get("H [m]"), errors="coerce"),
            "water_temperature_c": pd.to_numeric(
                raw.get("water temperature [C]"), errors="coerce"
            ),
        }
    ).set_index("timestamp")
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    frame = frame.reindex(full_index)
    invalid_flow = frame["river_flow_m3s"].le(0.0)
    frame.loc[invalid_flow, "river_flow_m3s"] = np.nan

    missing_before = frame["river_flow_m3s"].isna()
    frame["river_flow_m3s"] = frame["river_flow_m3s"].interpolate(
        method="linear",
        limit=max_linear_gap_hours,
        limit_area="inside",
    )
    frame["flow_short_gap_imputed"] = (
        missing_before & frame["river_flow_m3s"].notna()
    )
    selected = _longest_complete_block(frame)
    if len(selected) < 365 * 24:
        raise ValueError(
            "No complete Langtang Khola block spans at least one year after "
            "short-gap interpolation."
        )

    selected.index = selected.index.tz_localize("Asia/Kathmandu")
    critical, noncritical = generate_literature_calibrated_loads(selected.index)
    selected["critical_load_kw"] = critical
    selected["noncritical_load_kw"] = noncritical
    output = selected.reset_index(names="timestamp")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    provenance: dict[str, object] = {
        "study_site": {
            "name": "Discharge Station Langtang Khola (Kyanjing)",
            "latitude": 28.21081,
            "longitude": 85.56169,
            "elevation_m": 3850,
            "timezone": "NPT / Asia-Kathmandu",
            "watershed": "Gandaki",
            "river_system": "Trisuli",
        },
        "source": {
            "provider": "ICIMOD and Utrecht University",
            "doi": SOURCE_DOI,
            "citation": SOURCE_CITATION,
            "file": SOURCE_RELATIVE.as_posix(),
            "sha256": sha256_file(source_path),
            "license": "Free to use with attribution; ICIMOD download/use agreement retained.",
        },
        "raw_record": {
            "rows": int(len(frame)),
            "start": str(frame.index.min()),
            "end": str(frame.index.max()),
            "valid_discharge_rows_before_short_gap_fill": int(
                (~missing_before).sum()
            ),
            "missing_discharge_rows_before_short_gap_fill": int(
                missing_before.sum()
            ),
            "short_gap_imputed_rows": int(frame["flow_short_gap_imputed"].sum()),
            "maximum_linear_gap_hours": max_linear_gap_hours,
        },
        "selected_contiguous_block": {
            "rows": int(len(selected)),
            "start": selected.index.min().isoformat(),
            "end": selected.index.max().isoformat(),
            "observed_flow_rows": int(
                len(selected) - selected["flow_short_gap_imputed"].sum()
            ),
            "short_gap_imputed_flow_rows": int(
                selected["flow_short_gap_imputed"].sum()
            ),
            "flow_min_m3s": float(selected["river_flow_m3s"].min()),
            "flow_max_m3s": float(selected["river_flow_m3s"].max()),
            "flow_mean_m3s": float(selected["river_flow_m3s"].mean()),
        },
        "field_provenance": {
            "river_flow_m3s": "measured/derived from ICIMOD stage-height record",
            "stage_height_m": "ICIMOD record where available",
            "water_temperature_c": "ICIMOD record where available; not used as air temperature",
            "critical_load_kw": "literature-calibrated synthetic demand",
            "noncritical_load_kw": "literature-calibrated synthetic demand",
            "plant_parameters": "hypothetical control-study assumptions, not surveyed",
        },
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output, provenance


def _prediction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    persistence: np.ndarray,
) -> dict[str, object]:
    model_scores = regression_metrics(actual, predicted)
    persistence_scores = regression_metrics(actual, persistence)
    leadwise = []
    for lead in range(actual.shape[1]):
        scores = regression_metrics(actual[:, lead], predicted[:, lead])
        baseline = regression_metrics(actual[:, lead], persistence[:, lead])
        leadwise.append(
            {
                "lead_hour": lead + 1,
                **scores,
                "persistence_mae_m3s": baseline["mae_m3s"],
            }
        )
    improvement = (
        100.0
        * (persistence_scores["mae_m3s"] - model_scores["mae_m3s"])
        / persistence_scores["mae_m3s"]
    )
    return {
        "int8_qat_lstm": model_scores,
        "persistence": persistence_scores,
        "mae_improvement_over_persistence_pct": float(improvement),
        "leadwise": leadwise,
    }


def _plot_real_profile(frame: pd.DataFrame, output_path: Path) -> None:
    daily = frame.set_index("timestamp")[
        ["river_flow_m3s", "critical_load_kw", "noncritical_load_kw"]
    ].resample("D").mean()
    fig, axes = plt.subplots(2, 1, figsize=(6.45, 4.2), sharex=True)
    axes[0].plot(daily.index, daily["river_flow_m3s"], color="#1768AC", lw=1.1)
    axes[0].set_ylabel("Flow\n(m$^3$ s$^{-1}$)")
    axes[0].set_title("Observed Langtang Khola flow and synthetic tiered demand")
    axes[1].plot(
        daily.index, daily["critical_load_kw"], color="#D1495B", label="Critical"
    )
    axes[1].plot(
        daily.index,
        daily["noncritical_load_kw"],
        color="#F28E2B",
        label="Non-critical",
    )
    axes[1].set_ylabel("Mean load (kW)")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_real_forecast(
    timestamps: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
) -> None:
    count = min(336, len(actual))
    x = pd.to_datetime(timestamps[:count])
    fig, axis = plt.subplots(figsize=(6.45, 2.75))
    axis.plot(x, actual[:count], color="#1768AC", lw=1.3, label="Observed flow")
    axis.plot(
        x,
        predicted[:count],
        color="#F28E2B",
        lw=1.1,
        ls="--",
        label="INT8 QAT-LSTM, 24-h lead",
    )
    axis.set_ylabel("Flow (m$^3$ s$^{-1}$)")
    axis.set_title("Langtang Khola 24-hour-ahead forecast")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_real_stress(
    timestamps: pd.DatetimeIndex,
    flow: np.ndarray,
    generation: np.ndarray,
    critical: np.ndarray,
    noncritical: np.ndarray,
    controlled: dict[str, np.ndarray],
    baseline_critical: np.ndarray,
    output_path: Path,
) -> None:
    hour = np.arange(len(flow))
    fig, axes = plt.subplots(3, 1, figsize=(6.45, 5.8), sharex=True)
    axes[0].plot(hour, flow, color="#1768AC", lw=1.3)
    axes[0].set_ylabel("Observed flow\n(m$^3$ s$^{-1}$)")
    axes[0].set_title(
        "Observed low-flow stress window: "
        f"{timestamps[0]:%Y-%m-%d} to {timestamps[-1]:%Y-%m-%d}"
    )
    axes[1].plot(hour, generation, color="#2A9D8F", label="Generation")
    axes[1].plot(hour, critical + noncritical, color="#636363", ls=":", label="Total demand")
    axes[1].plot(hour, critical, color="#D1495B", label="Critical demand")
    axes[1].set_ylabel("Power (kW)")
    axes[1].legend(frameon=False, ncol=3)
    axes[2].plot(
        hour,
        100.0 * controlled["critical_served_kw"] / critical,
        color="#2A9D8F",
        label="Critical supply with MPC",
    )
    axes[2].plot(
        hour,
        100.0 * baseline_critical / critical,
        color="#D1495B",
        ls="--",
        label="Critical supply without control",
    )
    axes[2].fill_between(
        hour,
        0.0,
        100.0 * controlled["curtailment_fraction"],
        color="#F28E2B",
        alpha=0.25,
        label="Non-critical curtailment",
    )
    axes[2].set_ylim(0.0, 106.0)
    axes[2].set_ylabel("Supply / command (%)")
    axes[2].set_xlabel("Hour in observed 96-hour window")
    axes[2].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_langtang_experiment(
    root: Path,
    epochs: int | None = None,
    qat_epochs: int | None = None,
    quiet: bool = False,
) -> dict[str, object]:
    source_path = root / SOURCE_RELATIVE
    if not source_path.exists():
        raise FileNotFoundError(
            f"Langtang discharge archive is missing: {source_path}. "
            "Real mode does not fall back to synthetic flow."
        )

    data_dir = root / "data"
    results_dir = root / "results"
    figures_dir = root / "figures"
    for directory in (data_dir, results_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    hybrid_csv = data_dir / "langtang_khola_real_flow_hybrid_load.csv"
    provenance_json = data_dir / "langtang_data_provenance.json"
    hybrid_frame, provenance = prepare_langtang_hybrid_dataset(
        source_path, hybrid_csv, provenance_json
    )

    preprocess_config = RealDataConfig(
        site_name="Langtang Khola",
        latitude=28.21081,
        longitude=85.56169,
        coordinates_are_provisional=False,
        max_imputed_fraction=0.01,
    )
    prepared = RealMicrogridDataPipeline(config=preprocess_config).prepare(hybrid_csv)
    prepared.quality_report["source"] = "data/langtang_khola_real_flow_hybrid_load.csv"
    prepared.quality_report["site"]["coordinates_status"] = "source-metadata-confirmed"
    prepared.quality_report["site"]["field_validation_status"] = (
        "Measured archive ingested; station identity verified against the supplied "
        "ICIMOD metadata and station sheet."
    )
    default_model = ModelConfig()
    model_config = ModelConfig(
        **{
            **asdict(default_model),
            "epochs": epochs or default_model.epochs,
            "qat_epochs": qat_epochs or default_model.qat_epochs,
        }
    )

    model = build_lstm_model(len(prepared.feature_names), model_config)
    float_history = train_float_model(
        model,
        prepared.x_train,
        prepared.y_train,
        prepared.x_validation,
        prepared.y_validation,
        model_config,
        verbose=0 if quiet else 2,
    )
    qat_model = build_qat_model(model, model_config)
    qat_history = fine_tune_qat(
        qat_model,
        prepared.x_train,
        prepared.y_train,
        prepared.x_validation,
        prepared.y_validation,
        model_config,
        verbose=0 if quiet else 2,
    )
    export_model = build_fixed_batch_qat_clone(
        qat_model, len(prepared.feature_names), model_config
    )
    tflite_path = results_dir / "langtang_hydro_qat_lstm_int8.tflite"
    tflite_model = export_full_int8_tflite(
        export_model, prepared.x_train, tflite_path
    )
    memory = tflite_memory_report(tflite_model)
    print_memory_report(memory)

    prediction_scaled = predict_tflite_int8(tflite_model, prepared.x_test)
    prediction = prepared.target_scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).reshape(prediction_scaled.shape)
    actual = prepared.target_scaler.inverse_transform(
        prepared.y_test.reshape(-1, 1)
    ).reshape(prepared.y_test.shape)
    last_flow_scaled = prepared.x_test[:, -1, 0]
    last_flow = prepared.target_scaler.inverse_transform(
        last_flow_scaled.reshape(-1, 1)
    )[:, 0]
    persistence = np.repeat(last_flow[:, None], model_config.forecast_hours, axis=1)
    forecast_metrics = _prediction_metrics(actual, prediction, persistence)

    lead24_timestamps = prepared.test.target_end_timestamps
    _plot_real_profile(
        hybrid_frame,
        figures_dir / "figure_7_langtang_real_profile.png",
    )
    _plot_real_forecast(
        lead24_timestamps,
        actual[:, -1],
        prediction[:, -1],
        figures_dir / "figure_8_langtang_forecast_accuracy.png",
    )

    forecast_rows = []
    for sample in range(len(actual)):
        for lead in range(model_config.forecast_hours):
            forecast_rows.append(
                {
                    "target_start_utc": str(
                        prepared.test.target_start_timestamps[sample]
                    ),
                    "lead_hour": lead + 1,
                    "actual_flow_m3s": float(actual[sample, lead]),
                    "int8_prediction_m3s": float(prediction[sample, lead]),
                    "persistence_m3s": float(persistence[sample, lead]),
                }
            )
    pd.DataFrame(forecast_rows).to_csv(
        data_dir / "langtang_forecast_test_all_leads.csv", index=False
    )
    pd.DataFrame(forecast_metrics["leadwise"]).to_csv(
        data_dir / "langtang_forecast_leadwise_metrics.csv", index=False
    )

    test_starts = pd.to_datetime(prepared.test.target_start_timestamps, utc=True)
    first_lead_flow = actual[:, 0]
    rolling_mean = (
        pd.Series(first_lead_flow)
        .rolling(96, min_periods=96)
        .mean()
        .to_numpy()
    )
    stress_end = int(np.nanargmin(rolling_mean))
    stress_start = stress_end - 95
    stress_slice = slice(stress_start, stress_end + 1)
    stress_timestamps_utc = test_starts[stress_slice]
    stress_timestamps_local = stress_timestamps_utc.tz_convert("Asia/Kathmandu")
    stress_flow = first_lead_flow[stress_slice]
    rolling_forecasts = prediction[stress_slice]

    source_local_index = hybrid_frame.set_index("timestamp")
    critical = source_local_index.loc[
        stress_timestamps_local, "critical_load_kw"
    ].to_numpy(dtype=np.float32)
    noncritical = source_local_index.loc[
        stress_timestamps_local, "noncritical_load_kw"
    ].to_numpy(dtype=np.float32)

    control_grid = GridConfig(environmental_flow_m3s=3.35)
    grid = MicroHydroGrid(control_grid)
    controlled = grid.simulate(
        stress_flow,
        critical,
        noncritical,
        rolling_forecasts,
        LookAheadMPC(control_grid),
    )
    baseline_critical, baseline_noncritical = proportional_no_control_baseline(
        controlled["generation_kw"], critical, noncritical
    )
    controlled_metrics = resilience_metrics(
        critical,
        controlled["critical_served_kw"],
        noncritical,
        controlled["noncritical_served_kw"],
    )
    baseline_metrics = resilience_metrics(
        critical, baseline_critical, noncritical, baseline_noncritical
    )
    controlled_metrics["blackout_hours_avoided"] = (
        baseline_metrics["critical_shortfall_hours"]
        - controlled_metrics["critical_shortfall_hours"]
    )
    _plot_real_stress(
        stress_timestamps_local,
        stress_flow,
        controlled["generation_kw"],
        critical,
        noncritical,
        controlled,
        baseline_critical,
        figures_dir / "figure_9_langtang_mpc_stress_test.png",
    )
    stress_frame = pd.DataFrame(
        {
            "timestamp_npt": stress_timestamps_local.astype(str),
            "observed_flow_m3s": stress_flow,
            "critical_load_kw_synthetic": critical,
            "noncritical_load_kw_synthetic": noncritical,
            "generation_kw_hypothetical": controlled["generation_kw"],
            "mpc_curtailment_fraction": controlled["curtailment_fraction"],
            "mpc_critical_served_kw": controlled["critical_served_kw"],
            "mpc_noncritical_served_kw": controlled["noncritical_served_kw"],
            "baseline_critical_served_kw": baseline_critical,
            "baseline_noncritical_served_kw": baseline_noncritical,
        }
    )
    stress_frame.to_csv(data_dir / "langtang_observed_low_flow_stress.csv", index=False)

    metrics: dict[str, object] = {
        "provenance": provenance,
        "preprocessing": prepared.quality_report,
        "model": {
            "feature_count": len(prepared.feature_names),
            "parameter_count_float": int(model.count_params()),
            "lookback_hours": model_config.lookback_hours,
            "forecast_hours": model_config.forecast_hours,
            "float_final_val_loss": float(float_history.history["val_loss"][-1]),
            "qat_final_val_loss": float(qat_history.history["val_loss"][-1]),
            **memory,
        },
        "forecast": forecast_metrics,
        "observed_low_flow_stress": {
            "start_npt": stress_timestamps_local[0].isoformat(),
            "end_npt": stress_timestamps_local[-1].isoformat(),
            "plant_assumptions": {
                **asdict(control_grid),
                "status": (
                    "Hypothetical control-study parameters; environmental flow "
                    "is a feasibility-preserving assumption, not a surveyed value."
                ),
            },
            "with_mpc": controlled_metrics,
            "without_control": baseline_metrics,
        },
    }
    metrics_path = results_dir / "langtang_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics["forecast"], indent=2))
    print(json.dumps(metrics["observed_low_flow_stress"], indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--qat-epochs", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_langtang_experiment(
        args.project_root.resolve(),
        epochs=args.epochs,
        qat_epochs=args.qat_epochs,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
