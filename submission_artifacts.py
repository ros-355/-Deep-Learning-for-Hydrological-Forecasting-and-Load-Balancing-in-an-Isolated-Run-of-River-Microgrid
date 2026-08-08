"""Create submission-grade CSV datasets, extended metrics, and analysis figures."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from code.config import DataConfig, GridConfig, ModelConfig
from code.data import SyntheticMicrogridData, generate_himalayan_data, hydraulic_power_kw
from code.grid import LookAheadMPC, MicroHydroGrid, proportional_no_control_baseline
from code.metrics import regression_metrics, resilience_metrics
from code.model import predict_tflite_int8
from code.preprocessing import prepare_sequence_data
from code.stress_test import make_stress_test


NPT = timezone(timedelta(hours=5, minutes=45))


def _timestamp(year: int, hour_index: int) -> str:
    start = datetime(year, 1, 1, tzinfo=NPT)
    return (start + timedelta(hours=int(hour_index))).isoformat()


def _write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def _file_record(path: Path, rows: int, columns: Sequence[str]) -> dict[str, object]:
    return {
        "file": path.name,
        "rows": int(rows),
        "columns": list(columns),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def export_annual_dataset(
    data: SyntheticMicrogridData,
    data_config: DataConfig,
    model_config: ModelConfig,
    output_path: Path,
) -> dict[str, object]:
    headers = [
        "timestamp_npt",
        "hour_index",
        "day_of_year",
        "hour_of_day",
        "data_split",
        "river_flow_m3s",
        "available_generation_kw",
        "critical_load_kw",
        "noncritical_load_kw",
        "total_load_kw",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "is_flash_flood",
        "is_drought",
        "generator_seed",
    ]
    train_end = int(data_config.hours * model_config.train_fraction)
    validation_end = int(
        data_config.hours
        * (model_config.train_fraction + model_config.validation_fraction)
    )

    def rows() -> Iterable[Sequence[object]]:
        for index in range(data_config.hours):
            split = (
                "train"
                if index < train_end
                else "validation"
                if index < validation_end
                else "test"
            )
            yield [
                _timestamp(data_config.year, index),
                index,
                index // 24 + 1,
                index % 24,
                split,
                f"{data.flow_m3s[index]:.8f}",
                f"{data.available_generation_kw[index]:.8f}",
                f"{data.critical_load_kw[index]:.8f}",
                f"{data.noncritical_load_kw[index]:.8f}",
                f"{data.total_load_kw[index]:.8f}",
                f"{data.hour_sin[index]:.8f}",
                f"{data.hour_cos[index]:.8f}",
                f"{data.day_sin[index]:.8f}",
                f"{data.day_cos[index]:.8f}",
                int(data.flood_mask[index]),
                int(data.drought_mask[index]),
                data_config.seed,
            ]

    _write_csv(output_path, headers, rows())
    return _file_record(output_path, data_config.hours, headers)


def export_forecast_test_dataset(
    data: SyntheticMicrogridData,
    target_indices: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    data_config: DataConfig,
    output_path: Path,
) -> tuple[dict[str, object], np.ndarray]:
    headers = [
        "sample_index",
        "forecast_origin_hour",
        "forecast_origin_timestamp_npt",
        "target_hour_index",
        "target_timestamp_npt",
        "lead_hour",
        "actual_flow_m3s",
        "int8_qat_lstm_flow_m3s",
        "forecast_error_m3s",
        "absolute_error_m3s",
        "squared_error_m6s2",
        "absolute_percentage_error_pct",
        "persistence_flow_m3s",
        "persistence_absolute_error_m3s",
    ]
    origins = target_indices[:, 0] - 1
    persistence = data.flow_m3s[origins, None] * np.ones_like(actual)

    def rows() -> Iterable[Sequence[object]]:
        for sample in range(actual.shape[0]):
            origin = int(origins[sample])
            for lead in range(actual.shape[1]):
                target_hour = int(target_indices[sample, lead])
                error = float(predicted[sample, lead] - actual[sample, lead])
                absolute_error = abs(error)
                ape = 100.0 * absolute_error / max(float(actual[sample, lead]), 0.05)
                persistence_error = abs(
                    float(persistence[sample, lead] - actual[sample, lead])
                )
                yield [
                    sample,
                    origin,
                    _timestamp(data_config.year, origin),
                    target_hour,
                    _timestamp(data_config.year, target_hour),
                    lead + 1,
                    f"{actual[sample, lead]:.8f}",
                    f"{predicted[sample, lead]:.8f}",
                    f"{error:.8f}",
                    f"{absolute_error:.8f}",
                    f"{error * error:.10f}",
                    f"{ape:.8f}",
                    f"{persistence[sample, lead]:.8f}",
                    f"{persistence_error:.8f}",
                ]

    _write_csv(output_path, headers, rows())
    record = _file_record(output_path, actual.size, headers)
    return record, persistence


def export_leadwise_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    persistence: np.ndarray,
    output_path: Path,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    headers = [
        "lead_hour",
        "sample_count",
        "lstm_mae_m3s",
        "lstm_rmse_m3s",
        "lstm_mape_pct",
        "lstm_r2",
        "lstm_bias_m3s",
        "lstm_p90_absolute_error_m3s",
        "persistence_mae_m3s",
        "persistence_rmse_m3s",
        "persistence_mape_pct",
        "persistence_r2",
        "mae_skill_vs_persistence_pct",
    ]
    result: list[dict[str, float]] = []
    csv_rows: list[list[object]] = []
    for lead in range(actual.shape[1]):
        lstm = regression_metrics(actual[:, lead], predicted[:, lead])
        naive = regression_metrics(actual[:, lead], persistence[:, lead])
        bias = float(np.mean(predicted[:, lead] - actual[:, lead]))
        p90 = float(np.percentile(np.abs(predicted[:, lead] - actual[:, lead]), 90))
        skill = 100.0 * (naive["mae_m3s"] - lstm["mae_m3s"]) / max(
            naive["mae_m3s"], 1.0e-9
        )
        row = {
            "lead_hour": float(lead + 1),
            "sample_count": float(actual.shape[0]),
            "lstm_mae_m3s": lstm["mae_m3s"],
            "lstm_rmse_m3s": lstm["rmse_m3s"],
            "lstm_mape_pct": lstm["mape_pct"],
            "lstm_r2": lstm["r2"],
            "lstm_bias_m3s": bias,
            "lstm_p90_absolute_error_m3s": p90,
            "persistence_mae_m3s": naive["mae_m3s"],
            "persistence_rmse_m3s": naive["rmse_m3s"],
            "persistence_mape_pct": naive["mape_pct"],
            "persistence_r2": naive["r2"],
            "mae_skill_vs_persistence_pct": skill,
        }
        result.append(row)
        csv_rows.append(
            [
                lead + 1,
                actual.shape[0],
                f"{lstm['mae_m3s']:.8f}",
                f"{lstm['rmse_m3s']:.8f}",
                f"{lstm['mape_pct']:.8f}",
                f"{lstm['r2']:.8f}",
                f"{bias:.8f}",
                f"{p90:.8f}",
                f"{naive['mae_m3s']:.8f}",
                f"{naive['rmse_m3s']:.8f}",
                f"{naive['mape_pct']:.8f}",
                f"{naive['r2']:.8f}",
                f"{skill:.8f}",
            ]
        )
    _write_csv(output_path, headers, csv_rows)
    return _file_record(output_path, len(result), headers), result


def export_stress_test_datasets(
    stress: dict[str, np.ndarray],
    controlled: dict[str, np.ndarray],
    baseline_critical: np.ndarray,
    baseline_noncritical: np.ndarray,
    dispatch_path: Path,
    forecast_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    dispatch_headers = [
        "stress_hour",
        "actual_flow_m3s",
        "generation_kw",
        "critical_demand_kw",
        "noncritical_demand_kw",
        "total_demand_kw",
        "mpc_curtailment_fraction",
        "mpc_critical_served_kw",
        "mpc_noncritical_served_kw",
        "mpc_spill_kw",
        "mpc_unserved_critical_kw",
        "baseline_critical_served_kw",
        "baseline_noncritical_served_kw",
        "forecast_min_flow_next_24h_m3s",
    ]
    dispatch_rows = []
    for hour in range(len(stress["hour"])):
        dispatch_rows.append(
            [
                hour,
                f"{stress['flow_m3s'][hour]:.8f}",
                f"{controlled['generation_kw'][hour]:.8f}",
                f"{stress['critical_load_kw'][hour]:.8f}",
                f"{stress['noncritical_load_kw'][hour]:.8f}",
                f"{stress['critical_load_kw'][hour] + stress['noncritical_load_kw'][hour]:.8f}",
                f"{controlled['curtailment_fraction'][hour]:.8f}",
                f"{controlled['critical_served_kw'][hour]:.8f}",
                f"{controlled['noncritical_served_kw'][hour]:.8f}",
                f"{controlled['spill_kw'][hour]:.8f}",
                f"{controlled['unserved_critical_kw'][hour]:.8f}",
                f"{baseline_critical[hour]:.8f}",
                f"{baseline_noncritical[hour]:.8f}",
                f"{np.min(stress['rolling_forecasts_m3s'][hour]):.8f}",
            ]
        )
    _write_csv(dispatch_path, dispatch_headers, dispatch_rows)

    forecast_headers = [
        "forecast_origin_stress_hour",
        "lead_hour",
        "target_stress_hour",
        "forecast_flow_m3s",
        "target_is_padded_beyond_scenario",
    ]
    forecast_rows = []
    horizon = stress["rolling_forecasts_m3s"].shape[1]
    scenario_hours = len(stress["hour"])
    for origin in range(scenario_hours):
        for lead in range(horizon):
            raw_target = origin + lead
            forecast_rows.append(
                [
                    origin,
                    lead + 1,
                    min(raw_target, scenario_hours - 1),
                    f"{stress['rolling_forecasts_m3s'][origin, lead]:.8f}",
                    int(raw_target >= scenario_hours),
                ]
            )
    _write_csv(forecast_path, forecast_headers, forecast_rows)
    return (
        _file_record(dispatch_path, len(dispatch_rows), dispatch_headers),
        _file_record(forecast_path, len(forecast_rows), forecast_headers),
    )


def export_mpc_sensitivity(
    stress: dict[str, np.ndarray],
    output_path: Path,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    headers = [
        "reserve_margin_pct",
        "actuation_lead_hours",
        "curtailment_ramp_per_hour",
        "critical_reliability_pct",
        "critical_eens_kwh",
        "noncritical_curtailed_kwh",
        "critical_shortfall_hours",
        "first_curtailment_hour",
        "peak_curtailment_fraction",
    ]
    rows: list[list[object]] = []
    results: list[dict[str, float]] = []
    base = GridConfig()
    for reserve in (0.00, 0.05, 0.10):
        for lead in (0, 3, 6):
            cfg = replace(base, reserve_margin=reserve, actuation_lead_hours=lead)
            grid = MicroHydroGrid(cfg)
            controlled = grid.simulate(
                stress["flow_m3s"],
                stress["critical_load_kw"],
                stress["noncritical_load_kw"],
                stress["rolling_forecasts_m3s"],
                LookAheadMPC(cfg),
            )
            scores = resilience_metrics(
                stress["critical_load_kw"],
                controlled["critical_served_kw"],
                stress["noncritical_load_kw"],
                controlled["noncritical_served_kw"],
            )
            active = np.flatnonzero(controlled["curtailment_fraction"] > 1.0e-8)
            first_hour = float(active[0]) if active.size else -1.0
            record = {
                "reserve_margin_pct": reserve * 100.0,
                "actuation_lead_hours": float(lead),
                "curtailment_ramp_per_hour": cfg.curtailment_ramp_per_hour,
                **scores,
                "first_curtailment_hour": first_hour,
                "peak_curtailment_fraction": float(
                    np.max(controlled["curtailment_fraction"])
                ),
            }
            results.append(record)
            rows.append(
                [
                    f"{reserve * 100.0:.2f}",
                    lead,
                    f"{cfg.curtailment_ramp_per_hour:.2f}",
                    f"{scores['critical_reliability_pct']:.8f}",
                    f"{scores['critical_eens_kwh']:.8f}",
                    f"{scores['noncritical_curtailed_kwh']:.8f}",
                    f"{scores['critical_shortfall_hours']:.0f}",
                    f"{first_hour:.0f}",
                    f"{record['peak_curtailment_fraction']:.8f}",
                ]
            )
    _write_csv(output_path, headers, rows)
    return _file_record(output_path, len(rows), headers), results


def plot_system_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.3, 3.6, 2.0, 1.2, "Sensors /\nsynthetic twin", "#DCEAF7"),
        (3.0, 3.6, 2.0, 1.2, "72 h feature\nbuffer", "#E9E2F5"),
        (5.7, 3.6, 2.0, 1.2, "INT8 QAT-LSTM\n24 h forecast", "#E5F3E8"),
        (8.4, 3.6, 2.0, 1.2, "Look-ahead\nMPC", "#FFF0D8"),
        (8.4, 1.0, 2.0, 1.2, "Tiered load\ncontactors", "#F9DDDD"),
        (5.7, 1.0, 2.0, 1.2, "Micro-hydro\nplant", "#DCEAF7"),
        (3.0, 1.0, 2.0, 1.2, "Critical /\nflexible loads", "#E5F3E8"),
    ]
    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="#334155",
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)

    arrows = [
        ((2.3, 4.2), (3.0, 4.2)),
        ((5.0, 4.2), (5.7, 4.2)),
        ((7.7, 4.2), (8.4, 4.2)),
        ((9.4, 3.6), (9.4, 2.2)),
        ((8.4, 1.6), (7.7, 1.6)),
        ((5.7, 1.6), (5.0, 1.6)),
        ((4.0, 2.2), (4.0, 3.6)),
        ((6.7, 2.2), (6.7, 3.6)),
    ]
    for start, stop in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                stop,
                arrowstyle="-|>",
                mutation_scale=12,
                color="#475569",
                linewidth=1.2,
            )
        )
    ax.text(
        0.3,
        5.35,
        "Edge digital-twin information and control loop",
        fontsize=13,
        weight="bold",
        color="#0F172A",
    )
    ax.text(
        0.3,
        0.25,
        "Safety invariant: available generation is allocated to critical demand before admitted flexible demand.",
        fontsize=8.5,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_leadwise_performance(
    leadwise: list[dict[str, float]],
    output_path: Path,
) -> None:
    leads = np.asarray([row["lead_hour"] for row in leadwise])
    lstm_mae = np.asarray([row["lstm_mae_m3s"] for row in leadwise])
    persistence_mae = np.asarray([row["persistence_mae_m3s"] for row in leadwise])
    bias = np.asarray([row["lstm_bias_m3s"] for row in leadwise])
    p90 = np.asarray([row["lstm_p90_absolute_error_m3s"] for row in leadwise])
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.6), sharex=True)
    axes[0].plot(leads, lstm_mae, marker="o", ms=3.5, label="INT8 QAT-LSTM")
    axes[0].plot(
        leads,
        persistence_mae,
        marker="s",
        ms=3.2,
        linestyle="--",
        label="Persistence",
    )
    axes[0].set_ylabel("MAE (m$^3$ s$^{-1}$)")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Forecast performance by lead time")
    axes[1].plot(leads, bias, color="#D95F02", marker="o", ms=3.2, label="Bias")
    axes[1].plot(leads, p90, color="#1B9E77", marker="^", ms=3.2, label="90th |error|")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Forecast lead (h)")
    axes[1].set_ylabel("Error (m$^3$ s$^{-1}$)")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mpc_sensitivity(
    sensitivity: list[dict[str, float]],
    output_path: Path,
) -> None:
    reserves = [0.0, 5.0, 10.0]
    leads = [0.0, 3.0, 6.0]
    curtailed = np.zeros((len(reserves), len(leads)))
    first_hour = np.zeros_like(curtailed)
    for row in sensitivity:
        i = reserves.index(row["reserve_margin_pct"])
        j = leads.index(row["actuation_lead_hours"])
        curtailed[i, j] = row["noncritical_curtailed_kwh"]
        first_hour[i, j] = row["first_curtailment_hour"]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for ax, matrix, title, fmt in (
        (axes[0], curtailed, "Non-critical curtailed energy (kWh)", ".0f"),
        (axes[1], first_hour, "First curtailment command (hour)", ".0f"),
    ):
        image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(leads)), [str(int(x)) for x in leads])
        ax.set_yticks(range(len(reserves)), [str(int(x)) for x in reserves])
        ax.set_xlabel("Actuation lead (h)")
        ax.set_ylabel("Reserve margin (%)")
        ax.set_title(title, fontsize=10)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_submission_artifacts(root: Path) -> None:
    """Generate all CSVs and extended figures for a completed training run."""

    root = root.resolve()
    data_dir = root / "data"
    figures_dir = root / "figures"
    results_dir = root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig()
    grid_cfg = GridConfig()
    model_cfg = ModelConfig()
    data = generate_himalayan_data(data_cfg, grid_cfg)
    sequences = prepare_sequence_data(data.feature_matrix(), data.flow_m3s, model_cfg)
    tflite_model = (results_dir / "hydro_qat_lstm_int8.tflite").read_bytes()
    prediction_scaled = predict_tflite_int8(tflite_model, sequences.x_test)
    prediction_flow = sequences.target_scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).reshape(prediction_scaled.shape)
    actual_flow = sequences.target_scaler.inverse_transform(
        sequences.y_test.reshape(-1, 1)
    ).reshape(sequences.y_test.shape)

    files: list[dict[str, object]] = []
    files.append(
        export_annual_dataset(
            data,
            data_cfg,
            model_cfg,
            data_dir / "synthetic_annual_microgrid_2026.csv",
        )
    )
    forecast_record, persistence = export_forecast_test_dataset(
        data,
        sequences.target_indices_test,
        actual_flow,
        prediction_flow,
        data_cfg,
        data_dir / "forecast_test_all_leads.csv",
    )
    files.append(forecast_record)
    leadwise_record, leadwise = export_leadwise_metrics(
        actual_flow,
        prediction_flow,
        persistence,
        data_dir / "forecast_leadwise_metrics.csv",
    )
    files.append(leadwise_record)

    stress = make_stress_test(
        forecast_horizon=grid_cfg.mpc_horizon_hours, grid_config=grid_cfg
    )
    controlled = MicroHydroGrid(grid_cfg).simulate(
        stress["flow_m3s"],
        stress["critical_load_kw"],
        stress["noncritical_load_kw"],
        stress["rolling_forecasts_m3s"],
        LookAheadMPC(grid_cfg),
    )
    baseline_critical, baseline_noncritical = proportional_no_control_baseline(
        controlled["generation_kw"],
        stress["critical_load_kw"],
        stress["noncritical_load_kw"],
    )
    dispatch_record, stress_forecast_record = export_stress_test_datasets(
        stress,
        controlled,
        baseline_critical,
        baseline_noncritical,
        data_dir / "stress_test_dispatch.csv",
        data_dir / "stress_test_rolling_forecasts.csv",
    )
    files.extend([dispatch_record, stress_forecast_record])
    sensitivity_record, sensitivity = export_mpc_sensitivity(
        stress, data_dir / "mpc_sensitivity.csv"
    )
    files.append(sensitivity_record)

    plot_system_architecture(figures_dir / "figure_4_system_architecture.png")
    plot_leadwise_performance(leadwise, figures_dir / "figure_5_leadwise_metrics.png")
    plot_mpc_sensitivity(sensitivity, figures_dir / "figure_6_mpc_sensitivity.png")

    persistence_scores = regression_metrics(actual_flow, persistence)
    dataset_manifest = {
        "description": (
            "Synthetic and derived reproducibility datasets for the Himalayan "
            "microgrid digital-twin paper. These are not field observations."
        ),
        "timezone": "Asia/Kathmandu (UTC+05:45)",
        "synthetic_year": data_cfg.year,
        "generator_seed": data_cfg.seed,
        "files": files,
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2), encoding="utf-8"
    )

    metrics_path = results_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["persistence_baseline"] = persistence_scores
    metrics["leadwise_forecast"] = {
        "lead_1": leadwise[0],
        "lead_12": leadwise[11],
        "lead_24": leadwise[23],
    }
    metrics["mpc_sensitivity"] = {
        "scenario_count": len(sensitivity),
        "critical_reliability_min_pct": min(
            row["critical_reliability_pct"] for row in sensitivity
        ),
        "critical_reliability_max_pct": max(
            row["critical_reliability_pct"] for row in sensitivity
        ),
        "noncritical_curtailed_min_kwh": min(
            row["noncritical_curtailed_kwh"] for row in sensitivity
        ),
        "noncritical_curtailed_max_kwh": max(
            row["noncritical_curtailed_kwh"] for row in sensitivity
        ),
    }
    metrics["submission_datasets"] = {
        "csv_file_count": len(files),
        "total_csv_rows": int(sum(int(record["rows"]) for record in files)),
        "manifest": "data/dataset_manifest.json",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {len(files)} CSV files with {metrics['submission_datasets']['total_csv_rows']:,} rows")
    print(f"Persistence metrics: {persistence_scores}")


def main() -> None:
    build_submission_artifacts(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
