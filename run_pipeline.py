"""End-to-end experiment entry point for the Himalayan microgrid digital twin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from code.config import DataConfig, GridConfig, ModelConfig
from code.data import generate_himalayan_data
from code.grid import LookAheadMPC, MicroHydroGrid, proportional_no_control_baseline
from code.metrics import regression_metrics, resilience_metrics
from code.model import (
    build_lstm_model,
    build_fixed_batch_qat_clone,
    build_qat_model,
    export_full_int8_tflite,
    fine_tune_qat,
    predict_tflite_int8,
    print_memory_report,
    tflite_memory_report,
    train_float_model,
)
from code.preprocessing import prepare_sequence_data
from code.stress_test import make_stress_test
from code.submission_artifacts import build_submission_artifacts
from code.visualization import (
    plot_annual_synthetic_profile,
    plot_forecast_accuracy,
    plot_stress_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root folder receiving models, metrics, and figures.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--qat-epochs", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig()
    grid_cfg = GridConfig()
    default_model_cfg = ModelConfig()
    model_cfg = ModelConfig(
        **{
            **default_model_cfg.__dict__,
            "epochs": args.epochs or default_model_cfg.epochs,
            "qat_epochs": args.qat_epochs or default_model_cfg.qat_epochs,
        }
    )
    data = generate_himalayan_data(data_cfg, grid_cfg)
    sequences = prepare_sequence_data(data.feature_matrix(), data.flow_m3s, model_cfg)

    model = build_lstm_model(data.feature_matrix().shape[1], model_cfg)
    float_history = train_float_model(
        model,
        sequences.x_train,
        sequences.y_train,
        sequences.x_validation,
        sequences.y_validation,
        model_cfg,
        verbose=0 if args.quiet else 2,
    )
    qat_model = build_qat_model(model, model_cfg)
    qat_history = fine_tune_qat(
        qat_model,
        sequences.x_train,
        sequences.y_train,
        sequences.x_validation,
        sequences.y_validation,
        model_cfg,
        verbose=0 if args.quiet else 2,
    )

    tflite_path = results_dir / "hydro_qat_lstm_int8.tflite"
    export_model = build_fixed_batch_qat_clone(
        qat_model, data.feature_matrix().shape[1], model_cfg
    )
    tflite_model = export_full_int8_tflite(
        export_model, sequences.x_train, tflite_path
    )
    memory = tflite_memory_report(tflite_model)
    print_memory_report(memory)

    prediction_scaled = predict_tflite_int8(tflite_model, sequences.x_test)
    prediction_flow = sequences.target_scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).reshape(prediction_scaled.shape)
    actual_flow = sequences.target_scaler.inverse_transform(
        sequences.y_test.reshape(-1, 1)
    ).reshape(sequences.y_test.shape)
    forecast_scores = regression_metrics(actual_flow, prediction_flow)

    lead_index = model_cfg.forecast_hours - 1
    lead_hours = sequences.target_indices_test[:, lead_index]
    plot_forecast_accuracy(
        lead_hours,
        data.flow_m3s[lead_hours],
        prediction_flow[:, lead_index],
        figures_dir / "figure_2_forecast_accuracy.png",
    )
    plot_annual_synthetic_profile(
        np.arange(365),
        data.flow_m3s,
        data.critical_load_kw,
        data.noncritical_load_kw,
        figures_dir / "figure_1_synthetic_profile.png",
    )

    stress = make_stress_test(
        forecast_horizon=grid_cfg.mpc_horizon_hours, grid_config=grid_cfg
    )
    grid = MicroHydroGrid(grid_cfg)
    controller = LookAheadMPC(grid_cfg)
    controlled = grid.simulate(
        stress["flow_m3s"],
        stress["critical_load_kw"],
        stress["noncritical_load_kw"],
        stress["rolling_forecasts_m3s"],
        controller,
    )
    baseline_critical, baseline_noncritical = proportional_no_control_baseline(
        controlled["generation_kw"],
        stress["critical_load_kw"],
        stress["noncritical_load_kw"],
    )
    controlled_scores = resilience_metrics(
        stress["critical_load_kw"],
        controlled["critical_served_kw"],
        stress["noncritical_load_kw"],
        controlled["noncritical_served_kw"],
    )
    baseline_scores = resilience_metrics(
        stress["critical_load_kw"],
        baseline_critical,
        stress["noncritical_load_kw"],
        baseline_noncritical,
    )
    controlled_scores["blackout_hours_avoided"] = (
        baseline_scores["critical_shortfall_hours"]
        - controlled_scores["critical_shortfall_hours"]
    )

    plot_stress_test(
        stress["hour"],
        stress["flow_m3s"],
        controlled["generation_kw"],
        stress["critical_load_kw"],
        stress["noncritical_load_kw"],
        controlled["critical_served_kw"],
        baseline_critical,
        controlled["noncritical_served_kw"],
        controlled["curtailment_fraction"],
        figures_dir / "figure_3_mpc_stress_test.png",
    )

    metrics = {
        "dataset": {
            "hours": int(data_cfg.hours),
            "flood_hours": int(np.sum(data.flood_mask)),
            "drought_hours": int(np.sum(data.drought_mask)),
            "flow_min_m3s": float(np.min(data.flow_m3s)),
            "flow_max_m3s": float(np.max(data.flow_m3s)),
            "mean_critical_load_kw": float(np.mean(data.critical_load_kw)),
            "mean_noncritical_load_kw": float(np.mean(data.noncritical_load_kw)),
        },
        "model": {
            "parameter_count_float": int(model.count_params()),
            "lookback_hours": model_cfg.lookback_hours,
            "forecast_hours": model_cfg.forecast_hours,
            "float_final_val_loss": float(float_history.history["val_loss"][-1]),
            "qat_final_val_loss": float(qat_history.history["val_loss"][-1]),
            **forecast_scores,
            **memory,
        },
        "stress_test_with_mpc": controlled_scores,
        "stress_test_without_control": baseline_scores,
    }
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        results_dir / "stress_test_results.npz",
        **stress,
        **{f"controlled_{key}": value for key, value in controlled.items()},
        baseline_critical_served_kw=baseline_critical,
        baseline_noncritical_served_kw=baseline_noncritical,
    )
    print(f"\nForecast metrics: {forecast_scores}")
    print(f"MPC stress-test metrics: {controlled_scores}")
    print(f"Saved metrics to {metrics_path}")
    build_submission_artifacts(root)


if __name__ == "__main__":
    main()
