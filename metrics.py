"""Forecast and microgrid resilience metrics."""

from __future__ import annotations

import numpy as np


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    mape = np.mean(np.abs(error) / np.maximum(np.abs(actual), 0.05)) * 100.0
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1.0 - np.sum(error**2) / max(denominator, 1.0e-12)
    return {
        "mae_m3s": float(mae),
        "rmse_m3s": float(rmse),
        "mape_pct": float(mape),
        "r2": float(r2),
        "bias_m3s": float(np.mean(error)),
        "p90_abs_error_m3s": float(np.percentile(np.abs(error), 90.0)),
    }


def resilience_metrics(
    critical_demand_kw: np.ndarray,
    critical_served_kw: np.ndarray,
    noncritical_demand_kw: np.ndarray,
    noncritical_served_kw: np.ndarray,
) -> dict[str, float]:
    critical_demand = np.asarray(critical_demand_kw)
    critical_served = np.asarray(critical_served_kw)
    noncritical_demand = np.asarray(noncritical_demand_kw)
    noncritical_served = np.asarray(noncritical_served_kw)
    critical_shortfall = np.maximum(critical_demand - critical_served, 0.0)
    curtailed = np.maximum(noncritical_demand - noncritical_served, 0.0)
    reliability = np.mean(critical_shortfall <= 1.0e-6) * 100.0
    return {
        "critical_reliability_pct": float(reliability),
        "critical_eens_kwh": float(np.sum(critical_shortfall)),
        "noncritical_curtailed_kwh": float(np.sum(curtailed)),
        "critical_shortfall_hours": float(np.count_nonzero(critical_shortfall > 1.0e-6)),
    }
