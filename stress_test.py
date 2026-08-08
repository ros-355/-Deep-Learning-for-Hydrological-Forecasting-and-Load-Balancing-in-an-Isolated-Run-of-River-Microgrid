"""Deterministic flow-collapse scenario for controller validation."""

from __future__ import annotations

import numpy as np

from .config import GridConfig


def make_stress_test(
    hours: int = 96,
    forecast_horizon: int = 24,
    seed: int = 77,
    grid_config: GridConfig | None = None,
) -> dict[str, np.ndarray]:
    del grid_config
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    flow = np.full(hours, 0.78, dtype=np.float64)
    drop_start = 42
    drop_end = 50
    flow[drop_start:drop_end] = np.linspace(0.78, 0.27, drop_end - drop_start)
    flow[drop_end:72] = 0.27
    flow[72:82] = np.linspace(0.27, 0.62, 10)
    flow[82:] = 0.62
    flow += rng.normal(0.0, 0.008, hours)
    flow = np.clip(flow, 0.12, None)

    critical = 24.0 + 4.5 * np.exp(-0.5 * (((t % 24) - 19.0) / 2.8) ** 2)
    noncritical = (
        42.0
        + 16.0 * ((t % 24 >= 8) & (t % 24 <= 18))
        + 10.0 * np.exp(-0.5 * (((t % 24) - 20.0) / 2.0) ** 2)
    )

    forecasts = np.empty((hours, forecast_horizon), dtype=np.float32)
    for hour in range(hours):
        stop = min(hour + forecast_horizon, hours)
        known = flow[hour:stop]
        if len(known) < forecast_horizon:
            known = np.pad(known, (0, forecast_horizon - len(known)), mode="edge")
        forecast_error = rng.normal(0.0, 0.006, forecast_horizon)
        forecasts[hour] = np.maximum(known + forecast_error, 0.10)

    return {
        "hour": t,
        "flow_m3s": flow.astype(np.float32),
        "critical_load_kw": critical.astype(np.float32),
        "noncritical_load_kw": noncritical.astype(np.float32),
        "rolling_forecasts_m3s": forecasts,
        "drop_start": np.asarray(drop_start),
    }
