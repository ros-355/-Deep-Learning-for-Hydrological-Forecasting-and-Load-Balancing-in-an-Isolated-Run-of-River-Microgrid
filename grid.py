"""Micro-hydro plant model and tier-aware look-ahead MPC controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GridConfig
from .data import hydraulic_power_kw


@dataclass(frozen=True)
class GridStep:
    generation_kw: float
    critical_demand_kw: float
    noncritical_demand_kw: float
    critical_served_kw: float
    noncritical_served_kw: float
    curtailment_fraction: float
    spill_kw: float
    unserved_critical_kw: float


class LookAheadMPC:
    """A low-cost horizon controller suitable for execution beside a TinyML model."""

    def __init__(self, config: GridConfig | None = None) -> None:
        self.config = config or GridConfig()
        self.command = 0.0

    def reset(self) -> None:
        self.command = 0.0

    def compute_curtailment(
        self,
        forecast_flow_m3s: np.ndarray,
        forecast_critical_kw: np.ndarray,
        forecast_noncritical_kw: np.ndarray,
    ) -> float:
        """Return the present non-critical curtailment command in [0, 1].

        The optimization is a closed-form solution to a linear priority dispatch:
        minimize curtailed non-critical power subject to generation, reserve, and
        critical-load constraints. A three-hour actuation window intentionally
        advances the command ahead of the predicted deficit.
        """

        cfg = self.config
        horizon = min(
            cfg.mpc_horizon_hours,
            len(forecast_flow_m3s),
            len(forecast_critical_kw),
            len(forecast_noncritical_kw),
        )
        generation = hydraulic_power_kw(np.asarray(forecast_flow_m3s[:horizon]), cfg)
        critical = np.asarray(forecast_critical_kw[:horizon], dtype=np.float64)
        noncritical = np.asarray(forecast_noncritical_kw[:horizon], dtype=np.float64)
        dispatchable = generation * (1.0 - cfg.reserve_margin)
        total = critical + noncritical
        deficit_positions = np.flatnonzero(dispatchable + 1.0e-9 < total)

        target = 0.0
        if deficit_positions.size:
            first = int(deficit_positions[0])
            if first <= cfg.actuation_lead_hours:
                risk_stop = min(first + cfg.actuation_lead_hours + 1, horizon)
                required = np.clip(
                    (total[:risk_stop] - dispatchable[:risk_stop])
                    / np.maximum(noncritical[:risk_stop], 1.0e-6),
                    0.0,
                    1.0,
                )
                target = float(np.max(required))

        delta = np.clip(
            target - self.command,
            -cfg.curtailment_ramp_per_hour,
            cfg.curtailment_ramp_per_hour,
        )
        self.command = float(np.clip(self.command + delta, 0.0, 1.0))
        return self.command


class MicroHydroGrid:
    """Single-bus isolated grid with explicit critical-load priority."""

    def __init__(self, config: GridConfig | None = None) -> None:
        self.config = config or GridConfig()

    def step(
        self,
        flow_m3s: float,
        critical_demand_kw: float,
        noncritical_demand_kw: float,
        curtailment_fraction: float,
    ) -> GridStep:
        generation = float(hydraulic_power_kw(np.asarray([flow_m3s]), self.config)[0])
        curtailed_request = noncritical_demand_kw * (1.0 - curtailment_fraction)
        critical_served = min(critical_demand_kw, generation)
        remaining = max(generation - critical_served, 0.0)
        noncritical_served = min(curtailed_request, remaining)
        spill = max(generation - critical_served - noncritical_served, 0.0)
        return GridStep(
            generation_kw=generation,
            critical_demand_kw=float(critical_demand_kw),
            noncritical_demand_kw=float(noncritical_demand_kw),
            critical_served_kw=float(critical_served),
            noncritical_served_kw=float(noncritical_served),
            curtailment_fraction=float(curtailment_fraction),
            spill_kw=float(spill),
            unserved_critical_kw=float(max(critical_demand_kw - critical_served, 0.0)),
        )

    def simulate(
        self,
        actual_flow_m3s: np.ndarray,
        critical_load_kw: np.ndarray,
        noncritical_load_kw: np.ndarray,
        rolling_flow_forecasts_m3s: np.ndarray,
        controller: LookAheadMPC,
    ) -> dict[str, np.ndarray]:
        n = len(actual_flow_m3s)
        result = {
            "generation_kw": np.zeros(n),
            "critical_served_kw": np.zeros(n),
            "noncritical_served_kw": np.zeros(n),
            "curtailment_fraction": np.zeros(n),
            "spill_kw": np.zeros(n),
            "unserved_critical_kw": np.zeros(n),
        }
        controller.reset()
        for hour in range(n):
            forecast = rolling_flow_forecasts_m3s[hour]
            horizon = len(forecast)
            demand_stop = min(hour + horizon, n)
            critical_forecast = critical_load_kw[hour:demand_stop]
            noncritical_forecast = noncritical_load_kw[hour:demand_stop]
            if len(critical_forecast) < horizon:
                pad = horizon - len(critical_forecast)
                critical_forecast = np.pad(critical_forecast, (0, pad), mode="edge")
                noncritical_forecast = np.pad(noncritical_forecast, (0, pad), mode="edge")
            command = controller.compute_curtailment(
                forecast, critical_forecast, noncritical_forecast
            )
            step = self.step(
                actual_flow_m3s[hour],
                critical_load_kw[hour],
                noncritical_load_kw[hour],
                command,
            )
            for key in result:
                result[key][hour] = getattr(step, key)
        return result


def proportional_no_control_baseline(
    generation_kw: np.ndarray,
    critical_load_kw: np.ndarray,
    noncritical_load_kw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent an uncoordinated feeder where shortages affect both load tiers."""

    total = np.asarray(critical_load_kw) + np.asarray(noncritical_load_kw)
    ratio = np.minimum(np.asarray(generation_kw) / np.maximum(total, 1.0e-6), 1.0)
    return np.asarray(critical_load_kw) * ratio, np.asarray(noncritical_load_kw) * ratio
