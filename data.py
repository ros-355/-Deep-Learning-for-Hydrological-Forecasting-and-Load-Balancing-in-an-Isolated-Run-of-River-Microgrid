"""Synthetic one-year Himalayan run-of-river and tiered load generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DataConfig, GridConfig


@dataclass(frozen=True)
class SyntheticMicrogridData:
    """Hourly signals used by the forecasting and grid-control pipeline."""

    hour_index: np.ndarray
    flow_m3s: np.ndarray
    critical_load_kw: np.ndarray
    noncritical_load_kw: np.ndarray
    available_generation_kw: np.ndarray
    hour_sin: np.ndarray
    hour_cos: np.ndarray
    day_sin: np.ndarray
    day_cos: np.ndarray
    flood_mask: np.ndarray
    drought_mask: np.ndarray

    @property
    def total_load_kw(self) -> np.ndarray:
        return self.critical_load_kw + self.noncritical_load_kw

    def feature_matrix(self) -> np.ndarray:
        """Return model inputs in a stable, documented column order."""

        return np.column_stack(
            [
                self.flow_m3s,
                self.critical_load_kw,
                self.noncritical_load_kw,
                self.hour_sin,
                self.hour_cos,
                self.day_sin,
                self.day_cos,
            ]
        ).astype(np.float32)


def hydraulic_power_kw(flow_m3s: np.ndarray, grid: GridConfig) -> np.ndarray:
    """Convert usable river flow to capped electrical power."""

    usable_flow = np.maximum(np.asarray(flow_m3s) - grid.environmental_flow_m3s, 0.0)
    raw_power = 1000.0 * 9.81 * usable_flow * grid.net_head_m * grid.turbine_efficiency
    return np.minimum(raw_power / 1000.0, grid.rated_power_kw)


def _smooth_event_pulse(length: int, peak: float, decay: float = 3.2) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length)
    rise = np.sin(np.minimum(x / 0.20, 1.0) * np.pi / 2.0) ** 2
    recession = np.exp(-decay * np.maximum(x - 0.20, 0.0))
    return peak * rise * recession


def generate_himalayan_data(
    data_config: DataConfig | None = None,
    grid_config: GridConfig | None = None,
) -> SyntheticMicrogridData:
    """Generate 8,760 hourly samples with monsoon, winter, flood, and drought dynamics.

    The generator is synthetic by design: it provides a reproducible digital-twin
    benchmark and must not be interpreted as observations from a named catchment.
    """

    cfg = data_config or DataConfig()
    grid = grid_config or GridConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.hours
    hours = np.arange(n, dtype=np.int32)
    hour_of_day = hours % 24
    day_of_year = hours / 24.0

    monsoon = cfg.monsoon_amplitude_m3s * np.exp(
        -0.5 * ((day_of_year - 225.0) / 51.0) ** 2
    )
    snowmelt = cfg.snowmelt_amplitude_m3s * np.exp(
        -0.5 * ((day_of_year - 142.0) / 37.0) ** 2
    )
    winter_suppression = 0.10 * np.exp(-0.5 * ((day_of_year - 12.0) / 38.0) ** 2)
    diurnal_melt = 0.035 * np.maximum(
        np.sin(2.0 * np.pi * (hour_of_day - 9.0) / 24.0), 0.0
    )

    innovations = rng.normal(0.0, 0.025, n)
    correlated_noise = np.empty(n, dtype=np.float64)
    correlated_noise[0] = innovations[0]
    for index in range(1, n):
        correlated_noise[index] = 0.93 * correlated_noise[index - 1] + innovations[index]

    flow = (
        cfg.base_flow_m3s
        + monsoon
        + snowmelt
        - winter_suppression
        + diurnal_melt
        + correlated_noise
    )

    flood_mask = np.zeros(n, dtype=bool)
    flood_starts = rng.choice(np.arange(30 * 24, 300 * 24), cfg.flash_flood_events, replace=False)
    for start in flood_starts:
        length = int(rng.integers(8, 31))
        stop = min(start + length, n)
        pulse = _smooth_event_pulse(stop - start, float(rng.uniform(0.35, 1.05)))
        flow[start:stop] += pulse
        flood_mask[start:stop] = True

    drought_mask = np.zeros(n, dtype=bool)
    drought_starts = rng.choice(
        np.arange(15 * 24, 345 * 24), cfg.drought_events, replace=False
    )
    for start in drought_starts:
        length = int(rng.integers(48, 145))
        stop = min(start + length, n)
        taper = np.sin(np.linspace(0.0, np.pi, stop - start)) ** 2
        severity = float(rng.uniform(0.22, 0.46))
        flow[start:stop] *= 1.0 - severity * taper
        drought_mask[start:stop] = True

    flow = np.clip(flow, 0.10, 2.20).astype(np.float32)

    clinic_telecom_base = 19.0
    critical_morning = 5.5 * np.exp(-0.5 * ((hour_of_day - 8.0) / 2.2) ** 2)
    critical_evening = 8.0 * np.exp(-0.5 * ((hour_of_day - 19.0) / 2.8) ** 2)
    winter_heating = 3.2 * (1.0 + np.cos(2.0 * np.pi * (day_of_year - 10.0) / 365.0)) / 2.0
    critical_noise = rng.normal(0.0, 0.8, n)
    critical = np.clip(
        clinic_telecom_base
        + critical_morning
        + critical_evening
        + winter_heating
        + critical_noise,
        15.0,
        None,
    )

    daylight = np.maximum(np.sin(np.pi * (hour_of_day - 6.0) / 12.0), 0.0)
    dry_season = np.clip(1.0 - monsoon / max(cfg.monsoon_amplitude_m3s, 1e-6), 0.15, 1.0)
    irrigation = 36.0 * daylight * dry_season
    water_heating = (
        16.0 * np.exp(-0.5 * ((hour_of_day - 6.5) / 1.5) ** 2)
        + 19.0 * np.exp(-0.5 * ((hour_of_day - 20.5) / 1.9) ** 2)
    )
    productive_load = 11.0 * ((hour_of_day >= 9) & (hour_of_day <= 17))
    noncritical_noise = rng.normal(0.0, 1.8, n)
    noncritical = np.clip(
        7.0 + irrigation + water_heating + productive_load + noncritical_noise,
        3.0,
        None,
    )

    hour_sin = np.sin(2.0 * np.pi * hour_of_day / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour_of_day / 24.0)
    day_sin = np.sin(2.0 * np.pi * day_of_year / 365.0)
    day_cos = np.cos(2.0 * np.pi * day_of_year / 365.0)

    return SyntheticMicrogridData(
        hour_index=hours,
        flow_m3s=flow,
        critical_load_kw=critical.astype(np.float32),
        noncritical_load_kw=noncritical.astype(np.float32),
        available_generation_kw=hydraulic_power_kw(flow, grid).astype(np.float32),
        hour_sin=hour_sin.astype(np.float32),
        hour_cos=hour_cos.astype(np.float32),
        day_sin=day_sin.astype(np.float32),
        day_cos=day_cos.astype(np.float32),
        flood_mask=flood_mask,
        drought_mask=drought_mask,
    )
