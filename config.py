"""Typed configuration objects used throughout the digital-twin pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Synthetic Himalayan hydro-climatic and demand parameters."""

    seed: int = 42
    year: int = 2026
    hours: int = 365 * 24
    base_flow_m3s: float = 0.32
    monsoon_amplitude_m3s: float = 0.92
    snowmelt_amplitude_m3s: float = 0.26
    flash_flood_events: int = 10
    drought_events: int = 5


@dataclass(frozen=True)
class GridConfig:
    """Physical and control parameters for the micro-hydro plant."""

    net_head_m: float = 28.0
    turbine_efficiency: float = 0.72
    rated_power_kw: float = 120.0
    environmental_flow_m3s: float = 0.08
    reserve_margin: float = 0.05
    mpc_horizon_hours: int = 24
    actuation_lead_hours: int = 3
    curtailment_ramp_per_hour: float = 0.50


@dataclass(frozen=True)
class ModelConfig:
    """Training and TinyML export parameters."""

    lookback_hours: int = 72
    forecast_hours: int = 24
    lstm_units: int = 20
    dense_units: int = 24
    batch_size: int = 64
    epochs: int = 40
    qat_epochs: int = 10
    learning_rate: float = 1.0e-3
    qat_learning_rate: float = 2.5e-4
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    random_seed: int = 42
