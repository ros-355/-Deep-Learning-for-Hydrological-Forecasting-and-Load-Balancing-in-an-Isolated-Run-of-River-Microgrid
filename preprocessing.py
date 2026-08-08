"""Chronological sequence construction and train-only normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = np.mean(values, axis=0, keepdims=True)
        scale = np.std(values, axis=0, keepdims=True)
        return cls(mean.astype(np.float32), np.maximum(scale, 1.0e-6).astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.scale + self.mean).astype(np.float32)


@dataclass(frozen=True)
class SequenceData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    target_indices_test: np.ndarray
    feature_scaler: Standardizer
    target_scaler: Standardizer


def _make_windows(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = len(features) - lookback - horizon + 1
    if sample_count <= 0:
        raise ValueError("Not enough samples for the requested lookback and forecast horizon.")
    x = np.empty((sample_count, lookback, features.shape[1]), dtype=np.float32)
    y = np.empty((sample_count, horizon), dtype=np.float32)
    indices = np.empty((sample_count, horizon), dtype=np.int32)
    for sample in range(sample_count):
        target_start = sample + lookback
        x[sample] = features[sample:target_start]
        y[sample] = target[target_start : target_start + horizon]
        indices[sample] = np.arange(target_start, target_start + horizon)
    return x, y, indices


def prepare_sequence_data(
    features: np.ndarray,
    flow_target: np.ndarray,
    config: ModelConfig | None = None,
) -> SequenceData:
    """Split chronologically, fitting both scalers on training-period values only."""

    cfg = config or ModelConfig()
    n = len(features)
    train_end = int(n * cfg.train_fraction)
    validation_end = int(n * (cfg.train_fraction + cfg.validation_fraction))

    feature_scaler = Standardizer.fit(features[:train_end])
    target_scaler = Standardizer.fit(flow_target[:train_end, None])
    scaled_features = feature_scaler.transform(features)
    scaled_target = target_scaler.transform(flow_target[:, None]).ravel()
    x, y, target_indices = _make_windows(
        scaled_features, scaled_target, cfg.lookback_hours, cfg.forecast_hours
    )

    first_targets = target_indices[:, 0]
    train_mask = target_indices[:, -1] < train_end
    validation_mask = (first_targets >= train_end) & (target_indices[:, -1] < validation_end)
    test_mask = first_targets >= validation_end

    return SequenceData(
        x_train=x[train_mask],
        y_train=y[train_mask],
        x_validation=x[validation_mask],
        y_validation=y[validation_mask],
        x_test=x[test_mask],
        y_test=y[test_mask],
        target_indices_test=target_indices[test_mask],
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )
