import numpy as np

from code.config import ModelConfig
from code.preprocessing import prepare_sequence_data


def test_sequence_shapes_and_chronological_boundaries():
    n = 500
    features = np.column_stack([np.arange(n), np.ones(n)]).astype(np.float32)
    target = np.linspace(0.1, 1.0, n, dtype=np.float32)
    config = ModelConfig(lookback_hours=24, forecast_hours=12)
    data = prepare_sequence_data(features, target, config)
    assert data.x_train.shape[1:] == (24, 2)
    assert data.y_train.shape[1] == 12
    assert data.target_indices_test.shape[1] == 12
    assert data.target_indices_test[0, 0] >= int(n * 0.85)
