import numpy as np

from code.config import ModelConfig
from code.model import build_lstm_model, build_qat_model


def test_float_and_qat_model_output_shape():
    config = ModelConfig(lookback_hours=12, forecast_hours=4, lstm_units=4, dense_units=4)
    model = build_lstm_model(feature_count=3, config=config)
    x = np.zeros((2, 12, 3), dtype=np.float32)
    assert model(x).shape == (2, 4)
    qat_model = build_qat_model(model, config)
    assert qat_model(x).shape == (2, 4)
