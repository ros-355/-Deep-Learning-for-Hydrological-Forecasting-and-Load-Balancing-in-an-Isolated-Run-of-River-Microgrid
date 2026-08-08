import numpy as np

from code.config import DataConfig, GridConfig
from code.data import generate_himalayan_data


def test_generator_is_reproducible_and_has_expected_shape():
    cfg = DataConfig(seed=8)
    first = generate_himalayan_data(cfg, GridConfig())
    second = generate_himalayan_data(cfg, GridConfig())
    assert len(first.flow_m3s) == 8760
    assert first.feature_matrix().shape == (8760, 7)
    np.testing.assert_allclose(first.flow_m3s, second.flow_m3s)


def test_monsoon_flow_exceeds_winter_flow():
    data = generate_himalayan_data(DataConfig(seed=9), GridConfig())
    winter = data.flow_m3s[: 45 * 24].mean()
    monsoon = data.flow_m3s[190 * 24 : 270 * 24].mean()
    assert monsoon > 2.0 * winter
    assert np.all(data.critical_load_kw > 0)
    assert np.all(data.noncritical_load_kw > 0)
