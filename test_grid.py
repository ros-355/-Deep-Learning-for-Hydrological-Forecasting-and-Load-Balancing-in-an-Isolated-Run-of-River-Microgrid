import numpy as np

from code.config import GridConfig
from code.grid import LookAheadMPC, MicroHydroGrid
from code.metrics import resilience_metrics
from code.stress_test import make_stress_test


def test_mpc_preserves_critical_load_in_stress_case():
    config = GridConfig()
    stress = make_stress_test(grid_config=config)
    result = MicroHydroGrid(config).simulate(
        stress["flow_m3s"],
        stress["critical_load_kw"],
        stress["noncritical_load_kw"],
        stress["rolling_forecasts_m3s"],
        LookAheadMPC(config),
    )
    metrics = resilience_metrics(
        stress["critical_load_kw"],
        result["critical_served_kw"],
        stress["noncritical_load_kw"],
        result["noncritical_served_kw"],
    )
    assert metrics["critical_reliability_pct"] == 100.0
    assert np.max(result["curtailment_fraction"]) > 0.20


def test_curtailment_is_commanded_before_flow_nadir():
    config = GridConfig()
    stress = make_stress_test(grid_config=config)
    result = MicroHydroGrid(config).simulate(
        stress["flow_m3s"],
        stress["critical_load_kw"],
        stress["noncritical_load_kw"],
        stress["rolling_forecasts_m3s"],
        LookAheadMPC(config),
    )
    first_command = int(np.flatnonzero(result["curtailment_fraction"] > 0.01)[0])
    nadir = int(np.argmin(stress["flow_m3s"]))
    assert first_command < nadir
