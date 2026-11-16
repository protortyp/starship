import numpy as np
import pytest

from starship.sim import CRASHED, FLYING, LANDED, SimConfig, Simulator
from starship.sim.mass import make_mass_params, sample
from starship.train.landing_metrics import LandingMetrics


@pytest.mark.parametrize("failure", [None, "speed", "horizontal", "tilt", "rate", "miss"])
def test_ground_measurements_agree_with_actual_touchdown(failure):
    cfg = SimConfig()
    cfg.spawn.pitch = 0.2 if failure == "tilt" else 0.0
    cfg.spawn.vertical_speed = -10 if failure == "speed" else -1
    sim = Simulator(1, cfg, device="cpu", auto_reset=True)
    com = sample(make_mass_params(cfg.vehicle), sim.snapshot().tanks, device="cpu")[0, 1]
    body = sim.state.body.numpy()
    body["pos"][0] = (30 if failure == "miss" else 0, 0, float(com) + 0.05)
    if failure == "horizontal":
        body["vel"][0, 0] = 5
    if failure == "rate":
        body["omega"][0, 0] = 0.5
    sim.state.body.assign(body)
    for _ in range(50):
        sim.step()
        if sim.terminal_snapshot().done[0]:
            break
    assert sim.terminal_snapshot().status[0] == (LANDED if failure is None else CRASHED)
    assert sim.snapshot().status[0] == FLYING  # The meter must use terminal, not reset state.
    excess = LandingMetrics(sim).excess(LandingMetrics(sim).sample())[0]
    if failure is None:
        assert (excess <= 0).all()
    else:
        index = {"speed": 0, "horizontal": 1, "tilt": 2, "rate": 3, "miss": 4}[failure]
        assert excess[index] > 0
    assert np.isfinite(excess).all()
