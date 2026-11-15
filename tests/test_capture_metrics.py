import numpy as np
import pytest

from starship.sim import CRASHED, LANDED
from starship.train.capture_metrics import CaptureMetrics
from tests.test_catch import at_plane, catch_config


@pytest.mark.parametrize(
    ("speed", "offset", "expected"), [(-1, 0, LANDED), (-20, 0, CRASHED), (-1, 2, CRASHED)]
)
def test_terminal_capture_measurements(speed, offset, expected):
    cfg = catch_config()
    cfg.spawn.vertical_speed = speed
    cfg.substeps = 1
    sim = at_plane(cfg, offset=offset)
    sim.step()
    assert sim.snapshot().status[0] == expected
    meter = CaptureMetrics(sim)
    measurements = meter.sample()
    residuals = meter.residuals(measurements)[0]
    if expected == LANDED:
        assert (residuals <= 1e-5).all()
    elif speed == -20:
        assert residuals[2] > 0
    else:
        assert residuals[0] > 0


def test_pin_speeds_include_rotation_even_when_com_is_slow():
    cfg = catch_config()
    cfg.substeps = 1
    sim = at_plane(cfg)
    sim.step()
    bodies = sim.terminal.body.numpy()
    bodies["vel"][0] = (0, 0, -0.1)
    bodies["omega"][0] = (0, 0, 0.2)
    sim.terminal.body.assign(bodies)
    meter = CaptureMetrics(sim)
    row = meter.sample()[0]
    # Rotation about body axis 3 makes one pin rise and the other fall.
    assert row[2] > 0.1
    assert row[3] > 0
    np.testing.assert_allclose(row[2] - row[3], 0.2, atol=1e-5)
    assert meter.residuals(row)[3] > 0
