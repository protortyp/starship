import math

import numpy as np
import pytest

from starship.sim.config import G0, VehicleConfig
from starship.sim.propulsion import (
    engine_eval,
    engine_rollout,
    engine_state,
    flap_rollout,
    make_propulsion_params,
    rcs_and_draw,
)

CFG = VehicleConfig()
DT = 0.01
P = make_propulsion_params(CFG, DT)
COM = 27.0
DEV = "cpu"


def test_thrust_sea_level_and_vacuum():
    force, torque, _ = engine_eval(P, engine_state(spool=1.0), 101325.0, COM, DEV)
    assert force[0] / 3 == pytest.approx(2.255e6, rel=2e-3)
    assert abs(force[1]) + abs(force[2]) < 1.0
    assert np.abs(torque).max() < 10.0  # symmetric ring, thrust through the axis
    force, _, _ = engine_eval(P, engine_state(spool=1.0), 0.0, COM, DEV)
    assert force[0] / 3 == pytest.approx(2.39e6, rel=1e-6)


def test_mass_flow_and_floor():
    _, _, mdot = engine_eval(P, engine_state(spool=1.0), 101325.0, COM, DEV)
    assert mdot / 3 == pytest.approx(2.39e6 / (347.0 * G0), rel=1e-5)
    assert mdot / 3 == pytest.approx(702.3, abs=0.5)
    # A throttle so low that the ambient-pressure loss exceeds it gives zero, not negative, thrust.
    force, _, _ = engine_eval(P, engine_state(spool=1.0, throttle=0.05), 101325.0, COM, DEV)
    assert force[0] == 0.0


def test_spool_up_and_down():
    steps = round(CFG.engine.ignition_delay / DT)
    r = engine_rollout(P, engine_state(), 1.0, 1.0, 0.0, 0.0, steps + 5, device=DEV)
    assert r["spool"][steps // 2 - 1, 0] == pytest.approx(0.5, abs=1e-3)
    assert r["spool"][steps - 1, 0] == pytest.approx(1.0, abs=1e-4)
    assert r["spool"][-1, 0] == 1.0
    off = engine_rollout(P, engine_state(spool=1.0), 0.0, 1.0, 0.0, 0.0, 40, device=DEV)
    assert off["spool"][round(CFG.engine.shutdown_time / DT) - 1, 0] == pytest.approx(0.0, abs=1e-4)
    dry = engine_rollout(
        P, engine_state(), 1.0, 1.0, 0.0, 0.0, 50, has_propellant=False, device=DEV
    )
    assert dry["spool"].max() == 0.0


def test_throttle_clamps_and_slews():
    r = engine_rollout(P, engine_state(throttle=1.0), 1.0, 0.0, 0.0, 0.0, 300, device=DEV)
    rate = CFG.engine.throttle_rate
    assert r["throttle"][9, 0] == pytest.approx(1.0 - 10 * rate * DT, abs=1e-5)
    assert r["throttle"][-1, 0] == pytest.approx(CFG.engine.min_throttle, abs=1e-6)
    r = engine_rollout(P, engine_state(throttle=0.9), 1.0, 2.0, 0.0, 0.0, 100, device=DEV)
    assert r["throttle"][-1, 0] == pytest.approx(1.0, abs=1e-6)


def test_gimbal_lag_rate_and_limit():
    e = CFG.engine
    # Small step: the lag dominates, ~63 % of the command after one time constant.
    cmd = 0.1 * e.gimbal_rate * e.gimbal_lag
    r = engine_rollout(P, engine_state(), 1.0, 1.0, cmd, 0.0, 200, device=DEV)
    assert r["gimbal2"][round(e.gimbal_lag / DT) - 1, 0] == pytest.approx(
        cmd * (1 - math.exp(-1)), rel=1e-3
    )
    # Large step: the rate limit dominates, then the angle saturates at the limit.
    r = engine_rollout(P, engine_state(), 1.0, 1.0, 1.0, -1.0, 200, device=DEV)
    rate = np.diff(r["gimbal2"][:, 0]) / DT
    assert rate.max() <= e.gimbal_rate * (1 + 1e-4)
    assert r["gimbal2"][19, 0] == pytest.approx(20 * e.gimbal_rate * DT, rel=1e-4)
    assert r["gimbal2"][-1, 0] == pytest.approx(e.gimbal_limit, rel=1e-6)
    assert r["gimbal3"][-1, 0] == pytest.approx(-e.gimbal_limit, rel=1e-6)
    assert np.abs(r["gimbal2"]).max() <= e.gimbal_limit * (1 + 1e-6)


def test_gimbal_torque_sign():
    # Thrust tilted towards +b2 at the tail (r ~ -L b1): r x F points along -b3, which rotates
    # b1 towards -b2, i.e. the nose swings to port.
    g = math.radians(5.0)
    force, torque, _ = engine_eval(P, engine_state(spool=1.0, gimbal2=g), 0.0, COM, DEV)
    assert force[1] == pytest.approx(force[0] * math.tan(g), rel=1e-5)
    lever = np.array([CFG.engine.pivot_station - COM, 0.0, 0.0])
    expected = np.cross(lever, force)
    assert expected[2] < 0.0
    assert torque[2] == pytest.approx(expected[2], rel=1e-4)
    assert abs(torque[0]) < 1e-3 * abs(torque[2])
    # A small rotation about -b3 maps b1 to b1 - eps b2: the nose moves towards -b2.
    axis = torque / np.linalg.norm(torque)
    assert np.cross(axis, [1.0, 0.0, 0.0])[1] < 0.0
    # Tilt towards +b3 gives a positive torque about b2 (nose towards -b3).
    _, torque, _ = engine_eval(P, engine_state(spool=1.0, gimbal3=g), 0.0, COM, DEV)
    assert torque[1] > 0.0
    assert abs(torque[2]) < 1e-3 * abs(torque[1])


def test_flap_actuator():
    f = CFG.flap_actuator
    angle, filtered = flap_rollout(P, 0.0, 2.0, 600, device=DEV)
    assert np.diff(angle[:, 0]).max() <= f.rate * DT * (1 + 1e-4)
    assert angle[-1] == pytest.approx([f.max_angle] * 4, rel=1e-6)
    assert filtered[-1] == pytest.approx([f.max_angle] * 4, rel=1e-4)
    angle, _ = flap_rollout(P, f.max_angle, -1.0, 600, device=DEV)
    assert angle[-1] == pytest.approx([f.min_angle] * 4, abs=1e-6)


def test_rcs_roll_torque():
    r = CFG.rcs
    full = r.thrust * r.arm * r.pairs
    torque, _ = rcs_and_draw(P, 0.5, [0.0] * 4, 0.0, DT, DEV)
    assert torque == pytest.approx([0.5 * full, 0.0, 0.0])
    torque, _ = rcs_and_draw(P, -3.0, [0.0] * 4, 0.0, DT, DEV)
    assert torque == pytest.approx([-full, 0.0, 0.0])


def test_propellant_split_header_first():
    mdot, dt = 460.0, 1.0
    before = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    _, after = rcs_and_draw(P, 0.0, before, mdot, dt, DEV)
    used = before - after
    assert used[:2] == pytest.approx([0.0, 0.0])
    assert used[2] / used[3] == pytest.approx(3.6, rel=1e-5)
    assert used.sum() == pytest.approx(mdot * dt, rel=1e-6)
    # Header LOX nearly empty: the remainder comes from the main tank, nothing goes negative.
    _, after = rcs_and_draw(P, 0.0, [1000.0, 1000.0, 50.0, 1000.0], mdot, dt, DEV)
    assert after[2] == 0.0
    assert after[0] == pytest.approx(1000.0 - (360.0 - 50.0), rel=1e-6)
    assert after[3] == pytest.approx(900.0, rel=1e-6)
    _, after = rcs_and_draw(P, 0.0, [10.0, 0.0, 0.0, 0.0], mdot, dt, DEV)
    assert after == pytest.approx([0.0, 0.0, 0.0, 0.0])
