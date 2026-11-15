import math

import numpy as np
import pytest
import warp as wp

from starship.sim import CRASHED, FLYING, LANDED, Simulator
from starship.sim.catch import structure_collision, tower_step
from starship.sim.physics import Params
from starship.sim.rigid import Body, orientation
from starship.train.actions import ACTION_DIM
from starship.train.env import RocketEnv
from tests.test_catch import at_plane, catch_config


def moving_config():
    cfg = catch_config()
    cfg.catch.moving = True
    cfg.catch.flap_collisions = True
    return cfg


@wp.kernel
def geometry_cases(p: Params, result: wp.array(dtype=int)):
    b = Body()
    b.pos = wp.vec3(0.0, 0.0, 75.0)  # com=0 for these geometry-only cases: base at 75 m
    b.rot = orientation(0.0, 0.0)
    angles = wp.vec4(wp.pi / 4.0)
    opened = p.catcher.open_angle
    result[0] = int(
        structure_collision(
            p.catcher, p.aero, b, b, 0.0, angles, angles, p.hull.length, p.hull.radius, 0.0, 0.0
        )
    )
    result[1] = int(
        structure_collision(
            p.catcher,
            p.aero,
            b,
            b,
            0.0,
            angles,
            angles,
            p.hull.length,
            p.hull.radius,
            opened,
            opened,
        )
    )
    folded = wp.vec4(85.0 * wp.pi / 180.0)
    result[2] = int(
        structure_collision(
            p.catcher, p.aero, b, b, 0.0, folded, folded, p.hull.length, p.hull.radius, 0.0, 0.0
        )
    )
    # Controller must not close while the lower fins still overlap the arm height.
    result[3] = int(tower_step(p.catcher, p.aero, b, 0.0, angles, opened, 0.1) == opened)


@wp.kernel
def scripted_passage(p: Params, result: wp.array(dtype=float)):
    """An upright 2 m/s approach: geometry feasibility, not a flight-control demonstration."""
    b = Body()
    b.pos = wp.vec3(0.0, 0.0, 100.0)
    b.rot = orientation(0.0, 0.0)
    b.vel = wp.vec3(0.0, 0.0, -2.0)
    flaps = wp.vec4(wp.pi / 4.0)
    angle = p.catcher.open_angle
    collision = wp.int32(0)
    first_close_base = float(-1.0)  # noqa: UP018 - Warp dynamic loop accumulator
    for step in range(2750):
        old = b
        previous_angle = angle
        angle = tower_step(p.catcher, p.aero, b, 0.0, flaps, angle, 0.01)
        if first_close_base < 0.0 and angle < previous_angle:
            first_close_base = b.pos[2]
        b.pos = wp.vec3(0.0, 0.0, 100.0 - float(step + 1) * 0.02)
        if structure_collision(
            p.catcher,
            p.aero,
            old,
            b,
            0.0,
            flaps,
            flaps,
            p.hull.length,
            p.hull.radius,
            previous_angle,
            angle,
        ):
            collision = 1
    result[0] = float(collision)
    result[1] = angle
    result[2] = first_close_base
    result[3] = b.pos[2] + p.catcher.station


def test_fins_hit_closed_arms_but_fit_through_open_arms():
    sim = Simulator(1, moving_config(), device="cpu")
    result = wp.zeros(4, dtype=int, device="cpu")
    wp.launch(geometry_cases, dim=1, inputs=[sim.params, result], device="cpu")
    np.testing.assert_array_equal(result.numpy(), [1, 0, 0, 1])


def test_scripted_full_passage_closes_arms_without_fin_strikes():
    sim = Simulator(1, moving_config(), device="cpu")
    result = wp.zeros(4, dtype=float, device="cpu")
    wp.launch(scripted_passage, dim=1, inputs=[sim.params, result], device="cpu")
    collision, angle, close_base, pin_height = result.numpy()
    assert collision == 0
    assert angle == pytest.approx(0)
    assert 67 < close_base < 69
    assert pin_height == pytest.approx(sim.config.catch.height)


def test_closure_is_rate_limited_and_open_arms_cannot_catch():
    cfg = moving_config()
    sim = at_plane(cfg, clearance=10)
    initial = sim.snapshot().arm_angle[0]
    sim.step()
    actual = sim.snapshot().arm_angle[0]
    assert actual < initial
    assert initial - actual == pytest.approx(
        cfg.catch.angular_speed * cfg.dt * cfg.substeps, abs=1e-6
    )
    missed = at_plane(cfg)
    missed.step()
    assert missed.snapshot().status[0] == CRASHED
    caught = at_plane(cfg)
    caught.state.arm_angle.zero_()
    caught.step()
    assert caught.snapshot().status[0] == LANDED


def test_arm_state_copies_resets_and_survives_terminal_autoreset():
    cfg = moving_config()
    source = at_plane(cfg, clearance=10)
    source.step()
    copy = Simulator(1, cfg, device="cpu")
    copy.copy_from(source, [0])
    np.testing.assert_array_equal(copy.snapshot().arm_angle, source.snapshot().arm_angle)
    copy.reset()
    assert copy.snapshot().arm_angle[0] == pytest.approx(math.pi / 6)
    env = RocketEnv(1, cfg, device="cpu")
    np.testing.assert_allclose(env.observe().numpy()[:, 31], 1)
    env.sim.state.body.assign(at_plane(cfg).state.body.numpy())
    env.sim.state.arm_angle.zero_()
    step = env.step(np.zeros((1, ACTION_DIM), dtype=np.float32))
    assert env.final_status.numpy()[0] == LANDED
    assert step.final_obs.numpy()[0, 31] == 0
    assert step.obs.numpy()[0, 31] == pytest.approx(1)
    assert env.sim.terminal_snapshot().arm_angle[0] == 0
    assert env.sim.snapshot().status[0] == FLYING


def test_moving_mode_requires_fin_collisions():
    cfg = moving_config()
    cfg.catch.flap_collisions = False
    with pytest.raises(ValueError, match="flap collisions"):
        Simulator(1, cfg, device="cpu")
