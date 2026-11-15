import math

import numpy as np
import pytest

from starship.sim import CRASHED, FLYING, LANDED, SimConfig, Simulator
from starship.sim.mass import make_mass_params, sample
from starship.sim.rigid import orientation
from starship.train.actions import ACTION_DIM
from starship.train.curriculum import Curriculum
from starship.train.env import RocketEnv
from tests.test_train import compose_cfg


def catch_config():
    cfg = SimConfig()
    cfg.catch.enabled = True
    cfg.spawn.pitch = 0
    cfg.spawn.vertical_speed = -1
    cfg.spawn.altitude = 200
    return cfg


def at_plane(cfg, clearance=0.005, offset=0.0, heading=0.0, pitch=0.0):
    sim = Simulator(1, cfg, device="cpu")
    bodies = sim.state.body.numpy()
    com = sample(make_mass_params(cfg.vehicle), sim.snapshot().tanks, device="cpu")[0, 1]
    bodies["pos"][0] = (offset, 0, cfg.catch.height - cfg.catch.station + com + clearance)
    bodies["rot"][0] = np.array(orientation(float(pitch), float(heading)))
    sim.state.body.assign(bodies)
    return sim


def test_capture_requires_both_pins_and_safe_speed():
    for speed, offset, heading, expected in (
        (-1, 0, 0, LANDED),
        (-1, 0, math.pi, LANDED),  # symmetric pin pair
        (-20, 0, 0, CRASHED),
        (-1, 2, 0, CRASHED),
        (-1, 0, math.pi / 2, CRASHED),
        (1, 0, 0, FLYING),
    ):
        cfg = catch_config()
        cfg.spawn.vertical_speed = speed
        cfg.substeps = 1
        sim = at_plane(cfg, offset=offset, heading=heading)
        sim.step()
        assert sim.snapshot().status[0] == expected


def test_catch_mode_never_rewards_ground_touchdown():
    cfg = catch_config()
    cfg.spawn.altitude = 1
    cfg.spawn.offset = 100
    sim = Simulator(1, cfg, device="cpu")
    sim.step()
    assert sim.snapshot().status[0] == CRASHED


def test_tower_collision_above_catch_plane():
    cfg = catch_config()
    sim = at_plane(cfg, clearance=40, offset=cfg.catch.tower_x)
    sim.step()
    assert sim.snapshot().status[0] == CRASHED


def test_fast_crossing_cannot_tunnel_past_catch_gate():
    cfg = catch_config()
    cfg.spawn.vertical_speed = -500
    sim = at_plane(cfg, clearance=1)
    sim.step()
    assert sim.snapshot().status[0] == CRASHED
    assert sim.snapshot().pos[0, 2] > 40


def test_catch_observations_expose_heading_and_target():
    env = RocketEnv(2, catch_config(), device="cpu")
    obs = env.observe().numpy()
    np.testing.assert_allclose(np.linalg.norm(obs[:, 28:31], axis=1), 1, atol=1e-6)
    ground = RocketEnv(2, device="cpu").observe().numpy()
    np.testing.assert_array_equal(ground[:, 28:], 0)
    result = env.step(np.zeros((2, ACTION_DIM), dtype=np.float32))
    assert np.isfinite(result.reward.numpy()).all()


def test_catch_curriculum_stays_enabled_through_entry():
    cfg = compose_cfg("curriculum=catch")
    curriculum = Curriculum(cfg, cfg.curriculum)
    for stage in range(len(curriculum.stages)):
        sim, _ = curriculum.configs(stage)
        assert sim.catch.enabled
        assert sim.catch.tolerance == 0.75
    assert sim.spawn.entry
    assert sim.spawn.altitude == 20000


def test_invalid_catch_geometry_rejected():
    cfg = catch_config()
    cfg.catch.pin_radius = 4
    with pytest.raises(ValueError, match="catch geometry"):
        Simulator(1, cfg, device="cpu")


def test_arm_collision_and_pin_velocity_are_not_com_only():
    cfg = catch_config()
    cfg.landing.max_horizontal_speed = 0.1
    cfg.substeps = 1
    for sideways, omega in ((2.0, (0, 0, 0)), (0.0, (0, 0.08, 0))):
        sim = at_plane(cfg)
        bodies = sim.state.body.numpy()
        bodies["pos"][0, 1] = sideways
        bodies["omega"][0] = omega
        sim.state.body.assign(bodies)
        sim.step()
        assert sim.snapshot().status[0] == CRASHED


def test_capture_reward_and_final_observation_survive_autoreset():
    cfg = catch_config()
    env = RocketEnv(1, cfg, device="cpu")
    fixture = at_plane(cfg)
    env.sim.state.body.assign(fixture.state.body.numpy())
    result = env.step(np.zeros((1, ACTION_DIM), dtype=np.float32))
    assert env.final_status.numpy()[0] == LANDED
    assert result.terminal_reward.numpy()[0] == pytest.approx(150)
    assert result.terminated.numpy()[0] == 1
    assert np.isfinite(result.final_obs.numpy()).all()
    assert env.final_catch_error.numpy()[0] < 0.01
    assert env.sim.snapshot().status[0] == FLYING
    assert env.sim.snapshot().pos[0, 2] == pytest.approx(cfg.spawn.altitude)


def test_catch_curriculum_has_no_cheap_structural_failure_floor():
    cfg = compose_cfg("curriculum=catch")
    curriculum = Curriculum(cfg, cfg.curriculum)
    for stage in range(len(curriculum.stages)):
        _, reward = curriculum.configs(stage)
        assert reward.crash_penalty == reward.tumble_penalty == 200
        assert reward.timeout_penalty == reward.out_of_bounds_penalty == 200
    # Other tasks retain their own shaping; this experiment changes catch failure ordering.
    ground = compose_cfg("curriculum=landing")
    assert Curriculum(ground, ground.curriculum).configs(0)[1].crash_penalty == 20
    sim_cfg, reward = curriculum.configs(0)
    env = RocketEnv(1, sim_cfg, reward_config=reward, device="cpu")
    fixture_cfg = catch_config()
    fixture_cfg.catch.moving = True
    fixture_cfg.catch.policy_controlled = True
    fixture_cfg.catch.flap_collisions = True
    fixture = at_plane(fixture_cfg)
    env.sim.copy_from(fixture, [0])
    env.step(np.zeros((1, ACTION_DIM), dtype=np.float32))
    assert env.sim.terminal_snapshot().catch_failure[0] == 7
    assert env.terminal_reward.numpy()[0] <= -200
