import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import warp as wp
from tensordict import TensorDict

from starship.sim import SimConfig, Simulator
from starship.sim.atmosphere import sample as sample_air
from starship.train import ObsConfig, RewardConfig, RocketEnv
from starship.train.checkpoint import load_policy
from starship.train.curriculum import Curriculum
from starship.train.encoding import LEGACY_OBS_DIM, OBS_DIM
from starship.train.runner import export_policy
from tests import test_train


def entry_config(**overrides):
    cfg = SimConfig(max_time=240)
    cfg.spawn = replace(cfg.spawn, entry=True, altitude=20000, **overrides)
    return cfg


@pytest.mark.parametrize("heading", [-0.2, 0, 0.2])
def test_entry_kinematics_and_belly_orientation(heading):
    cfg = entry_config(heading=heading, speed_noise=10, flight_path_angle_noise=0.05)
    sim = Simulator(128, cfg, device="cpu", seed=42)
    snap = sim.snapshot()
    speed = np.linalg.norm(snap.vel, axis=1)
    angle = np.arcsin(-snap.vel[:, 2] / speed)
    assert np.all((speed >= 270) & (speed <= 290))
    assert np.ptp(speed) > 18
    assert np.max(np.abs(angle - cfg.spawn.flight_path_angle)) <= 0.050001
    bearing = np.arctan2(-snap.pos[:, 1], -snap.pos[:, 0])
    drift = np.arctan2(snap.vel[:, 1], snap.vel[:, 0])
    delta = np.arctan2(np.sin(drift - bearing), np.cos(drift - bearing))
    np.testing.assert_allclose(delta, heading, atol=5e-7)
    for q, velocity in zip(snap.rot, snap.vel, strict=True):
        body_velocity = np.array(wp.quat_rotate_inv(wp.quat(*q), wp.vec3(*velocity)))
        np.testing.assert_allclose(body_velocity[:2], 0, atol=1e-4)
        assert body_velocity[2] > 0


def test_entry_noise_and_rng_are_reproducible():
    cfg = entry_config(heading_noise=0.05, offset_noise=100, pitch_noise=0.04)
    a, b = (Simulator(64, cfg, device="cpu", seed=17) for _ in range(2))
    first = a.snapshot().pos.copy()
    for _ in range(2):
        a.reset()
        b.reset()
        for name in ("pos", "vel", "rot", "omega"):
            np.testing.assert_array_equal(getattr(a.snapshot(), name), getattr(b.snapshot(), name))
    assert not np.array_equal(first, a.snapshot().pos)
    snap = a.snapshot()
    drift = np.arctan2(snap.vel[:, 1], snap.vel[:, 0])
    bearing = np.arctan2(-snap.pos[:, 1], -snap.pos[:, 0])
    error = np.arctan2(np.sin(drift - bearing), np.cos(drift - bearing))
    assert np.max(np.abs(error)) < 0.1
    assert np.std(error) > 0.02


@pytest.mark.parametrize("stage", [10, 11, 12, 13])
def test_entry_placement_reaches_flip_corridor_in_reference_glide(stage):
    cfg = test_train.compose_cfg("curriculum=landing")
    sim_cfg, _ = Curriculum(cfg, cfg.curriculum).configs(stage)
    sim_cfg.spawn = replace(
        sim_cfg.spawn,
        altitude_noise=0,
        speed_noise=0,
        flight_path_angle_noise=0,
        heading_noise=0,
        offset_noise=0,
    )
    sim = Simulator(1, sim_cfg, device="cpu")
    snap = sim.snapshot()
    start_range = np.linalg.norm(snap.pos[0, :2])
    assert 0 < start_range < sim_cfg.max_offset
    # Independent fine-step integration checks the cheap spawn footprint against its reference.
    heights = np.linspace(0, 21000, 421)
    densities = sample_air(heights, device="cpu")[:, 2]
    mass = sim_cfg.vehicle.mass.dry_mass + snap.tanks[0].sum()
    pos, vel = snap.pos[0].astype(float), snap.vel[0].astype(float)
    dt, elapsed = 0.02, 0.0
    while pos[2] > 300 and elapsed < 240:
        density = np.interp(pos[2], heights, densities)
        drag = 0.5 * density * sim_cfg.spawn.entry_drag_area * np.linalg.norm(vel) / mass
        acceleration = -drag * vel
        acceleration[2] -= 9.80665 * (6356766 / (6356766 + pos[2])) ** 2
        vel += acceleration * dt
        pos += vel * dt
        elapsed += dt
    assert pos[2] <= 300
    assert np.linalg.norm(pos[:2]) < 0.025 * start_range
    assert np.linalg.norm(vel[:2]) < 15


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speed", -1),
        ("speed_noise", 300),
        ("flight_path_angle", 0),
        ("flight_path_angle_noise", 2),
        ("entry_drag_area", 0),
        ("heading_noise", -1),
        ("speed", float("nan")),
    ],
)
def test_invalid_entry_parameters(field, value):
    with pytest.raises(ValueError, match="entry spawn"):
        Simulator(1, entry_config(**{field: value}), device="cpu")


def test_bounded_potential_values_and_informative_gradient():
    env = RocketEnv(1, reward_config=RewardConfig(distance_saturation=1000), device="cpu")
    # Evaluate through the production reward path, including the pre-step potential snapshot.
    values = []
    for altitude in (0, 1, 1000, 20000, 21000):
        bodies = env.sim.state.body.numpy()
        bodies["pos"] = (0, 0, altitude)
        bodies["vel"] = 0
        env.sim.state.body.assign(bodies)
        env.step(np.zeros((1, 11), dtype=np.float32))
        values.append(env._potential_before.numpy()[0, 0])
    np.testing.assert_allclose(values, [0, -50 / 1001, -25, -1000 / 21, -1050 / 22], atol=1e-5)
    assert all(-50 < v <= 0 for v in values)
    assert np.all(np.diff(values) < 0)


def test_old_observation_channels_unchanged_below_clamp_and_new_scales():
    env = RocketEnv(4, obs_config=ObsConfig(mirror=False), device="cpu")
    bodies = env.sim.state.body.numpy()
    bodies["pos"] = [(200, 300, 0), (200, 300, 1000), (200, 300, 5000), (200, 300, 20000)]
    bodies["vel"] = (120, 0, -250)
    env.sim.state.body.assign(bodies)
    obs = env.observe().numpy()
    for i, b in enumerate(bodies):
        raw_pad = np.array(wp.quat_rotate_inv(wp.quat(*b["rot"]), -wp.vec3(*b["pos"])))
        if i < 3:
            np.testing.assert_array_equal(
                obs[i, 6:9], (raw_pad * np.float32(1 / 3000)).astype(np.float32)
            )
        else:
            assert np.linalg.norm(obs[i, 6:9]) == pytest.approx(5100 / 3000)
    np.testing.assert_allclose(obs[:, 12], [0, 1 / 3, 5 / 3, 5 / 3])
    air = sample_air(bodies["pos"][:, 2], device="cpu")
    speed = np.linalg.norm(bodies["vel"], axis=1)
    np.testing.assert_allclose(obs[:, 24], np.log1p(bodies["pos"][:, 2] / 1000) / 3)
    np.testing.assert_allclose(obs[:, 25], speed / air[:, 3])
    np.testing.assert_allclose(obs[:, 26], 0.5 * air[:, 2] * speed**2 / 5000, rtol=2e-7)
    np.testing.assert_allclose(obs[:, 27], math.log1p(math.hypot(200, 300) / 1000) / 3)


def test_entry_stages_preserve_legacy_and_raise_handover():
    cfg = test_train.compose_cfg("curriculum=landing")
    curriculum = Curriculum(cfg, cfg.curriculum)
    for stage in range(9):
        sim, reward = curriculum.configs(stage)
        assert not sim.spawn.entry
        assert reward.distance_saturation == 0
    assert curriculum.configs(9)[0] == curriculum.configs(8)[0]
    for stage, altitude, speed, angle in [
        (10, 5000, 110, 65),
        (11, 8000, 200, 60),
        (12, 12000, 200, 60),
        (13, 20000, 280, 64),
    ]:
        sim, reward = curriculum.configs(stage)
        assert sim.spawn.entry
        assert sim.spawn.altitude == altitude
        assert sim.spawn.speed == speed
        assert sim.spawn.flight_path_angle == pytest.approx(math.radians(angle))
        assert sim.max_time == 240
        assert reward.distance_saturation == 1000


def test_checkpoint_migration_preserves_actions_values_and_optimizer(tmp_path, monkeypatch):
    cfg = test_train.compose_cfg("num_envs=8")
    make_env = test_train.make_env

    def legacy_env(*args):
        env = make_env(*args)
        env._obs = env._obs[:, :LEGACY_OBS_DIM]
        return env

    with monkeypatch.context() as patch:
        patch.setattr(test_train, "make_env", legacy_env)
        old = test_train.make_runner(cfg, tmp_path / "old", eval_envs=4)
    test_train.collect(old)
    old.alg.update()
    for model in (old.alg.actor, old.alg.critic):
        model.obs_normalizer.count.fill_(3_000_000_000)
        model.eval()
        model.reset()
    checkpoint = tmp_path / "legacy.pt"
    old.save(str(checkpoint))
    exported = tmp_path / "policy.pt"
    export_policy(old.alg.actor, exported)
    policy = load_policy(exported)
    resumed = test_train.make_runner(cfg, tmp_path / "new", eval_envs=4)
    resumed.load(str(checkpoint))
    for previous, model in (
        (old.alg.actor, resumed.alg.actor),
        (old.alg.critic, resumed.alg.critic),
    ):
        model.eval()
        for name in ("_mean", "_var", "_std"):
            torch.testing.assert_close(
                getattr(model.obs_normalizer, name)[:, :24],
                getattr(previous.obs_normalizer, name),
                rtol=0,
                atol=0,
            )
            assert torch.all(
                getattr(model.obs_normalizer, name)[:, 24:] == (0 if name == "_mean" else 1)
            )
        assert model.obs_normalizer.count == 3_000_000_000
        assert not model.rnn.rnn.weight_ih_l0[:, 24:].any()
    for _ in range(5):
        obs = resumed.env.get_observations()
        legacy = TensorDict({"policy": obs["policy"][:, :24]}, batch_size=[8])
        with torch.no_grad():
            expected = old.alg.actor(legacy)
            torch.testing.assert_close(resumed.alg.actor(obs), expected, rtol=0, atol=1e-7)
            torch.testing.assert_close(policy(obs), expected, rtol=0, atol=1e-7)
            torch.testing.assert_close(
                resumed.alg.critic(obs), old.alg.critic(legacy), rtol=0, atol=1e-7
            )
        resumed.env.step(expected)
    for state in resumed.alg.optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].ndim == 2 and state[key].shape[-1] == OBS_DIM:
                assert not state[key][:, 24:].any()
    resumed._reset_rollout()
    resumed.alg.train_mode()
    test_train.collect(resumed)
    assert resumed.alg.update()["optim/updates"] > 0
