import math
from dataclasses import replace

import numpy as np
import pytest

from starship.sim import CRASHED, LANDED, TIMEOUT, SimConfig, SpawnConfig, WorldConfig
from starship.train import OBS_DIM, ObsConfig, RocketEnv

ZERO_G_VACUUM = WorldConfig(gravity=0.0, sea_level_density=0.0)
NOISY_SPAWN = SpawnConfig().with_noise(1.0)


def make_env(num_envs=1, **overrides) -> RocketEnv:
    return RocketEnv(num_envs, replace(SimConfig(), **overrides), device="cpu")


def expected_obs(env: RocketEnv, pos, vel, theta, omega, fuel) -> np.ndarray:
    r, o = env.config.rocket, env.obs_config
    bottom = (
        pos[1] - 0.5 * r.length * abs(math.cos(theta)) - 0.5 * r.diameter * abs(math.sin(theta))
    )
    return np.array(
        [
            pos[0] / o.position_scale,
            bottom / o.position_scale,
            vel[0] / o.velocity_scale,
            vel[1] / o.velocity_scale,
            math.sin(theta),
            math.cos(theta),
            omega / o.angular_rate_scale,
            fuel / r.fuel_mass,
        ]
    )


def test_base_scenario_initial_observation():
    env = make_env()
    obs = env.reset().numpy()
    assert obs.shape == (1, OBS_DIM)
    assert obs.dtype == np.float32
    # Lying flat at 3 km: lowest point is half a diameter below the center, falling at 90 m/s.
    np.testing.assert_allclose(obs[0], [0.0, 2.9955, 0.0, -1.8, 1.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_observations_match_state_under_random_actions():
    env = make_env(num_envs=32, spawn=NOISY_SPAWN)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(50):
        result = env.step(rng.integers(0, 8, size=32, dtype=np.int32))
    obs = result.obs.numpy()
    s = env.sim.snapshot()
    for i in range(32):
        want = expected_obs(
            env, s.pos[i], s.vel[i], float(s.theta[i]), float(s.omega[i]), float(s.fuel[i])
        )
        np.testing.assert_allclose(obs[i], want, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(obs[:, 4] ** 2 + obs[:, 5] ** 2, 1.0, rtol=1e-6)


def test_custom_obs_scales():
    cfg = ObsConfig(position_scale=100.0, velocity_scale=10.0)
    env = RocketEnv(1, obs_config=cfg, device="cpu")
    obs = env.reset().numpy()
    assert obs[0, 1] == pytest.approx(29.955)
    assert obs[0, 3] == pytest.approx(-9.0)


def run_until_done(env: RocketEnv, max_steps: int = 10_000):
    env.reset()
    for _ in range(max_steps):
        result = env.step(np.zeros(env.num_envs, dtype=np.int32))
        if result.terminated.numpy()[0] or result.truncated.numpy()[0]:
            return result
    raise AssertionError("episode did not end")


def test_crash_is_terminated_with_final_obs_at_ground():
    env = make_env(spawn=SpawnConfig(y=100.0, vy=-50.0))
    result = run_until_done(env)
    assert result.terminated.numpy()[0] == 1
    assert result.truncated.numpy()[0] == 0
    assert env.sim.terminal_snapshot().status[0] == CRASHED
    final_obs = result.final_obs.numpy()[0]
    assert final_obs[1] == pytest.approx(0.0, abs=2e-3)
    # The live observation already belongs to the respawned episode.
    assert result.obs.numpy()[0, 1] == pytest.approx(0.0955, abs=1e-3)


def test_timeout_is_truncated():
    env = make_env(world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0), max_time=1.0)
    result = run_until_done(env)
    assert result.truncated.numpy()[0] == 1
    assert result.terminated.numpy()[0] == 0
    assert env.sim.terminal_snapshot().status[0] == TIMEOUT


def test_flags_clear_on_the_next_step():
    env = make_env(spawn=SpawnConfig(y=100.0, vy=-50.0))
    run_until_done(env)
    result = env.step(np.zeros(1, dtype=np.int32))
    assert result.terminated.numpy()[0] == 0
    assert result.truncated.numpy()[0] == 0


def potential(env: RocketEnv, pos, vel, theta) -> float:
    r, w = env.config.rocket, env.reward_config
    h = max(
        pos[1] - 0.5 * r.length * abs(math.cos(theta)) - 0.5 * r.diameter * abs(math.sin(theta)),
        0.0,
    )
    return -(
        w.distance_weight * math.hypot(pos[0], h) / 1000.0
        + w.speed_weight * float(np.linalg.norm(vel))
        + w.tilt_weight * abs(theta) * math.exp(-h / w.tilt_fade_altitude)
    )


def run_episode(env: RocketEnv, actions):
    """Returns (sum of rewards, initial state, terminal snapshot) for env 0."""
    env.reset()
    start = env.sim.snapshot()
    total = 0.0
    for k in range(env.config.max_steps):
        result = env.step(actions(k))
        total += float(result.reward.numpy()[0])
        if result.terminated.numpy()[0] or result.truncated.numpy()[0]:
            return total, start, env.sim.terminal_snapshot()
    raise AssertionError("episode did not end")


def test_shaping_telescopes_over_an_episode():
    env = make_env(spawn=NOISY_SPAWN)
    rng = np.random.default_rng(0)
    total, start, term = run_episode(env, lambda _: rng.integers(0, 8, size=1, dtype=np.int32))
    w = env.reward_config
    assert term.status[0] == CRASHED
    terminal = -(w.crash_penalty + w.crash_speed_weight * float(np.linalg.norm(term.vel[0])))
    shaping = potential(env, term.pos[0], term.vel[0], float(term.theta[0])) - potential(
        env, start.pos[0], start.vel[0], float(start.theta[0])
    )
    assert total == pytest.approx(shaping + terminal, rel=1e-3)


def test_landing_reward_includes_fuel_bonus():
    env = make_env(world=ZERO_G_VACUUM, spawn=SpawnConfig(y=25.5, vy=-1.0, theta=0.0))
    total, start, term = run_episode(env, lambda _: np.zeros(1, dtype=np.int32))
    w = env.reward_config
    assert term.status[0] == LANDED
    assert term.fuel[0] == env.config.rocket.fuel_mass
    shaping = potential(env, term.pos[0], term.vel[0], 0.0) - potential(
        env, start.pos[0], start.vel[0], 0.0
    )
    assert total == pytest.approx(shaping + w.landed_reward + w.fuel_bonus, rel=1e-4)


def test_crash_penalty_grows_with_impact_speed():
    def crash_total(vy):
        env = make_env(world=ZERO_G_VACUUM, spawn=SpawnConfig(y=30.0, vy=vy, theta=0.0))
        return run_episode(env, lambda _: np.zeros(1, dtype=np.int32))[0]

    # Shaping also rewards losing altitude, but the speed term and impact penalty dominate.
    assert crash_total(-50.0) < crash_total(-10.0) < 0.0


def test_timeout_is_penalized():
    env = make_env(world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0), max_time=1.0)
    total, start, term = run_episode(env, lambda _: np.zeros(1, dtype=np.int32))
    assert term.status[0] == TIMEOUT
    shaping = potential(env, term.pos[0], term.vel[0], float(term.theta[0])) - potential(
        env, start.pos[0], start.vel[0], float(start.theta[0])
    )
    assert total == pytest.approx(shaping - env.reward_config.timeout_penalty, rel=1e-4)
