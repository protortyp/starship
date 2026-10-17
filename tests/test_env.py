import math
from dataclasses import replace

import numpy as np
import pytest

from starship.sim import CRASHED, LANDED, MAIN, TIMEOUT, SimConfig, SpawnConfig, WorldConfig
from starship.train import OBS_DIM, ObsConfig, RewardConfig, RocketEnv

ZERO_G_VACUUM = WorldConfig(gravity=0.0, sea_level_density=0.0)
NOISY_SPAWN = SpawnConfig().with_noise(1.0)


def make_env(num_envs=1, gamma=1.0, fuel_cost=0.0, time_cost=0.0, **overrides) -> RocketEnv:
    # gamma = 1 and no fuel cost make the shaping telescope exactly, which the episode-level
    # checks rely on.
    reward = RewardConfig(gamma=gamma, fuel_cost=fuel_cost, time_cost=time_cost)
    return RocketEnv(
        num_envs, replace(SimConfig(), **overrides), reward_config=reward, device="cpu"
    )


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
        + w.speed_weight
        * max(
            float(np.linalg.norm(vel))
            - math.sqrt(2 * w.profile_decel * h)
            - w.profile_touchdown_speed,
            0.0,
        )
        + w.lateral_weight * abs(pos[0]) * math.exp(-h / w.lateral_fade_altitude)
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
    shaping = -potential(env, start.pos[0], start.vel[0], float(start.theta[0]))
    assert total == pytest.approx(shaping + terminal, rel=1e-3)


def test_landing_reward_includes_fuel_bonus():
    env = make_env(world=ZERO_G_VACUUM, spawn=SpawnConfig(y=25.5, vy=-1.0, theta=0.0))
    total, start, term = run_episode(env, lambda _: np.zeros(1, dtype=np.int32))
    w = env.reward_config
    assert term.status[0] == LANDED
    assert term.fuel[0] == env.config.rocket.fuel_mass
    shaping = -potential(env, start.pos[0], start.vel[0], 0.0)
    assert total == pytest.approx(shaping + w.landed_reward + w.fuel_bonus, rel=1e-4)


def test_crash_penalty_grows_with_impact_speed():
    def crash_total(vy):
        env = make_env(world=ZERO_G_VACUUM, spawn=SpawnConfig(y=30.0, vy=vy, theta=0.0))
        return run_episode(env, lambda _: np.zeros(1, dtype=np.int32))[0]

    assert crash_total(-50.0) < crash_total(-10.0) < 0.0


def test_timeout_is_penalized():
    env = make_env(world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0), max_time=1.0)
    total, start, term = run_episode(env, lambda _: np.zeros(1, dtype=np.int32))
    assert term.status[0] == TIMEOUT
    shaping = -potential(env, start.pos[0], start.vel[0], float(start.theta[0]))
    assert total == pytest.approx(shaping - env.reward_config.timeout_penalty, rel=1e-4)


def test_shaping_discounts_the_next_potential():
    env = make_env(gamma=0.9, spawn=SpawnConfig(y=1000.0, vy=-50.0))
    env.reset()
    s0 = env.sim.snapshot()
    reward = env.step(np.zeros(1, dtype=np.int32)).reward.numpy()[0]
    s1 = env.sim.snapshot()
    phi0 = potential(env, s0.pos[0], s0.vel[0], float(s0.theta[0]))
    phi1 = potential(env, s1.pos[0], s1.vel[0], float(s1.theta[0]))
    assert reward == pytest.approx(0.9 * phi1 - phi0, rel=1e-4)


def test_discounted_shaping_return_is_independent_of_the_trajectory():
    """With a zero terminal potential, sum_t gamma^t F_t = -potential(start) for any actions."""
    gamma = 0.99
    totals = []
    for action in (0, 1, 4):
        env = make_env(gamma=gamma, spawn=SpawnConfig(y=300.0, vy=-40.0))
        env.reset()
        start = env.sim.snapshot()
        discounted, scale = 0.0, 1.0
        for _ in range(env.config.max_steps):
            result = env.step(np.full(1, action, dtype=np.int32))
            discounted += scale * float(
                result.reward.numpy()[0] - result.terminal_reward.numpy()[0]
            )
            scale *= gamma
            if result.terminated.numpy()[0] or result.truncated.numpy()[0]:
                break
        totals.append(discounted)
    expected = -potential(env, start.pos[0], start.vel[0], float(start.theta[0]))
    assert totals == pytest.approx([expected] * 3, rel=1e-3)


def test_fuel_cost_is_charged_per_kg_burned():
    def first_reward(fuel_cost, action):
        env = make_env(fuel_cost=fuel_cost, world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0))
        env.reset()
        return env.step(np.full(1, action, dtype=np.int32)).reward.numpy()[0]

    cfg = SimConfig()
    burned = cfg.rocket.main_mass_flow * cfg.policy_dt
    assert first_reward(0.01, MAIN) - first_reward(0.0, MAIN) == pytest.approx(
        -0.01 * burned, rel=1e-3
    )
    assert first_reward(0.01, 0) == pytest.approx(first_reward(0.0, 0))


def test_time_cost_is_charged_per_second_of_flight():
    def episode_total(time_cost):
        env = make_env(time_cost=time_cost, spawn=SpawnConfig(y=100.0, vy=-50.0))
        total, _, term = run_episode(env, lambda _: np.zeros(1, dtype=np.int32))
        return total, int(term.steps[0])

    with_cost, steps = episode_total(2.0)
    without_cost, _ = episode_total(0.0)
    assert with_cost - without_cost == pytest.approx(-2.0 * steps * SimConfig().policy_dt, rel=1e-4)


def test_configs_can_change_in_place():
    env = make_env(num_envs=4)
    env.reset()
    buffers = (env.obs.ptr, env.sim.state.pos.ptr)
    env.set_configs(
        sim_config=replace(SimConfig(), spawn=SpawnConfig(y=1234.0)),
        reward_config=RewardConfig(gamma=1.0, fuel_cost=0.0, time_cost=10.0),
    )
    env.reset()
    assert (env.obs.ptr, env.sim.state.pos.ptr) == buffers
    np.testing.assert_allclose(env.sim.snapshot().pos[:, 1], 1234.0)
    reward = env.step(np.zeros(4, dtype=np.int32)).reward.numpy()
    reference = make_env(num_envs=4, spawn=SpawnConfig(y=1234.0))
    reference.reset()
    baseline = reference.step(np.zeros(4, dtype=np.int32)).reward.numpy()
    np.testing.assert_allclose(reward - baseline, -10.0 * SimConfig().policy_dt, rtol=1e-4)


def test_speed_shaping_only_penalizes_excess_over_the_descent_profile():
    def braking_gain(altitude):
        """Reward of braking minus reward of coasting for one step."""
        rewards = []
        for action in (MAIN, 0):
            env = make_env(world=ZERO_G_VACUUM, spawn=SpawnConfig(y=altitude, vy=-80.0, theta=0.0))
            env.reset()
            rewards.append(env.step(np.full(1, action, dtype=np.int32)).reward.numpy()[0])
        return rewards[0] - rewards[1]

    # 80 m/s is within the safe profile at 3 km (~150 m/s), so braking there earns nothing;
    # at 100 m the profile allows ~29 m/s, so braking pays.
    assert abs(braking_gain(3000.0)) < 0.01
    assert braking_gain(100.0) > 0.1

    def descending_cost(vy):
        """Shaping of one step descending at vy from 200 m, without the distance term."""
        env = RocketEnv(
            1,
            replace(SimConfig(), world=ZERO_G_VACUUM, spawn=SpawnConfig(y=225.0, vy=vy, theta=0.0)),
            reward_config=RewardConfig(gamma=1.0, fuel_cost=0.0, distance_weight=0.0),
            device="cpu",
        )
        env.reset()
        return env.step(np.zeros(1, dtype=np.int32)).reward.numpy()[0]

    # Descending within the profile costs nothing, unlike a speed term faded in with altitude.
    assert descending_cost(-20.0) == pytest.approx(0.0, abs=1e-6)
    assert descending_cost(-60.0) < -0.01


def test_lateral_shaping_rewards_moving_toward_the_pad_near_the_ground():
    def step_reward(lateral_weight, vx):
        env = RocketEnv(
            1,
            replace(
                SimConfig(),
                world=ZERO_G_VACUUM,
                spawn=SpawnConfig(x=100.0, y=60.0, vx=vx, vy=0.0, theta=0.0),
            ),
            reward_config=RewardConfig(gamma=1.0, fuel_cost=0.0, lateral_weight=lateral_weight),
            device="cpu",
        )
        env.reset()
        return env.step(np.zeros(1, dtype=np.int32)).reward.numpy()[0]

    # Moving 1 m toward the pad (vx = -10 m/s for 0.1 s) at 60 m altitude.
    gain = step_reward(0.1, -10.0) - step_reward(0.0, -10.0)
    assert gain == pytest.approx(0.1 * 1.0 * math.exp(-35.0 / 500.0), rel=1e-3)


def test_soft_touchdown_credit_smooths_the_landing_cliff():
    def terminal(soft_speed, vy, x=0.0):
        spawn = SpawnConfig(x=x, y=25.5, vy=vy, theta=0.0)
        env = RocketEnv(
            1,
            replace(SimConfig(), world=ZERO_G_VACUUM, spawn=spawn),
            reward_config=RewardConfig(gamma=1.0, fuel_cost=0.0, soft_touchdown_speed=soft_speed),
            device="cpu",
        )
        _, _, term = run_episode(env, lambda _: np.zeros(1, dtype=np.int32))
        return env.terminal_reward.numpy()[0], int(term.status[0])

    hard, status = terminal(0.0, -6.0)
    soft, _ = terminal(10.0, -6.0)
    assert status == CRASHED
    assert soft - hard == pytest.approx(100.0 * (1.0 - 6.0 / 10.0), rel=1e-3)
    off_pad, _ = terminal(10.0, -6.0, x=60.0)
    assert off_pad == pytest.approx(hard, rel=1e-3)
