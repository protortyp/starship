import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import warp as wp

from starship.sim import CRASHED, LANDED, OUT_OF_BOUNDS, TIMEOUT, TUMBLED, SimConfig, Simulator
from starship.sim.mass import make_mass_params, sample
from starship.train import OBS_DIM, ObsConfig, RewardConfig, RocketEnv
from starship.train.actions import ACTION_DIM, CONTINUOUS, CONTINUOUS_HEADS, ENGINE_HEADS
from starship.train.encoding import make_obs_scales, observe_kernel
from starship.train.reward import REWARD_TERMS
from starship.train.rsl_env import RslRocketEnv


@pytest.fixture(params=[str(device) for device in wp.get_devices()])
def device(request):
    return request.param


def config(**spawn):
    cfg = SimConfig()
    cfg.spawn = replace(cfg.spawn, altitude=150.0, vertical_speed=-5.0, pitch=0.0)
    cfg.spawn = replace(cfg.spawn, **spawn)
    return cfg


def actions(n, engines=0.0):
    result = np.zeros((n, ACTION_DIM), dtype=np.float32)
    result[:, :ENGINE_HEADS] = engines
    return result


def com(cfg):
    v, s = cfg.vehicle, cfg.spawn
    tanks = [
        [
            v.tanks[name].capacity * fill
            for name, fill in (
                ("main_lox", s.main_fill),
                ("main_ch4", s.main_fill),
                ("header_lox", s.header_fill),
                ("header_ch4", s.header_fill),
            )
        ]
    ]
    return float(sample(make_mass_params(v), tanks, device="cpu")[0, 1])


def test_observation_layout_and_scale(device):
    env = RocketEnv(16, device=device, obs_config=ObsConfig(mirror=False))
    obs = env.obs.numpy()
    assert OBS_DIM == 24
    assert obs.shape == (16, 24)
    assert np.isfinite(obs).all()
    assert np.abs(obs).max() <= 2
    np.testing.assert_allclose(np.linalg.norm(obs[:, :3], axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(obs[:, 3:6], axis=1), 0.85, atol=1e-6)
    np.testing.assert_allclose(obs[:, 6:9], -obs[:, :3], atol=1e-6)
    np.testing.assert_array_equal(obs[:, 9:12], 0)
    np.testing.assert_array_equal(obs[:, 12], 1)
    np.testing.assert_allclose(obs[:, 13:17], 45 / 85, atol=1e-6)
    np.testing.assert_array_equal(obs[:, 17:20], 0)
    np.testing.assert_allclose(obs[:, 20], 0.4)
    np.testing.assert_array_equal(obs[:, 21:23], 0)
    np.testing.assert_array_equal(obs[:, 23], 1)
    for _ in range(5):
        result = env.step(actions(16, 1))
        assert np.isfinite(result.obs.numpy()).all()
        np.testing.assert_allclose(result.reward.numpy(), result.reward_terms.numpy().sum(1))


def test_heading_invariance(device):
    env = RocketEnv(
        2, config(offset=30, horizontal_speed=8), device=device, obs_config=ObsConfig(mirror=False)
    )
    b = env.sim.state.body.numpy()
    q = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 1.3)
    b[1] = b[0]
    for name in ("pos", "vel"):
        b[name][1] = wp.quat_rotate(q, wp.vec3(*b[name][0]))
    b["rot"][1] = wp.mul(q, wp.quat(*b["rot"][0]))
    env.sim.state.body.assign(b)
    obs = env.observe().numpy()
    np.testing.assert_allclose(obs[0], obs[1], atol=1e-6)


def mirror_pair(env):
    s = env.sim.state
    b = s.body.numpy()
    b[1] = b[0]
    for name in ("pos", "vel"):
        b[name][1, 0] *= -1
    b["omega"][1] *= (-1, 1, -1)
    rotation = np.array(wp.quat_to_matrix(wp.quat(*b["rot"][0]))).reshape(3, 3)
    reflected = np.diag([-1, 1, 1]) @ rotation @ np.diag([1, -1, 1])
    b["rot"][1] = wp.quat_from_matrix(wp.mat33(*reflected.flatten()))
    s.body.assign(b)
    e = s.engine.numpy()
    for name in e.dtype.names:
        e[name][1] = e[name][0, [0, 2, 1]]
        if name.startswith("gimbal2"):
            e[name][1] *= -1
    s.engine.assign(e)
    f = s.flaps.numpy()
    for name in f.dtype.names:
        f[name][1] = f[name][0, [1, 0, 3, 2]]
    s.flaps.assign(f)
    env.mirrored.assign(np.array([0, 1], dtype=np.int32))


def test_mirrored_trajectories_and_observations(device):
    cfg = config(offset=20, horizontal_speed=8, pitch=0.25, angular_rate_noise=0.02)
    env = RocketEnv(2, cfg, device=device, seed=4)
    e = env.sim.state.engine.numpy()
    e["spool"][0] = (0.2, 0.4, 0.7)
    env.sim.state.engine.assign(e)
    mirror_pair(env)
    np.testing.assert_allclose(*env.observe().numpy(), atol=2e-6)
    raw = actions(2)
    raw[:, :3] = (1, 0, 1)
    raw[:, 3:] = (0.4, 0.3, -0.1, -0.2, 0.4, 0.1, 0.3, 0.5)
    for _ in range(20):
        result = env.step(raw)
        assert not env.sim.terminal.done.numpy().any()
        np.testing.assert_allclose(*result.obs.numpy(), atol=3e-5)
        np.testing.assert_allclose(*result.reward.numpy(), atol=3e-5)
        snap = env.sim.snapshot()
        np.testing.assert_allclose(snap.pos[1], snap.pos[0] * (-1, 1, 1), atol=3e-4)
        np.testing.assert_allclose(snap.vel[1], snap.vel[0] * (-1, 1, 1), atol=3e-4)
        np.testing.assert_allclose(snap.omega[1], snap.omega[0] * (-1, 1, -1), atol=3e-5)
    np.testing.assert_array_equal(env.sim.commands.engine_on.numpy(), [[1, 0, 1], [1, 1, 0]])
    np.testing.assert_allclose(*env.sim.commands.gimbal2.numpy() * [[1], [-1]])
    np.testing.assert_allclose(env.sim.commands.rcs.numpy().sum(), 0, atol=1e-7)


def test_action_ranges_and_engine_heads(device):
    env = RocketEnv(3, config(), device=device, obs_config=ObsConfig(mirror=False))
    raw = actions(3)
    raw[:, 3:] = np.array([-100, 0, 100])[:, None]
    raw[:, :3] = (0.5, 0.5001, 1)
    env.step(raw)
    squashed = env.squashed_actions.numpy()
    assert squashed.shape == (3, CONTINUOUS_HEADS)
    limits = np.array([(lo, hi) for _, lo, hi in CONTINUOUS])
    np.testing.assert_allclose(squashed[0], limits[:, 0], atol=1e-7)
    np.testing.assert_allclose(squashed[1], limits.mean(1), atol=1e-7)
    np.testing.assert_allclose(squashed[2], limits[:, 1], atol=1e-7)
    for name, index in (("throttle", 0), ("gimbal2", 1), ("gimbal3", 2)):
        np.testing.assert_allclose(
            getattr(env.sim.commands, name).numpy(), np.repeat(squashed[:, index, None], 3, axis=1)
        )
    np.testing.assert_array_equal(env.sim.commands.flaps.numpy(), squashed[:, 3:7])
    np.testing.assert_array_equal(env.sim.snapshot().spool[:, 0], 0)
    assert (env.sim.snapshot().spool[:, 1:] > 0).all()


@pytest.mark.parametrize("gamma", [1.0, 0.95])
def test_shaping_telescopes(device, gamma):
    cfg = config(altitude=50, vertical_speed=-40, pitch=0.15)
    env = RocketEnv(1, cfg, reward_config=RewardConfig(gamma=gamma), device=device)
    total = np.zeros(3)
    initial = None
    for step in range(50):
        result = env.step(actions(1))
        if initial is None:
            initial = env._potential_before.numpy()[0].copy()
            assert (initial < 0).all()
        total += gamma**step * result.reward_terms.numpy()[0, :3]
        if result.terminated.numpy()[0] or result.truncated.numpy()[0]:
            break
    else:
        pytest.fail("episode did not end")
    np.testing.assert_allclose(total, -initial, atol=2e-5)


@pytest.mark.parametrize("status", [LANDED, CRASHED, TUMBLED, OUT_OF_BOUNDS, TIMEOUT])
def test_terminal_rewards_and_final_observations(device, status):
    cfg = config()
    if status in (LANDED, CRASHED):
        cfg.spawn.altitude = com(cfg) - 0.001
        cfg.spawn.vertical_speed = -0.2 if status == LANDED else -12.0
    elif status == TUMBLED:
        cfg.spawn.pitch = math.radians(130)
    elif status == OUT_OF_BOUNDS:
        cfg.max_offset, cfg.spawn.offset = 10, 20
    else:
        cfg.max_time = cfg.dt
    env = RocketEnv(1, cfg, obs_config=ObsConfig(mirror=False), device=device, seed=3)
    twin = Simulator(1, cfg, device=device, seed=3)
    initial_load = env.landing_load.numpy()[0]
    result = env.step(actions(1, 1))
    assert env.final_status.numpy()[0] == status
    assert result.truncated.numpy()[0] == (status == TIMEOUT)
    assert result.terminated.numpy()[0] == (status != TIMEOUT)
    terminal_reward = result.terminal_reward.numpy()[0]
    weights = env.reward_config
    if status == LANDED:
        fraction = env.final_tanks.numpy().sum() / initial_load
        assert terminal_reward == pytest.approx(
            weights.landed_bonus + weights.fuel_bonus * fraction
        )
    elif status == CRASHED:
        assert terminal_reward < -weights.crash_penalty
    else:
        assert (
            terminal_reward
            == -{
                TUMBLED: weights.tumble_penalty,
                OUT_OF_BOUNDS: weights.out_of_bounds_penalty,
                TIMEOUT: weights.timeout_penalty,
            }[status]
        )
    twin.step(
        **{
            name: getattr(env.sim.commands, name).numpy()
            for name in ("engine_on", "throttle", "gimbal2", "gimbal3", "flaps", "rcs")
        }
    )
    expected = wp.zeros((1, OBS_DIM), dtype=float, device=device)
    wp.launch(
        observe_kernel,
        dim=1,
        inputs=[twin.state, env.landing_load, env.final_mirrored, env.obs_scales, expected],
        device=device,
    )
    np.testing.assert_allclose(result.final_obs.numpy(), expected.numpy(), atol=1e-6)
    assert result.final_obs.numpy()[0, 17:20].min() > 0
    np.testing.assert_array_equal(result.obs.numpy()[0, 17:20], 0)
    np.testing.assert_array_equal(env.sim.state.steps.numpy(), 0)
    assert not np.array_equal(result.final_obs.numpy(), result.obs.numpy())
    burned = initial_load - env.final_tanks.numpy().sum()
    terms = result.reward_terms.numpy()[0]
    assert terms[REWARD_TERMS.index("fuel")] == pytest.approx(-weights.fuel_cost * burned)
    assert terms[REWARD_TERMS.index("time")] == pytest.approx(
        -weights.time_cost * env.final_time.numpy()[0]
    )


@pytest.mark.parametrize(
    ("component", "weight"),
    [
        ("vertical", "crash_vertical_weight"),
        ("horizontal", "crash_horizontal_weight"),
        ("tilt", "crash_tilt_weight"),
        ("rate", "crash_rate_weight"),
        ("miss", "crash_miss_weight"),
    ],
)
def test_each_touchdown_excess_is_penalized(device, component, weight):
    cfg = config(altitude=0, vertical_speed=0)
    if component == "vertical":
        cfg.spawn.vertical_speed = -10
    elif component == "horizontal":
        cfg.spawn.horizontal_speed = 5
    elif component == "tilt":
        cfg.spawn.pitch = 0.3
    elif component == "miss":
        cfg.spawn.offset = 30
    env = RocketEnv(1, cfg, device=device)
    if component == "rate":
        b = env.sim.state.body.numpy()
        b["omega"][0] = (0.3, 0, 0)
        env.sim.state.body.assign(b)
    env.step(actions(1))
    baseline = env.terminal_reward.numpy()[0]
    assert baseline < -env.reward_config.crash_penalty
    # Replay from the identical state with this excess weight disabled.
    other = RocketEnv(
        1, cfg, reward_config=replace(env.reward_config, **{weight: 0}), device=device
    )
    if component == "rate":
        other.sim.state.body.assign(b)
    other.step(actions(1))
    assert other.terminal_reward.numpy()[0] > baseline


def test_jerk_history_resets_on_respawn(device):
    cfg = config()
    cfg.max_time = 0.2
    env = RocketEnv(1, cfg, device=device)
    raw = actions(1)
    env.step(raw)
    jerk_index = REWARD_TERMS.index("jerk")
    assert env.reward_terms.numpy()[0, jerk_index] == 0
    previous = env.squashed_actions.numpy().copy()
    raw[:, 3:] = 0.8
    env.step(raw)
    expected = -env.reward_config.jerk_cost * ((env.squashed_actions.numpy() - previous) ** 2).sum()
    assert env.reward_terms.numpy()[0, jerk_index] == pytest.approx(expected)
    assert env.truncated.numpy()[0]
    np.testing.assert_array_equal(env.previous_actions.numpy(), 0)
    env.step(actions(1))
    assert env.reward_terms.numpy()[0, jerk_index] == 0


def test_mirror_respawn_rng_and_explicit_reset(device):
    env = RocketEnv(128, config(), device=device, seed=9)
    other = RocketEnv(128, config(), device=device, seed=9)
    before = env.mirrored.numpy().copy()
    np.testing.assert_array_equal(before, other.mirrored.numpy())
    assert set(before) == {0, 1}
    body = env.sim.state.body.numpy()
    body["pos"][:64, 2] = 0
    env.sim.state.body.assign(body)
    env.step(actions(128))
    after = env.mirrored.numpy().copy()
    np.testing.assert_array_equal(env.final_mirrored.numpy(), before)
    np.testing.assert_array_equal(after[64:], before[64:])
    assert np.any(after[:64] != before[:64])
    env.reset()
    assert np.any(env.mirrored.numpy() != after)
    assert not env.sim.terminal.done.numpy().any()
    assert not env.terminated.numpy().any()
    np.testing.assert_array_equal(env.previous_actions.numpy(), 0)
    np.testing.assert_array_equal(env.reward.numpy(), 0)


def test_configs_preserve_episode_fuel_baseline(device):
    cfg = config()
    cfg.max_time = 0.05
    env = RocketEnv(1, cfg, device=device)
    before = env.landing_load.numpy().copy()
    new_cfg = replace(cfg, spawn=replace(cfg.spawn, header_fill=0.5))
    env.set_configs(new_cfg, replace(env.reward_config, timeout_penalty=7))
    np.testing.assert_array_equal(env.landing_load.numpy(), before)
    env.step(actions(1))
    assert env.terminal_reward.numpy()[0] == -7
    assert env.final_obs.numpy()[0, 23] == 1
    assert env.obs.numpy()[0, 23] == 1
    assert env.landing_load.numpy()[0] < before[0]


def test_rsl_adapter_shapes_timeouts_and_owned_rollouts(device):
    cfg = config()
    cfg.max_time = 0.05
    env = RslRocketEnv(RocketEnv(4, cfg, device=device))
    assert env.num_envs == 4
    assert env.num_actions == ACTION_DIM
    assert env.max_episode_length == 1
    obs = env.get_observations()
    assert obs.batch_size == (4,)
    assert obs["policy"].shape == (4, OBS_DIM)
    old_obs = obs["policy"].clone()
    raw = torch.ones((4, ACTION_DIM), device=device)
    next_obs, rewards, dones, extras = env.step(raw)
    assert rewards.shape == dones.shape == extras["time_outs"].shape == (4,)
    assert dones.all()
    assert not extras["time_outs"].any()
    assert extras["truncated"].all()
    assert (env.terminal_reward == -env.env.reward_config.timeout_penalty).all()
    np.testing.assert_array_equal(env.final_status.cpu().numpy(), TIMEOUT)
    assert env.actions.shape == (4, ACTION_DIM)
    assert env.squashed_actions.shape == (4, CONTINUOUS_HEADS)
    assert env.engine_on.shape == (4, 3)
    assert env.final_pos.shape == env.final_vel.shape == env.final_omega.shape == (4, 3)
    assert env.final_rot.shape == env.final_tanks.shape == (4, 4)
    assert env.final_steps.shape == env.mirrored.shape == env.final_mirrored.shape == (4,)
    torch.testing.assert_close(obs["policy"], old_obs)
    assert next_obs["policy"].data_ptr() != wp.to_torch(env.env.obs).data_ptr()
    assert env.reward.data_ptr() == wp.to_torch(env.env.reward).data_ptr()
    stored_reward = env.reward.clone()
    rewards.add_(100)
    torch.testing.assert_close(env.reward, stored_reward)
    env.set_configs(replace(cfg, max_time=0.3), env.env.reward_config)
    assert env.max_episode_length == 3
    _, _, dones, extras = env.step(torch.zeros_like(raw))
    assert not dones.any()
    assert not extras["time_outs"].any()
    assert not env.final_obs.any()


def test_validation(device):
    with pytest.raises(ValueError, match="positive"):
        make_obs_scales(ObsConfig(velocity=0))
    with pytest.raises(ValueError, match="gamma"):
        RocketEnv(1, device=device, reward_config=RewardConfig(gamma=1.1))
    env = RocketEnv(2, device=device)
    with pytest.raises(ValueError, match="shape"):
        env.step(actions(1))


def test_cuda_stream_interop(device):
    if not wp.get_device(device).is_cuda:
        pytest.skip("requires CUDA")
    env = RslRocketEnv(RocketEnv(32, config(), device=device))
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        obs = env.get_observations()["policy"]
        torch.testing.assert_close(obs[:, 23], torch.ones(32, device=device))
        raw = torch.zeros((32, ACTION_DIM), device=device)
        raw[:, :3] = 1
        next_obs, _, _, _ = env.step(raw)
        assert (next_obs["policy"][:, 17:20] > 0).all()
    # reset uses Warp's stream and must wait for the preceding step on the Torch stream.
    env.env.reset()
    with torch.cuda.stream(stream):
        fresh = env.get_observations()["policy"]
        assert not fresh[:, 17:20].any()
        env.env.step(torch.ones_like(raw))
        assert (env.get_observations()["policy"][:, 17:20] > 0).all()
    stream.synchronize()
