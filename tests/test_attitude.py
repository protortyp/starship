import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import warp as wp

from starship.sim import CRASHED, LANDED, TUMBLED, Simulator
from starship.sim.physics import Attitude, attitude, track_attitude
from starship.sim.rigid import Body, rotate_by_body_rate
from starship.train import RewardConfig, RocketEnv
from starship.train.diagnose import attitude_lines, failure_lines
from starship.train.metrics import ATTITUDE_METRICS, EpisodeTracker
from starship.train.reward import REWARD_TERMS
from starship.train.rsl_env import RslRocketEnv
from tests.test_env import actions, config


@pytest.fixture(params=[str(device) for device in wp.get_devices()])
def device(request):
    return request.param


@wp.kernel
def spin(out: wp.array(dtype=Attitude), pitch: float, omega: wp.vec3):
    b = Body()
    b.rot = attitude(pitch, 0.7)
    b.omega = omega
    a = track_attitude(Attitude(), b, wp.vec3(0.0), 0.0)
    for _ in range(100):
        b.rot = rotate_by_body_rate(b.rot, omega, 0.01)
        a = track_attitude(a, b, wp.vec3(0.0), 0.01)
    out[0] = a


@pytest.mark.parametrize(
    ("pitch", "omega", "roll", "horizontal"),
    [
        (0.0, (0.5, 0.0, 0.0), 0.5, 0.0),
        (math.pi / 2, (0.0, 0.5, 0.0), 0.0, 0.5),
        (math.pi / 2, (0.0, 0.0, 0.5), 0.0, 0.0),
        # Belly-down roll is also rotation about a world-horizontal axis.
        (math.pi / 2, (-0.5, 0.0, 0.0), 0.5, 0.5),
    ],
)
def test_rotation_travel_and_peak_rates(device, pitch, omega, roll, horizontal):
    out = wp.zeros(1, dtype=Attitude, device=device)
    wp.launch(spin, 1, inputs=[out, pitch, wp.vec3(*omega)], device=device)
    a = out.numpy()[0]
    assert a["roll_travel"] == pytest.approx(roll, abs=2e-6)
    assert a["horizontal_travel"] == pytest.approx(horizontal, abs=2e-6)
    assert a["peak_roll_rate"] == pytest.approx(roll, abs=2e-6)
    assert a["peak_horizontal_rate"] == pytest.approx(horizontal, abs=2e-6)


@wp.kernel
def scripted_tilts(tilts: wp.array(dtype=float), out: wp.array(dtype=Attitude)):
    a = Attitude()
    for i in range(tilts.shape[0]):
        b = Body()
        b.rot = attitude(tilts[i], 0.0)
        a = track_attitude(a, b, wp.vec3(0.0), 0.01)
        out[i] = a


@pytest.mark.parametrize(
    ("tilts", "flips", "reflips"),
    [
        ([90, 70, 50, 20, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0]),
        ([0, 40, 70, 65, 55, 65, 20, 70], [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1, 1, 2]),
        ([45, 20, 40, 20], [0, 0, 0, 0], [0, 0, 0, 0]),
    ],
)
def test_flip_hysteresis(device, tilts, flips, reflips):
    out = wp.zeros(len(tilts), dtype=Attitude, device=device)
    wp.launch(
        scripted_tilts,
        1,
        inputs=[wp.array(np.radians(tilts), dtype=float, device=device), out],
        device=device,
    )
    np.testing.assert_array_equal(out.numpy()["flips"], flips)
    np.testing.assert_array_equal(out.numpy()["reflips"], reflips)


@wp.kernel
def sideslip_sample(flow: wp.vec3, out: wp.array(dtype=Attitude)):
    b = Body()
    b.rot = attitude(0.0, 0.0)
    # Ground velocity deliberately differs: the cost must use the supplied aerodynamic flow.
    b.vel = wp.vec3(0.0, 0.0, 100.0)
    out[0] = track_attitude(Attitude(), b, flow, 0.1)


@pytest.mark.parametrize(
    ("flow", "expected"),
    [((0, 20, 0), 0), ((0, 0, 100), 0), ((0, 100, 0), 0.1), ((30, -40, 0), 0.064)],
)
def test_sideslip_uses_air_relative_flow_and_speed_gate(device, flow, expected):
    out = wp.zeros(1, dtype=Attitude, device=device)
    wp.launch(sideslip_sample, 1, inputs=[wp.vec3(*flow), out], device=device)
    assert out.numpy()["sideslip"][0] == pytest.approx(expected)


def set_rate(sim, omega):
    bodies = sim.state.body.numpy()
    bodies["omega"][:] = omega
    sim.state.body.assign(bodies)


def test_substeps_match_single_steps_and_copy_preserves_counters(device):
    cfg = config(pitch=0.4, vertical_speed=-80, angular_rate_noise=0.1)
    batched = Simulator(3, cfg, device=device, seed=8)
    single = Simulator(3, replace(cfg, substeps=1), device=device, seed=8)
    batched.step()
    for _ in range(cfg.substeps):
        single.step()
    np.testing.assert_array_equal(batched.state.attitude.numpy(), single.state.attitude.numpy())
    saved = batched.state.attitude.numpy().copy()
    single.copy_from(batched, [2, 0, 1])
    np.testing.assert_array_equal(single.state.attitude.numpy(), saved[[2, 0, 1]])
    batched.copy_from(batched, [1, 2, 0])
    np.testing.assert_array_equal(batched.state.attitude.numpy(), saved[[1, 2, 0]])
    batched.reset()
    for name in ("roll_travel", "horizontal_travel", "sideslip", "flips", "reflips"):
        np.testing.assert_array_equal(batched.state.attitude.numpy()[name], 0)


@pytest.mark.parametrize("training", [False, True])
def test_partial_terminal_step_survives_auto_reset(device, training):
    cfg = config(altitude=0, pitch=0)
    weights = RewardConfig(roll_travel_cost=2, horizontal_rotation_cost=2)
    env = RocketEnv(1, cfg, reward_config=weights, device=device) if training else None
    sim = env.sim if training else Simulator(1, cfg, device=device, auto_reset=True)
    set_rate(sim, (0.5, 0.1, 0.0))
    if training:
        env.step(actions(1))
    else:
        sim.step()
    assert sim.terminal.done.numpy()[0]
    assert sim.terminal.time.numpy()[0] == pytest.approx(cfg.dt)
    a = sim.terminal.attitude.numpy()[0]
    assert a["roll_travel"] == pytest.approx(a["peak_roll_rate"] * cfg.dt, rel=1e-4)
    assert a["horizontal_travel"] > 0
    assert sim.state.attitude.numpy()["roll_travel"][0] == 0
    if training:
        terms = env.reward_terms.numpy()[0]
        assert terms[8] == pytest.approx(-2 * a["roll_travel"])
        assert terms[9] == pytest.approx(-2 * a["horizontal_travel"])
        adapter = RslRocketEnv(env)
        assert adapter.final_roll_travel.item() == a["roll_travel"]
        env.reset()
        assert adapter.final_roll_travel.item() == 0


def test_default_costs_preserve_legacy_reward_sum_exactly(device):
    env = RocketEnv(4, config(altitude=40, pitch=0.4, angular_rate_noise=0.2), device=device)
    for _ in range(30):
        env.step(actions(4))
        terms = env.reward_terms.numpy()
        legacy = np.zeros(4, dtype=np.float32)
        for column in terms[:, :8].T:
            legacy += column
        np.testing.assert_array_equal(env.reward.numpy(), legacy)
        np.testing.assert_array_equal(terms[:, 8:], 0)


def test_costs_telescope_to_episode_totals_across_reset(device):
    cfg = config(pitch=0.5, vertical_speed=-80, angular_rate_noise=0.2)
    cfg.max_time = 0.3
    weights = RewardConfig(
        roll_travel_cost=2, horizontal_rotation_cost=2, sideslip_cost=0.5, reflip_cost=5
    )
    env = RocketEnv(2, cfg, reward_config=weights, device=device)
    for _ in range(2):
        total = np.zeros((2, 4))
        for _ in range(3):
            env.step(actions(2))
            total += env.reward_terms.numpy()[:, 8:]
        assert env.sim.terminal.done.numpy().all()
        a = env.sim.terminal.attitude.numpy()
        expected = np.stack(
            [
                -2 * a["roll_travel"],
                -2 * a["horizontal_travel"],
                -0.5 * a["sideslip"],
                -5 * a["reflips"],
            ],
            axis=1,
        )
        np.testing.assert_allclose(total, expected, atol=1e-7)
        assert (total <= 0).all()


def test_spinning_trajectory_costs_more_than_one_clean_flip(device):
    # Near the atmosphere model's ceiling, aerodynamic moments barely perturb the spins.
    cfg = config(altitude=86000, pitch=math.pi / 2, vertical_speed=0)
    cfg.max_altitude, cfg.max_time = 90000, 10.0
    env = RocketEnv(
        2,
        cfg,
        reward_config=RewardConfig(roll_travel_cost=2, horizontal_rotation_cost=2),
        device=device,
    )
    set_rate(env.sim, [(0.72, 0, 0), (0, 0.16, 0)])
    total = np.zeros(2)
    for _ in range(100):
        env.step(actions(2))
        total += env.reward_terms.numpy()[:, 8:10].sum(axis=1)
    assert env.sim.terminal.done.numpy().all()
    # Belly-down roll contributes to both integrals; the clean flip only to horizontal travel.
    np.testing.assert_allclose(total, [-28.8, -3.2], rtol=1e-3)
    np.testing.assert_array_equal(env.sim.terminal.attitude.numpy()["flips"], [0, 1])
    np.testing.assert_array_equal(env.sim.terminal.attitude.numpy()["reflips"], 0)


def test_reflip_cost_fires_only_on_threshold_transition(device):
    env = RocketEnv(1, config(pitch=0), reward_config=RewardConfig(reflip_cost=5), device=device)
    for pitch, expected in [(70, -5), (65, 0), (50, 0), (65, 0), (20, 0), (70, -5)]:
        bodies = env.sim.state.body.numpy()
        bodies["rot"][0] = attitude(math.radians(pitch), 0.0)
        bodies["omega"][0] = 0
        env.sim.state.body.assign(bodies)
        env.step(actions(1))
        assert env.reward_terms.numpy()[0, REWARD_TERMS.index("reflip_cost")] == expected


def test_attitude_metrics_subsets_and_diagnostic_warnings(device):
    env = RslRocketEnv(RocketEnv(3, config(), device=device))
    env.final_status[:] = torch.tensor([LANDED, CRASHED, CRASHED], device=device)
    env.final_steps[:] = 1
    env.final_roll_travel[:] = torch.tensor([math.pi, 0, math.pi / 2], device=device)
    env.final_reflips[:] = torch.tensor([1, 0, 2], device=device)
    tracker = EpisodeTracker(env)
    tracker.step(torch.ones(3, dtype=torch.bool, device=device))
    stats = {k.removeprefix("test/"): v for k, v in tracker.summarize("test").items()}
    assert stats["attitude/roll_travel_deg"] == pytest.approx(90)
    assert stats["landed/attitude/roll_travel_deg"] == pytest.approx(180)
    assert stats["crashed/attitude/roll_travel_deg"] == pytest.approx(45)
    assert stats["attitude/reflips"] == 1
    for key in ATTITUDE_METRICS:
        assert f"attitude/{key}" in stats
    lines = attitude_lines(stats)
    assert "- WARN landed/attitude/roll_travel_deg: 180" in lines
    assert "- WARN attitude/reflips: 1" in lines
    assert "- attitude/roll_travel_deg: 90" in lines
    assert tracker.summarize("test") == {}


def test_flight_peaks_and_event_altitudes(device):
    cfg = config(altitude=2000, vertical_speed=-80, pitch=math.pi / 2, offset=300, offset_noise=0)
    env = RocketEnv(2, cfg, device=device)
    a = env.sim.state.attitude.numpy()[0]
    assert a["start_offset"] == pytest.approx(300, rel=1e-4)
    assert a["flip_altitude"] == -1
    assert a["ignition_altitude"] == -1
    for _ in range(20):
        env.step(actions(2))
    a = env.sim.state.attitude.numpy()[0]
    b = env.sim.state.body.numpy()[0]
    speed = np.linalg.norm(b["vel"])
    assert a["peak_mach"] == pytest.approx(speed / 340, rel=0.05)
    assert a["peak_dynamic_pressure"] == pytest.approx(0.5 * 1.0 * speed**2, rel=0.2)
    # Falling belly-first near terminal speed, drag approaches 1 g.
    assert 0.3 < a["peak_load"] < 1.5
    assert a["flip_altitude"] == -1
    assert a["ignition_altitude"] == -1

    lit = RocketEnv(1, config(lit_throttle=0.6), device=device)
    assert lit.sim.state.attitude.numpy()[0]["ignition_altitude"] == pytest.approx(150)
    unlit = RocketEnv(1, config(), device=device)
    for _ in range(5):
        unlit.step(actions(1))
    assert unlit.sim.state.attitude.numpy()[0]["ignition_altitude"] == -1
    altitude = unlit.sim.state.body.numpy()[0]["pos"][2]
    unlit.step(actions(1, engines=1.0))
    assert unlit.sim.state.attitude.numpy()[0]["ignition_altitude"] == pytest.approx(
        altitude, abs=1.0
    )


def test_failure_metrics_split_tumble_causes_and_skip_missing_events(device):
    env = RslRocketEnv(RocketEnv(4, config(), device=device))
    env.final_status[:] = torch.tensor([LANDED, TUMBLED, TUMBLED, CRASHED], device=device)
    env.final_steps[:] = 1
    env.final_pos[:] = torch.tensor(
        [[0, 0, 0], [0, 0, 900], [0, 0, 300], [0, 0, 0]], device=device, dtype=torch.float32
    )
    env.final_omega[:] = torch.tensor(
        [[0, 0, 0], [2.0, 0, 0], [0, 1.0, 0], [0, 0, 0]], device=device, dtype=torch.float32
    )
    env.final_flip_altitude[:] = torch.tensor([400, -1, -1, 200], device=device)
    env.final_ignition_altitude[:] = -1
    tracker = EpisodeTracker(env)
    tracker.step(torch.ones(4, dtype=torch.bool, device=device))
    stats = {k.removeprefix("test/"): v for k, v in tracker.summarize("test").items()}
    assert stats["tumbled/altitude_m"] == pytest.approx(600)
    assert stats["tumbled/cause_roll_rate"] == 0.5
    assert stats["tumbled/cause_transverse_rate"] == 0.5
    assert stats["flight/flip_altitude_share"] == 0.5
    assert stats["flight/flip_altitude_m"] == pytest.approx(300)
    assert stats["landed/flight/flip_altitude_m"] == pytest.approx(400)
    assert "flight/ignition_altitude_m" not in stats
    lines = failure_lines(stats)
    assert "- tumbles tripping roll_rate: 50%" in lines
    assert "- tumbled/altitude_m: 600" in lines
