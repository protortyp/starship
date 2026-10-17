import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import warp as wp

from starship.sim import FLYING, SimConfig, Simulator
from starship.sim.aero import sample as aero_sample
from starship.sim.config import WindConfig
from starship.sim.wind import FT, W20_LIGHT, dryden_scales, wind_profile, wind_series
from starship.train import RocketEnv
from starship.train.actions import ACTION_DIM
from starship.train.metrics import EpisodeTracker
from starship.train.rsl_env import RslRocketEnv
from tests.test_sim import belly_flop_controller

DEVICE = "cpu"


def steady(speed: float, **fields) -> WindConfig:
    """The same horizontal wind at every altitude, blowing towards +x."""
    return WindConfig(
        surface_speed=speed,
        surface_speed_min=speed,
        direction_noise=0.0,
        shear_exponent=0.0,
        boundary_layer=1.0e5,
        jet_height=2.0e5,
        **fields,
    )


def with_wind(wind: WindConfig, **spawn) -> SimConfig:
    cfg = SimConfig()
    cfg.wind = wind
    cfg.spawn = replace(cfg.spawn, **spawn)
    return cfg


def test_wind_sampling_leaves_the_spawn_stream_untouched():
    calm = Simulator(8, SimConfig(), device=DEVICE, seed=3)
    windy = Simulator(
        8, with_wind(WindConfig(surface_speed=5.0, turbulence=1.0)), device=DEVICE, seed=3
    )
    a, b = calm.snapshot(), windy.snapshot()
    for key in ("pos", "vel", "rot", "omega"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    np.testing.assert_array_equal(a.wind, 0.0)
    assert np.all(np.linalg.norm(b.wind, axis=1) > 0.0)
    w = windy.state.wind.numpy()
    assert np.all((w["turbulence"] >= 0) & (w["turbulence"] <= 1))
    assert np.all(np.linalg.norm(w["surface"], axis=1) <= 5.0)


def test_moving_with_the_wind_removes_the_aerodynamic_force():
    def horizontal_speed_after_one_step(wind: WindConfig) -> float:
        sim = Simulator(
            1, with_wind(wind, altitude=2000.0, vertical_speed=0.0, pitch=0.0), device=DEVICE
        )
        body = sim.state.body.numpy()
        body["vel"][0] = (10.0, 0.0, 0.0)
        sim.state.body.assign(body)
        sim.step()
        return sim.snapshot().vel[0]

    in_wind = horizontal_speed_after_one_step(steady(10.0))
    in_calm = horizontal_speed_after_one_step(WindConfig())
    assert in_wind[0] == pytest.approx(10.0, abs=1e-5)
    assert in_wind[1] == pytest.approx(0.0, abs=1e-6)
    assert in_calm[0] < 10.0 - 1e-3


def test_passive_descent_drifts_downwind():
    sims = [
        Simulator(1, with_wind(w, altitude=6000.0), device=DEVICE, seed=7)
        for w in (WindConfig(), steady(10.0))
    ]
    for _ in range(300):
        for sim in sims:
            sim.step(flaps=belly_flop_controller(sim))
    calm, windy = (sim.snapshot() for sim in sims)
    assert calm.status[0] == windy.status[0] == FLYING
    drift = windy.pos[0] - calm.pos[0]
    assert drift[0] > 50.0
    assert abs(drift[1]) < 0.6 * drift[0]  # sideslip couples some drift across the wind
    assert windy.vel[0, 0] - calm.vel[0, 0] > 3.0


def test_mean_profile_shears_in_the_boundary_layer_and_peaks_at_the_jet():
    cfg = WindConfig(
        surface_speed=10.0,
        surface_speed_min=10.0,
        direction_noise=0.0,
        jet_speed=30.0,
        jet_speed_min=30.0,
        jet_direction_noise=0.0,
    )
    z = np.array([0.0, 10.0, 100.0, 500.0, 1000.0, 3000.0, 12_000.0, 20_000.0, 22_000.0, 25_000.0])
    v = wind_profile(cfg, z)
    speed = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(v[:, 1:], 0.0, atol=1e-5)
    assert np.all(np.diff(speed[:7]) > 0)
    assert speed[1] == pytest.approx(10.0)
    assert speed[2] == pytest.approx(10.0 * 10.0**0.14, rel=1e-5)
    assert speed[4] == pytest.approx(10.0 * 100.0**0.14, rel=1e-5)
    assert speed[6] == pytest.approx(30.0)
    assert speed[7] == pytest.approx(6.0, rel=1e-5)
    np.testing.assert_allclose(speed[8:], 0.0)


@wp.kernel
def _scales_kernel(level: float, z: wp.array(dtype=float), out: wp.array(dtype=wp.vec4)):
    i = wp.tid()
    out[i] = dryden_scales(level, z[i])


def scales(level, altitudes):
    z = wp.array(np.asarray(altitudes, dtype=np.float32), dtype=float, device=DEVICE)
    out = wp.zeros(len(z), dtype=wp.vec4, device=DEVICE)
    wp.launch(_scales_kernel, dim=len(z), inputs=[level, z, out], device=DEVICE)
    return out.numpy()


def test_dryden_scales_follow_mil_f_8785c():
    low, blend_top, high = scales(1.0, [100.0, 1000.0 * FT, 3000.0])
    k = (0.177 + 0.000823 * 100.0 / FT) ** 0.4
    sigma_w = 0.1 * W20_LIGHT
    np.testing.assert_allclose(low, [sigma_w / k, 100.0 / k**3, sigma_w, 100.0], rtol=1e-5)
    np.testing.assert_allclose(blend_top, [sigma_w, 1000.0 * FT, sigma_w, 1000.0 * FT], rtol=1e-5)
    # Figure 7, light curve, between the 7500 ft (6.7 ft/s) and 15000 ft (4.6 ft/s) columns.
    h_ft = 3000.0 / FT
    light = (6.7 + (h_ft - 7500.0) / 7500.0 * (4.6 - 6.7)) * FT
    np.testing.assert_allclose(high, [light, 1750.0 * FT, light, 1750.0 * FT], rtol=1e-4)
    severe = scales(3.0, [3000.0])[0]
    assert severe[0] == pytest.approx(
        (23.6 + (h_ft - 7500.0) / 7500.0 * (22.1 - 23.6)) * FT, rel=1e-4
    )
    assert np.all(scales(2.0, [3000.0])[0][[0, 2]] > high[[0, 2]])
    assert np.all(scales(0.0, [50.0, 5000.0])[:, [0, 2]] == 0.0)


@pytest.mark.parametrize(("altitude", "speed"), [(100.0, 50.0), (3000.0, 100.0)])
def test_dryden_turbulence_has_the_specified_rms_and_correlation(altitude, speed):
    cfg = WindConfig(turbulence=1.0, turbulence_min=1.0)
    dt = 0.01
    series = wind_series(cfg, pos=(0, 0, altitude), vel=(0, 0, -speed), steps=20000, envs=32, dt=dt)
    sigma_h, scale_h, sigma_w, scale_w = scales(1.0, [altitude])[0]
    for axis, sigma, scale in ((0, sigma_h, scale_h), (1, sigma_h, scale_h), (2, sigma_w, scale_w)):
        y = series[:, :, axis].astype(np.float64)
        y -= y.mean()
        variance = (y * y).mean()
        assert math.sqrt(variance) == pytest.approx(sigma, rel=0.08)
        # The MIL-F-8785C transverse spectrum has R(tau) = (1 - x/2) e^-x with x = V tau / L.
        lag = round(scale / speed / dt)
        correlation = (y[:, lag:] * y[:, :-lag]).mean() / variance
        assert correlation == pytest.approx(0.5 * math.exp(-1.0), abs=0.05)
    assert np.abs(series.mean(axis=(0, 1))).max() < 0.15 * sigma_h


def test_gust_is_a_one_minus_cosine_pulse():
    cfg = WindConfig(gust_speed=5.0, gust_width=100.0, gust_interval=1e-3)
    speed = 50.0
    series = wind_series(cfg, pos=(0, 0, 1000.0), vel=(0, 0, -speed), steps=400, envs=1)[0]
    norm = np.linalg.norm(series, axis=1)
    assert np.all(series[:, 2] == 0.0)
    assert norm.max() <= 5.0
    end = int(np.argmax(norm[1:] < 1e-4)) + 1  # first return to calm closes the first pulse
    steps_per_metre = 1.0 / (speed * 0.01)
    assert 50.0 * steps_per_metre <= end <= 150.0 * steps_per_metre
    amplitude = norm[:end].max()
    assert int(np.argmax(norm[:end])) == pytest.approx(end / 2, abs=2)
    shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(1, end + 1) / end))
    np.testing.assert_allclose(norm[:end], amplitude * shape, atol=0.03 * amplitude)


def test_shedding_side_force_is_perpendicular_to_the_crossflow_and_scaled_from_it():
    cfg = SimConfig().vehicle
    cfg.aero.flap_force_scale = 0.0  # hull only
    velocity = (0.0, 0.0, -85.0)  # crossflow along -b3: the belly faces the flow
    calm = aero_sample(velocity, cfg=cfg, device=DEVICE)[0]
    shed = aero_sample(velocity, shed=(0.2, math.pi / 2), cfg=cfg, device=DEVICE)[0]
    quiet = aero_sample(velocity, shed=(0.2, 0.0), cfg=cfg, device=DEVICE)[0]
    reversed_ = aero_sample(velocity, shed=(0.2, 1.5 * math.pi), cfg=cfg, device=DEVICE)[0]
    np.testing.assert_allclose(quiet, calm, rtol=1e-6, atol=1e-3)
    side = shed - calm
    assert side[1] == pytest.approx(0.2 * calm[2], rel=1e-4)  # b1 x (-b3) = +b2
    np.testing.assert_allclose(side[[0, 2]], 0.0, atol=1e-3 * abs(calm[2]))
    np.testing.assert_allclose(reversed_ - calm, -side, rtol=1e-4, atol=1e-3)


def test_shedding_phase_advances_at_the_strouhal_frequency():
    cfg = with_wind(WindConfig(shedding_amplitude=0.1, shedding_strouhal=0.25), altitude=4000.0)
    sim = Simulator(2, cfg, device=DEVICE, seed=5)
    phases, crossflow = [], []
    for _ in range(40):
        s = sim.snapshot()
        phases.append(sim.state.wind.numpy()["shed_phase"].copy())
        x, y, z, w = s.rot.T
        axis = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
        along = np.sum(s.vel * axis, axis=1)
        crossflow.append(np.sqrt(np.maximum(np.sum(s.vel**2, axis=1) - along**2, 0.0)))
        sim.step(flaps=belly_flop_controller(sim))
    phases, crossflow = np.unwrap(np.array(phases), axis=0), np.array(crossflow)
    # Trapezoidal integral of 2 pi St V_n / D over the 0.1 s control steps.
    expected = 2 * np.pi * 0.25 / 9.0 * 0.1 * (0.5 * (crossflow[1:] + crossflow[:-1])).sum(axis=0)
    np.testing.assert_allclose(phases[-1] - phases[0], expected, rtol=0.02)
    assert phases[0, 0] != phases[0, 1]  # random phase per env
    assert np.all(sim.snapshot().status == FLYING)


def test_flap_noise_has_the_configured_rms_and_correlation_time():
    cfg = with_wind(WindConfig(flap_noise=0.15, flap_noise_time=0.2), altitude=5000.0)
    sim = Simulator(256, cfg, device=DEVICE, seed=11)
    noise = []
    for _ in range(60):
        sim.step(flaps=belly_flop_controller(sim))
        noise.append(sim.state.wind.numpy()["flap_noise"].copy())
    noise = np.array(noise)  # (steps, envs, 4)
    assert noise.std() == pytest.approx(0.15, rel=0.08)
    assert abs(noise.mean()) < 0.01
    lag_one = (noise[1:] * noise[:-1]).mean() / (noise * noise).mean()
    assert lag_one == pytest.approx(math.exp(-0.1 / 0.2), abs=0.05)
    assert np.all(sim.snapshot().status == FLYING)


def test_copies_keep_the_wind_sample_but_not_the_random_stream():
    cfg = with_wind(WindConfig(surface_speed=8.0, turbulence=2.0))
    source = Simulator(3, cfg, device=DEVICE, seed=1)
    copies = Simulator(4, cfg, device=DEVICE, seed=2)
    copies.copy_from(source, [2, 2, 0, 1])
    src, dst = source.state.wind.numpy(), copies.state.wind.numpy()
    for key in ("surface", "jet", "shear", "turbulence", "dryden1", "wind"):
        np.testing.assert_array_equal(dst[key], src[key][[2, 2, 0, 1]])
    assert dst["rng"][0] != dst["rng"][1]
    assert not np.array_equal(dst["rng"], src["rng"][[2, 2, 0, 1]])


def test_env_reports_the_episode_wind_in_terminal_views_and_metrics():
    cfg = with_wind(
        WindConfig(surface_speed=5.0, turbulence=1.0),
        altitude=60.0,
        vertical_speed=-30.0,
        pitch=0.0,
    )
    cfg.max_time = 20.0
    env = RslRocketEnv(RocketEnv(8, cfg, device=DEVICE, seed=2))
    tracker = EpisodeTracker(env)
    sampled = np.linalg.norm(env.env.sim.state.wind.numpy()["surface"], axis=1)
    done_once = torch.zeros(8, dtype=torch.bool)
    for _ in range(200):
        _, _, dones, _ = env.step(torch.zeros(8, ACTION_DIM))
        just_done = dones.bool() & ~done_once
        if just_done.any():
            final = env.env.final_wind.numpy()
            np.testing.assert_allclose(
                final[just_done.numpy(), 0], sampled[just_done.numpy()], rtol=1e-6
            )
            assert np.all((final[just_done.numpy(), 1] >= 0) & (final[just_done.numpy(), 1] <= 1))
        tracker.step(dones, record=~done_once)
        done_once |= dones.bool()
        if done_once.all():
            break
    assert done_once.all()
    summary = tracker.summarize("train/env")
    assert summary["train/env/wind/surface_speed"] == pytest.approx(sampled.mean(), rel=1e-5)
    assert 0 < summary["train/env/wind/turbulence"] < 1
    assert {"train/env/rate/landed_calm", "train/env/rate/landed_windy"} & set(summary)
    assert "train/env/crashed/wind/surface_speed" in summary
