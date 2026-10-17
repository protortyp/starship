import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from starship.render.export import (
    Samples,
    build_export,
    record_episode,
    resample,
    vehicle_geometry,
    write_json,
    write_usda,
)
from starship.sim import LANDED, SimConfig
from starship.sim.mass import make_mass_params, sample
from starship.train import ObsConfig, RewardConfig
from starship.train.actions import ACTION_DIM


class EnginesOffPolicy:
    """Stand-in for an rsl_rl actor: engines off, every continuous head at its range centre."""

    def __call__(self, obs, stochastic_output=False):
        return torch.zeros(obs.batch_size[0], ACTION_DIM)

    def reset(self, dones=None):
        pass


def hover_config() -> SimConfig:
    cfg = SimConfig()
    tanks = [
        [
            cfg.vehicle.tanks[n].capacity * f
            for n, f in (
                ("main_lox", cfg.spawn.main_fill),
                ("main_ch4", cfg.spawn.main_fill),
                ("header_lox", cfg.spawn.header_fill),
                ("header_ch4", cfg.spawn.header_fill),
            )
        ]
    ]
    com = float(sample(make_mass_params(cfg.vehicle), tanks, device="cpu")[0, 1])
    # Upright, tail 5 cm above the pad and sinking gently: a soft touchdown on the first step.
    cfg.spawn = replace(cfg.spawn, altitude=com + 0.05, vertical_speed=-1.0, pitch=0.0)
    return cfg


def test_recorded_touchdown_exports_landed_frames(tmp_path):
    cfg = hover_config()
    samples = record_episode(
        EnginesOffPolicy(), (cfg, ObsConfig(), RewardConfig()), seed=0, device="cpu"
    )
    assert samples.status == LANDED
    k = len(samples.time)
    assert k >= 2
    assert samples.com_position.shape == (k, 3)
    assert samples.quaternion.shape == (k, 4)
    assert samples.flaps.shape == (k, 4)
    assert samples.engine["spool"].shape == (k, 3)

    data = build_export(samples, cfg, fps=60.0, meta={"seed": 0})
    f = data["frames"]
    n = data["meta"]["frames"]
    assert data["meta"]["landed"] is True
    assert data["meta"]["status"] == "landed"
    t = np.array(f["time"])
    assert len(t) == n
    assert (np.diff(t) > 0).all()
    assert np.allclose(np.diff(t), 1 / 60, atol=1e-6)
    assert np.array(f["position"]).shape == (n, 3)
    assert np.array(f["quaternion"]).shape == (n, 4)
    assert np.array(f["flaps"]).shape == (n, 4)
    for key in ("spool", "throttle", "gimbal2", "gimbal3", "thrust"):
        assert np.array(f[key]).shape == (n, 3)
    assert f["status"][-1] == LANDED
    assert all(s == 0 for s in f["status"][:-1])
    # The engine-plane origin ends up on the pad, not the centre of mass.
    assert abs(f["altitude"][-1]) < 0.5
    assert f["com_position"][-1][2] > 20.0

    out = write_json(data, tmp_path / "landing.json")
    loaded = json.loads(out.read_text())
    assert loaded["vehicle"]["length"] == cfg.vehicle.hull.length
    assert len(loaded["vehicle"]["engines"]) == 3
    assert len(loaded["vehicle"]["flaps"]) == 4
    usd = write_usda(data, tmp_path / "landing.usda")
    text = usd.read_text()
    assert text.startswith("#usda 1.0")
    assert f"endTimeCode = {n - 1}" in text
    assert text.count('def Xform "Engine') == 3
    assert text.count('def Xform "Flap') == 4


def test_resample_hermite_matches_constant_velocity_motion():
    dt, k = 0.1, 11
    t = np.arange(k) * dt
    vel = np.array([3.0, -2.0, -10.0])
    pos = 100.0 + t[:, None] * vel
    q = np.tile([0.0, 0.0, 0.0, 1.0], (k, 1))
    samples = Samples(
        dt=dt,
        time=t,
        com_position=pos,
        com_station=np.full(k, 25.0),
        velocity=np.tile(vel, (k, 1)),
        quaternion=q,
        omega=np.zeros((k, 3)),
        flaps=np.linspace(0.0, 1.0, k)[:, None] * np.ones((1, 4)),
        engine={n: np.ones((k, 3)) for n in ("spool", "throttle", "gimbal2", "gimbal3")},
        status=LANDED,
    )
    f = resample(samples, fps=60.0)
    n = len(f["time"])
    assert n == 61
    assert (np.diff(f["time"]) > 0).all()
    expected = 100.0 + f["time"][:, None] * vel
    np.testing.assert_allclose(f["com_position"], expected, atol=1e-9)
    # Identity attitude: the engine plane sits one CoM station below the centre of mass along x.
    np.testing.assert_allclose(f["position"], expected - [25.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(f["velocity"], np.tile(vel, (n, 1)), atol=1e-9)
    np.testing.assert_allclose(f["flaps"][:, 0], f["time"], atol=1e-9)
    assert f["status"][-1] == LANDED


def test_vehicle_geometry_matches_config():
    cfg = SimConfig()
    g = vehicle_geometry(cfg)
    assert g["hull_profile"][0] == [0.0, 4.5]
    assert g["hull_profile"][-1][0] == pytest.approx(52.1)
    assert g["hull_profile"][-1][1] == pytest.approx(0.0, abs=1e-6)
    exits = np.array([e["exit"] for e in g["engines"]])
    np.testing.assert_allclose(np.linalg.norm(exits[:, 1:], axis=1), 1.15, atol=1e-6)
    assert [round(f["hinge"][0], 1) for f in g["flaps"]] == [6.5, 6.5, 43.0, 43.0]
