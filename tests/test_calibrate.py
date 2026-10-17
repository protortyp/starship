import csv
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from starship.sim.calibrate import FitParameters, fit, rmse, simulate
from starship.sim.config import SimConfig
from starship.sim.simulator import Simulator
from starship.sim.telemetry import Telemetry, load_telemetry


@pytest.fixture
def telemetry_csv(tmp_path):
    path = tmp_path / "tiny.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "MissionTime",
                "FileName",
                "StarshipAltitude_Raw",
                "StarshipAltitude_Corrected",
                "StarshipVelocity_Raw",
                "StarshipVelocity_Corrected",
            ]
        )
        for t in range(61):
            h = [26, 21][t] if t < 2 else max(0, np.ceil(20 - (t - 2) / 2))
            v = 3.6 * max(0, 100 - 10 * max(t - 45, 0))
            raw = "" if t == 10 else 9999 if t == 12 else v
            corrected = 9999 if t == 14 else v
            writer.writerow(["=A1+1", f"frame_{t * 30:06d}.png", h, h, raw, corrected])
    return path


@pytest.mark.parametrize("corrected", [False, True])
def test_load_frame_clock_window_and_quality_mask(telemetry_csv, corrected):
    data = load_telemetry(telemetry_csv, corrected=corrected)
    assert data.start_time == 2
    assert data.burn_time == 45
    assert data.t[-1] + data.start_time == 42
    assert data.altitude_m[0] == 20_000
    assert data.speed_mps[0] == 100
    assert np.all(np.diff(data.t) > 0)
    assert data.repaired[8, 1]
    assert data.repaired[10, 1]
    assert data.repaired[12, 1] == corrected
    assert np.all(data.speed_mps == 100)
    assert not data.repaired[:, 0].any()


def test_loader_rejects_missing_landing(telemetry_csv):
    rows = telemetry_csv.read_text().splitlines()
    telemetry_csv.write_text("\n".join(rows[:40]))
    with pytest.raises(ValueError, match="near-zero"):
        load_telemetry(telemetry_csv)


def test_quantization_and_repaired_channels_are_not_point_targets():
    data = Telemetry(
        np.arange(3.0),
        np.full(3, 1000.0),
        np.full(3, 100.0),
        np.array([[False, False], [False, False], [True, True]]),
    )
    prediction = np.array([[500, 103], [1600, 104], [1e6, 1e6]])
    np.testing.assert_allclose(rmse(data, prediction), [np.sqrt(5000), np.sqrt(12.5)])


@pytest.mark.skipif(platform.machine() != "arm64", reason="pre-change CPU reference is arm64")
def test_neutral_defaults_match_pre_calibration_simulator_exactly():
    # Captured before adding the multipliers, with CPU Warp 1.18.0; includes drift and moments.
    reference = np.load(Path(__file__).parent / "data" / "sim_neutral_reference.npz")
    for name, speed in [("vertical", 0), ("drift", 150)]:
        cfg = SimConfig()
        cfg.spawn.horizontal_speed = speed
        cfg.spawn.pitch_noise = 0.1
        sim = Simulator(3, cfg, device="cpu", seed=41)
        for _ in range(30):
            sim.step(flaps=[0.4, 0.5, 0.9, 0.8])
        for key, value in vars(sim.snapshot()).items():
            np.testing.assert_array_equal(value, reference[f"{name}_{key}"])


def test_cem_recovers_synthetic_drag_and_entry_angle():
    t = np.arange(0, 51.0)
    data = Telemetry(
        t,
        np.full(len(t), 15000.0),
        np.full(len(t), 250.0),
        np.zeros((len(t), 2), dtype=bool),
        altitude_half_width=0,
    )
    truth = FitParameters(1.23, 0.9, 149000, 68)
    trajectory, complete = simulate(data, [list(asdict(truth).values())])
    assert complete.all()
    data.altitude_m = trajectory[0, :, 0]
    data.speed_mps = trajectory[0, :, 1]
    # Mass and flap effectiveness need external priors: scalar telemetry alone is degenerate.
    bounds = np.array([[0.8, 1.6], [0.9, 0.9], [149000, 149000], [50, 85]])
    recovered, errors = fit(data, bounds=bounds, population=20, iterations=8, seed=12)
    assert recovered.crossflow_drag_scale == pytest.approx(truth.crossflow_drag_scale, abs=0.025)
    assert recovered.descent_angle_deg == pytest.approx(truth.descent_angle_deg, abs=0.8)
    assert errors[0] < 20
    assert errors[1] < 0.5


def test_terminated_trajectory_is_not_a_valid_candidate():
    data = Telemetry(
        np.arange(10.0), np.full(10, 50.0), np.full(10, 100.0), np.zeros((10, 2), dtype=bool)
    )
    prediction, complete = simulate(data, [[1, 1, 160000, 90]])
    assert np.isnan(prediction[0, -1]).all()
    assert not complete[0]
    with pytest.raises(RuntimeError, match="no candidate completed"):
        fit(data, population=4, iterations=1)


def test_batched_candidates_match_individual_rollouts():
    data = Telemetry(
        np.arange(6.0), np.full(6, 15000.0), np.full(6, 250.0), np.zeros((6, 2), dtype=bool)
    )
    candidates = np.array([[0.8, 0.7, 120000, 60], [1.3, 1.2, 170000, 75]])
    batch, complete = simulate(data, candidates)
    assert complete.all()
    assert not np.allclose(batch[0], batch[1])
    for i, candidate in enumerate(candidates):
        single, _ = simulate(data, candidate)
        np.testing.assert_array_equal(batch[i], single[0])
