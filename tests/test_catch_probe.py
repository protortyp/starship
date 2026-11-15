import csv
import subprocess

import numpy as np
import pytest

from starship.sim import Simulator
from starship.train import ObsConfig, RewardConfig
from starship.train.actions import ACTION_DIM
from starship.train.catch_probe import measure, replay, report_text
from starship.train.env import RocketEnv
from starship.train.runner import export_policy
from tests import test_train
from tests.test_catch import at_plane
from tests.test_policy_catch import policy_config


def test_geometry_tracks_both_pins_and_fin_clearance():
    sim = at_plane(policy_config(), clearance=10)
    initial = measure(sim)
    assert initial["pin_left_above_support_m"] == pytest.approx(10, abs=1e-5)
    assert initial["pin_right_above_support_m"] == pytest.approx(10, abs=1e-5)
    assert initial["pin_alignment_error_m"] < 1e-5
    assert initial["closure_time_s"] == pytest.approx(2)
    assert initial["pin_descent_eta_s"] == pytest.approx(10, abs=1e-5)
    body = sim.state.body.numpy().copy()
    body["pos"][:, 2] += 3
    sim.state.body.assign(body)
    moved = measure(sim)
    assert moved["aft_fin_clearance_m"] == pytest.approx(initial["aft_fin_clearance_m"] - 3)
    assert moved["pin_left_above_support_m"] == pytest.approx(13, abs=1e-5)


def test_failure_label_survives_autoreset_and_copy():
    cfg = policy_config()
    env = RocketEnv(1, cfg, device="cpu")
    env.sim.state.body.assign(at_plane(cfg).state.body)
    env.step(np.zeros((1, ACTION_DIM), dtype=np.float32))
    assert env.sim.terminal_snapshot().catch_failure[0] == 7
    assert env.sim.snapshot().catch_failure[0] == 0
    # Terminal probe reads the ended episode, not the fresh spawn.
    assert abs(measure(env.sim, terminal=True)["pin_left_above_support_m"]) < 1e-4
    source = at_plane(cfg)
    source.step()
    copied = Simulator(1, cfg, device="cpu")
    copied.copy_from(source, [0])
    assert copied.snapshot().catch_failure[0] == 7
    copied.reset()
    assert copied.snapshot().catch_failure[0] == 0


def test_probe_exports_deterministic_intervention_and_reward_trace(tmp_path):
    runner = test_train.make_runner(test_train.compose_cfg("num_envs=8"), tmp_path / "run")
    policy = tmp_path / "policy.pt"
    export_policy(runner.alg.actor, policy)
    cfg = policy_config()
    cfg.max_time = 0.3
    a = replay(policy, cfg, ObsConfig(), RewardConfig(), 0, "open", tmp_path, recording=True)
    b = replay(policy, cfg, ObsConfig(), RewardConfig(), 0, "open", tmp_path)
    assert a == b
    assert a["first_close"] is None
    assert a["status"] == "timeout"
    assert sum(a["reward_totals"].values()) == pytest.approx(a["return"], abs=1e-5)
    with (tmp_path / "open-seed-0.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert all(float(r["executed_close"]) == 0 for r in rows)
    assert all(0 <= float(r["close_probability"]) <= 1 for r in rows)
    assert "obs_31" in rows[0]
    assert "normalized_obs_31" in rows[0]
    assert rows[-1]["status"] == "timeout"
    text = report_text({"policy": str(policy), "stage_name": "test", "episodes": [a]})
    assert "never" in text
    assert "timeout" in text

    recorded = subprocess.run(
        ["rerun", "rrd", "print", str(tmp_path / "open-seed-0.rrd")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for entity in (
        "/decision/close_probability",
        "/state/aft_fin_clearance_m",
        "/inputs/normalized/obs_31",
        "/reward/reward_terminal",
        "/outcome",
    ):
        assert entity in recorded
