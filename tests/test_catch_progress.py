import importlib.util
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("catch_progress", ROOT / "tools/catch_progress.py")
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


def test_upright_rate_cutoff_is_not_labeled_as_tumbling():
    config = SimpleNamespace(
        max_pitch=math.radians(115), max_rate=math.radians(40), max_roll_rate=math.pi / 2
    )
    upright = (0, -math.sqrt(0.5), 0, math.sqrt(0.5))
    assert progress.rotation_cutoff_label(upright, (0, math.radians(40.04), 0), config) == (
        "pitch/yaw rate limit exceeded"
    )
    assert progress.rotation_cutoff_label(upright, (2, 0, 0), config) == "roll rate limit exceeded"
    inverted = (0, math.sqrt(0.5), 0, math.sqrt(0.5))
    assert progress.rotation_cutoff_label(inverted, (0, 0, 0), config) == "tilt limit exceeded"


def test_refresh_interval_and_final_checkpoint():
    assert progress.due(250, 0, 100)
    assert not progress.due(300, 250, 100)
    assert progress.due(350, 250, 100)
    assert not progress.due(250, 250, 100, finished=True)
    assert progress.due(300, 250, 100, finished=True)


def test_checkpoint_selection_excludes_incomplete_and_new_files(tmp_path):
    for index in (49, 99, 149):
        policy = tmp_path / f"policy_{index}.pt"
        policy.touch()
        if index != 99:
            (tmp_path / f"model_{index}.pt").touch()
        if index != 149:
            os.utime(policy, (time.time() - 10, time.time() - 10))
    assert progress.latest_checkpoint(tmp_path).name == "policy_49.pt"
    assert progress.latest_checkpoint(tmp_path, minimum_age=0).name == "policy_149.pt"


def test_body_uses_immutable_images_and_honest_outcomes():
    meta = {
        "iteration": 250,
        "policy_controlled": True,
        "num_envs": 1024,
        "rollout_steps": 128,
        "status": "running",
        "checkpoint": "policy_249.pt",
        "stage_name": "0 approach",
        "updated": "2026-11-15",
        "eval_catch_rate": 0,
        "episodes": [{"seed": 0, "duration_s": 1.2, "outcome": "crashed"}],
    }
    body = progress.pr_body(meta, "abc123", "owner/repo", 100)
    assert "iteration 250" in body
    assert "0.0%" in body
    assert "Seed 0: crashed" in body
    assert "owner/repo/abc123/docs/assets/catch-progress/seed-0.gif" in body
    assert "control its flight and the chopsticks" in body
    assert len(body.split("\n\n")[0]) < 250
    meta["diagnostic_continuation"] = True
    body = progress.pr_body(meta, "abc123", "owner/repo", 100)
    assert "diagnostic continuation past the training rotation cutoff" in body
    assert "0.0% latest training evaluation catch rate" in body


def test_training_status_requires_exact_session_not_updater_prefix(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def fake_run(command, **kwargs):
        assert command == ["tmux", "has-session", "-t", "=starship-catch"]
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(progress.subprocess, "run", fake_run)
    (tmp_path / "console.log").write_text("policy exported to policy.pt\n")
    assert progress.training_status(tmp_path, "starship-catch") == "finished"


def test_promotion_checkpoint_does_not_inherit_previous_stage_catch_rate():
    history = [{"stage": 1, "env_steps": 100}, {"stage": 2, "env_steps": 200}]
    assert progress.evaluation_stage(100, history, 2) == 0
    assert progress.evaluation_stage(101, history, 2) == 1
    assert progress.evaluation_stage(200, history, 2) == 1
    assert progress.evaluation_stage(201, history, 2) == 2
    assert progress.evaluation_stage(50, [], 3) == 3


def test_resumed_evaluation_uses_saved_transition_counter(tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(str(tmp_path)) as writer:
        writer.add_scalar("eval/env/rate/landed", 0.5, 1000)
        writer.add_scalar("eval/env/rate/landed", 0.75, 2000)
    assert progress.evaluation_rate(tmp_path, 10, 10, catch_enabled=False) is None
    assert (
        progress.evaluation_rate(tmp_path, 10, 10, catch_enabled=False, max_env_steps=1500) == 0.5
    )


def test_supervised_checkpoint_counter_is_not_a_ppo_iteration(tmp_path):
    for name in ("policy_demo_100.pt", "model_demo_100.pt"):
        path = tmp_path / name
        path.touch()
        os.utime(path, (time.time() - 10, time.time() - 10))
    assert progress.latest_checkpoint(tmp_path).name == "policy_demo_100.pt"
    assert progress.checkpoint_update(tmp_path / "policy_demo_100.pt") == 100
    assert progress.checkpoint_update(tmp_path / "policy_99.pt") == 100
    meta = {
        "iteration": 100,
        "num_envs": 128,
        "rollout_steps": 128,
        "checkpoint_kind": "supervised",
        "status": "running",
        "checkpoint": "policy_demo_100.pt",
        "stage_name": "ignition",
        "updated": "2026-11-15",
        "episodes": [],
        "eval_catch_rate": 0.25,
    }
    body = progress.pr_body(meta, "abc", "owner/repo", 100)
    assert "supervised update 100" in body
    assert "iteration" not in body
    assert "1,638,400 environment steps" not in body
    assert "25.0%" in body


def test_mixture_playback_uses_angle_target_without_changing_old_range_or_physics(tmp_path):
    import dataclasses
    import json

    from starship.sim.config import SimConfig

    cfg = SimConfig()
    cfg.spawn.pitch = 0.65
    cfg.spawn.pitch_noise = 0.65
    cfg.spawn.altitude = 800
    before = dataclasses.asdict(cfg)
    speed = {
        "pitch": 1.25,
        "pitch_noise": 0.05,
        "altitude": 1000,
        "vertical_speed": -70,
        "vertical_speed_noise": 10,
    }
    angle = {**speed, "pitch": 1.35, "vertical_speed": -60}
    (tmp_path / "mixture-plan.json").write_text(
        json.dumps({"angle_pitch": 1.35, "profiles": [before["spawn"], speed, angle]})
    )
    (old, old_label), (target, label) = progress.episode_profiles(tmp_path, cfg)
    assert old is cfg
    assert old_label == "Earlier range"
    assert dataclasses.asdict(cfg) == before
    assert label == "Target 77deg / 60m/s"
    expected = {**before, "spawn": {**before["spawn"], **angle}}
    assert dataclasses.asdict(target) == expected
    assert target.spawn is not cfg.spawn

    (tmp_path / "mixture-plan.json").write_text(
        json.dumps({"transition_height": 1000, "profiles": [before["spawn"], speed, {}, {}]})
    )
    (_, _), (target, label) = progress.episode_profiles(tmp_path, cfg)
    assert target.spawn.vertical_speed == -70
    assert target.spawn.pitch == 1.25
    assert label == "Target 72deg / 70m/s"
    cfg.catch.enabled = True
    assert progress.episode_profiles(tmp_path, cfg) == [(cfg, None), (cfg, None)]


def test_mixture_body_labels_profiles_without_assigning_old_rate_to_target():
    meta = {
        "iteration": 4450,
        "catch_enabled": False,
        "stage_name": "old range",
        "target_profile": "Target 77deg / 60m/s",
        "eval_catch_rate": 0.99,
        "episodes": [
            {"seed": 0, "profile": "Earlier range", "outcome": "landed"},
            {"seed": 1, "profile": "Target 77deg / 60m/s", "outcome": "crashed"},
        ],
    }
    body = progress.pr_body(meta, "abc123", "owner/repo", 25)
    assert "99.0%" not in body
    assert "Earlier range, seed 0: landed" in body
    assert "Target 77deg / 60m/s, seed 1: crashed" in body
    assert body.count("![") == 2
