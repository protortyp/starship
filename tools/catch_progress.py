"""Render actual Rerun GIFs and refresh one draft PR body; never creates PR comments.

Run with the repository's Python environment. Keep this worktree on the progress branch.
--render-only prepares assets without committing or making any GitHub changes.
"""

import argparse
import dataclasses
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/catch-progress"
# A shared editable environment may point at another worktree. Render and test this one.
sys.path.insert(0, str(ROOT / "src"))


class PublisherStopped(RuntimeError):
    pass


def command(*args, **kwargs):
    return subprocess.check_output(args, cwd=ROOT, text=True, **kwargs).strip()


def checkpoint_update(path):
    number = int(path.stem.rsplit("_", 1)[1])
    return number if path.name.startswith("policy_demo_") else number + 1


def latest_checkpoint(run, minimum_age=5):
    candidates = [
        p
        for p in [*run.glob("policy_[0-9]*.pt"), *run.glob("policy_demo_[0-9]*.pt")]
        if (run / p.name.replace("policy_", "model_")).exists()
        and time.time() - p.stat().st_mtime > minimum_age
    ]
    return max(candidates, key=checkpoint_update, default=None)


def due(iteration, previous, every, finished=False):
    return iteration > previous and (previous == 0 or iteration - previous >= every or finished)


def rotation_cutoff_label(rotation, omega, config):
    """Describe the configured termination condition, not physical recoverability."""
    x, y, z, w = rotation
    axis_z = 2 * (x * z - w * y)
    tilt = math.acos(max(-1.0, min(1.0, axis_z)))
    reasons = []
    if tilt > config.max_pitch:
        reasons.append("tilt limit exceeded")
    if math.hypot(omega[1], omega[2]) > config.max_rate:
        reasons.append("pitch/yaw rate limit exceeded")
    if abs(omega[0]) > config.max_roll_rate:
        reasons.append("roll rate limit exceeded")
    return "; ".join(reasons) or "attitude/rate cutoff"


def training_status(run, session):
    active = (
        subprocess.run(["tmux", "has-session", "-t", f"={session}"], capture_output=True).returncode
        == 0
    )
    if active:
        return "running"
    log = (run / "console.log").read_text(errors="replace")
    return (
        "finished"
        if any(
            marker in log
            for marker in ("policy exported to", "best standalone policy:", f"FINISHED {run}")
        )
        else "stopped (check training log)"
    )


def episode_profiles(run, cfg):
    """Render the old range and the active spawn-only mixture target."""
    plan_path = run / "mixture-plan.json"
    if cfg.catch.enabled or not plan_path.exists():
        return [(cfg, None), (cfg, None)]
    plan = json.loads(plan_path.read_text())
    profiles = plan.get("profiles", [])
    if "angle_pitch" in plan:
        target = next(p for p in profiles if p["pitch"] == plan["angle_pitch"])
    elif "transition_height" in plan and len(profiles) >= 4:
        target = profiles[1]
    else:
        return [(cfg, None), (cfg, None)]
    fields = ("pitch", "pitch_noise", "altitude", "vertical_speed", "vertical_speed_noise")
    spawn = dataclasses.replace(cfg.spawn, **{key: target[key] for key in fields})
    label = f"Target {math.degrees(spawn.pitch):.0f}deg / {abs(spawn.vertical_speed):g}m/s"
    return [(cfg, "Earlier range"), (dataclasses.replace(cfg, spawn=spawn), label)]


def render(policy_path, run, output):
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb
    import torch
    from rerun.experimental import ViewerClient

    from starship.render.rerun_viewer import flap_geometry, log_scene, log_step, run_configs
    from starship.sim.catch import CATCH_FAILURES
    from starship.sim.mass import make_mass_params
    from starship.train.checkpoint import load_policy
    from starship.train.env import RocketEnv
    from starship.train.rsl_env import RslRocketEnv

    torch.set_num_threads(2)
    update = checkpoint_update(policy_path)
    kind = "supervised" if policy_path.name.startswith("policy_demo_") else "ppo"
    checkpoint = torch.load(
        run / policy_path.name.replace("policy_", "model_"), map_location="cpu", weights_only=False
    )
    stage = checkpoint["infos"]["curriculum"]["stage"]
    cfg, obs_cfg, reward, stage_name = run_configs(run, stage)
    # Show the consequence of crossing the training cutoff, not a clip that ends mid-turn.
    # Actuators, collision geometry, capture conditions and timeout remain unchanged.
    profiles = episode_profiles(run, cfg)
    metadata = {
        "iteration": update,
        "checkpoint_kind": kind,
        "total_env_steps": checkpoint["infos"].get("total_env_steps"),
        "checkpoint": policy_path.name,
        "run": f"{run.parent.name}/{run.name}",
        "catch_enabled": cfg.catch.enabled,
        "moving_arms": cfg.catch.moving,
        "policy_controlled": cfg.catch.policy_controlled,
        "crash_penalty": reward.crash_penalty,
        "tumble_penalty": reward.tumble_penalty,
        "contact_penalty": reward.catch_contact_penalty,
        "observation_channels": 38 if obs_cfg.catch_geometry else 32,
        "stage": stage,
        "stage_name": stage_name,
        "updated": datetime.now(UTC).isoformat(),
        "episodes": [],
        "diagnostic_continuation": True,
        "profile_version": 1,
        "target_profile": profiles[1][1],
    }
    output.mkdir(parents=True, exist_ok=True)
    for seed in (0, 1):
        episode_cfg, profile_label = profiles[seed]
        playback_cfg = dataclasses.replace(episode_cfg, terminate_on_attitude_limits=False)
        env = RslRocketEnv(RocketEnv(1, playback_cfg, obs_cfg, reward, device="cpu", seed=seed))
        policy = load_policy(policy_path, "cpu", obs_dim=env.env.obs_dim)
        policy.reset()
        observation = env.get_observations()
        snapshots = [env.env.sim.snapshot()]
        first_cutoff = None
        with torch.inference_mode():
            for _ in range(env.max_episode_length):
                observation, _, done, _ = env.step(policy(observation, stochastic_output=False))
                snap = env.env.sim.snapshot()
                if done.item():
                    terminal = env.env.sim.terminal_snapshot()
                    snap = dataclasses.replace(
                        snap,
                        **{
                            field.name: getattr(terminal, field.name)
                            for field in dataclasses.fields(snap)
                            if hasattr(terminal, field.name)
                        },
                    )
                snapshots.append(snap)
                reason = rotation_cutoff_label(snap.rot[0], snap.omega[0], cfg)
                if first_cutoff is None and reason != "attitude/rate cutoff":
                    first_cutoff = {"time_s": float(snap.time[0]), "reason": reason}
                if done.item():
                    break
        duration = float(snapshots[-1].time[0])
        outcome = {
            1: "caught" if cfg.catch.enabled else "landed",
            2: "crashed",
            3: "out of bounds",
            4: "timeout",
            5: rotation_cutoff_label(snapshots[-1].rot[0], snapshots[-1].omega[0], cfg),
        }.get(int(snapshots[-1].status[0]), "unknown")
        if cfg.catch.enabled and int(snapshots[-1].status[0]) == 2:
            outcome = "crashed: " + CATCH_FAILURES[int(snapshots[-1].catch_failure[0])]
        final_vz = float(snapshots[-1].vel[0, 2])
        qx, qy, qz, qw = snapshots[-1].rot[0]
        final_tilt = math.degrees(math.acos(max(-1.0, min(1.0, 2 * (qx * qz - qw * qy)))))
        # Explicitly label terminal measurements, not the current frame's telemetry.
        end_label = outcome.split(":", 1)[0]
        overlay = output / f"overlay-{seed}.txt"
        overlay.write_text(
            f"{profile_label or f'Seed {seed}'} | {duration:.1f}s | End: {end_label}\n"
            f"Final vz {final_vz:.1f}m/s | tilt {final_tilt:.1f}deg"
        )
        opening = [float(np.degrees(s.arm_angle[0])) for s in snapshots]
        metadata["episodes"].append(
            {
                "seed": seed,
                "profile": profile_label,
                "spawn": dataclasses.asdict(episode_cfg.spawn),
                "duration_s": round(duration, 2),
                "outcome": outcome,
                "training_cutoff": first_cutoff,
                "final_vertical_speed_m_s": round(final_vz, 2),
                "final_tilt_deg": round(final_tilt, 2),
                "arm_open_min_deg": round(min(opening), 1),
                "arm_open_max_deg": round(max(opening), 1),
            }
        )
        # A fixed tower view and moving chase view in the same GIF. No successful-episode filtering.
        layout = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Tower" if cfg.catch.enabled else "Landing area",
                    origin="/world",
                    contents="/world/**",
                    line_grid=False,
                    eye_controls=rrb.EyeControls3D(
                        position=(150, -180, 150), look_target=(0, 0, 90)
                    ),
                ),
                rrb.Spatial3DView(
                    name="Ship",
                    origin="/world/chase",
                    contents="/world/**",
                    line_grid=False,
                    eye_controls=rrb.EyeControls3D(position=(70, -80, 30), look_target=(0, 0, 0)),
                ),
            ),
            collapse_panels=True,
        )
        rec = rr.RecordingStream(f"catch-iteration-{update}-seed-{seed}")
        frames = output / f"frames-{seed}"
        frames.mkdir(exist_ok=True)
        # The headless viewer runs independently of the desktop; terminate it after each clip.
        with (
            ViewerClient.spawn(
                headless=True,
                port=9891,
                memory_limit="512MiB",
                executable_path=str(Path(sys.executable).parent / "rerun"),
            ) as viewer,
            rec,
        ):
            rr.connect_grpc(viewer.url)
            rr.send_blueprint(layout)
            log_scene(cfg, 1)
            flaps = flap_geometry(cfg)
            mass = make_mass_params(cfg.vehicle)
            trails = [[]]
            for snap in snapshots:
                rr.set_time("sim_time", duration=float(snap.time[0]))
                log_step(cfg, snap, flaps, trails, mass)
            rec.flush()
            deadline = time.monotonic() + 20
            while not viewer.viewer_state().views:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Rerun blueprint did not load")
                time.sleep(0.1)
            time.sleep(0.5)
            for frame, t in enumerate(np.linspace(0, duration, 48)):
                viewer.set_time("sim_time", duration=float(t))
                time.sleep(0.06)
                viewer.save_screenshot(str((frames / f"frame-{frame:03d}.png").resolve()))
            rr.disconnect()
        # Keep each committed asset below the repository's 500 KiB file limit.
        gif = ASSETS / f"seed-{seed}.gif"
        ASSETS.mkdir(parents=True, exist_ok=True)
        for width in (960, 640, 480, 320):
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    "6",
                    "-i",
                    str(frames / "frame-%03d.png"),
                    "-filter_complex",
                    f"[0:v]tpad=stop_mode=clone:stop_duration=1.5,"
                    f"scale={width}:-1:flags=lanczos,"
                    f"drawtext=textfile='{overlay.resolve()}':fontcolor=white:"
                    f"fontsize={max(10, width // 48)}:x=8:y=8:box=1:"
                    "boxcolor=black@0.8:boxborderw=5,"
                    "split[a][b];[a]palettegen=max_colors=64[p];"
                    "[b][p]paletteuse=dither=none",
                    "-loop",
                    "0",
                    str(gif),
                ],
                check=True,
            )
            if gif.stat().st_size <= 500 * 1024:
                break
        else:
            raise RuntimeError(f"GIF exceeds repository size limit: {gif}")
    return metadata


def pr_body(meta, sha, repository, every):
    rate = meta.get("eval_catch_rate")
    task = "catch" if meta.get("catch_enabled", True) else "landing"
    catch = (
        "evaluation pending"
        if rate is None
        else f"{rate:.1%} latest training evaluation {task} rate"
    )
    label = "supervised update" if meta.get("checkpoint_kind") == "supervised" else "iteration"
    continuation = (
        " GIFs show diagnostic continuation past the training rotation cutoff."
        if meta.get("diagnostic_continuation")
        else ""
    )
    purpose = (
        "Training Starship to control its flight and the chopsticks for an end-to-end catch. "
        if meta.get("catch_enabled", True)
        else "Rebuilding ground landings with current encodings before chopstick catches. "
    )
    current = (
        f"Currently: {meta['target_profile']} ({label} {meta['iteration']:,}); "
        "GIFs show the earlier range and current target."
        if meta.get("target_profile")
        else f"Currently: {meta['stage_name']}, {catch} ({label} {meta['iteration']:,})."
    )
    lines = [
        purpose + current + continuation,
        "",
    ]
    for episode in meta["episodes"]:
        seed = episode["seed"]
        caption = f"{episode['profile']}, seed {seed}" if episode.get("profile") else f"Seed {seed}"
        lines += [
            f"![{caption}: {episode['outcome']}](https://raw.githubusercontent.com/"
            f"{repository}/{sha}/docs/assets/catch-progress/seed-{seed}.gif)",
            "",
        ]
    return "\n".join(lines)


def evaluation_stage(step, history, current_stage):
    # Promotion follows evaluation at the same step. That evaluation belongs to the old stage.
    stage = history[0]["stage"] - 1 if history else current_stage
    for promotion in history:
        if promotion["env_steps"] < step:
            stage = promotion["stage"]
    return stage


def evaluation_rate(run, iteration, batch, stage=None, catch_enabled=True, max_env_steps=None):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(run)).Reload()
    tag = "eval/env/rate/caught" if catch_enabled else "eval/env/rate/landed"
    if tag not in events.Tags()["scalars"]:
        return None
    history_path = run / "curriculum.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    limit = iteration * batch if max_env_steps is None else max_env_steps
    values = [
        event.value
        for event in events.Scalars(tag)
        if event.step <= limit
        and (stage is None or evaluation_stage(event.step, history, stage) == stage)
    ]
    return values[-1] if values else None


def refresh(args, policy_path, status):
    from omegaconf import OmegaConf

    config = OmegaConf.load(args.run / ".hydra/config.yaml")
    work = args.run / "pr-progress"
    meta = render(policy_path, args.run, work)
    meta.update(
        status=status,
        num_envs=config.num_envs,
        rollout_steps=config.runner.num_steps_per_env,
        initial_value=None
        if config.get("checkpoint") or meta.get("checkpoint_kind") == "supervised"
        else config.runner.algorithm.get("initial_value"),
        warmstart=(args.run / "warmstart.json").exists(),
    )
    if meta.get("checkpoint_kind") == "supervised":
        evaluation = json.loads((args.run / f"demo_{meta['iteration']}.json").read_text())
        meta["eval_catch_rate"] = evaluation["catch_rate"]
    else:
        meta["eval_catch_rate"] = evaluation_rate(
            args.run,
            meta["iteration"],
            meta["num_envs"] * meta["rollout_steps"],
            meta["stage"],
            meta["catch_enabled"],
            max_env_steps=meta.get("total_env_steps"),
        )
    (ASSETS / "latest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def publish(args, meta):
    if command("git", "branch", "--show-current") != "ce/catch-training-progress":
        raise PublisherStopped("Progress publisher must stay on ce/catch-training-progress")
    state = json.loads(
        command(
            "gh", "pr", "view", str(args.pr), "--repo", args.repository, "--json", "state,isDraft"
        )
    )
    if state["state"] != "OPEN" or not state["isDraft"]:
        raise PublisherStopped("Progress PR is no longer an open draft; publisher stopped")
    # Project instructions require these checks before every commit, including asset refreshes.
    check_env = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(Path(sys.executable).parent.parent),
        "UV_NO_SYNC": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    check_log = args.run / "pr-progress/checks.log"
    with check_log.open("w") as output:
        for task in ("check", "test"):
            subprocess.run(
                ["just", task],
                cwd=ROOT,
                env=check_env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
    staged = command("git", "diff", "--cached", "--name-only").splitlines()
    if any(not name.startswith("docs/assets/catch-progress/") for name in staged):
        raise PublisherStopped(
            "Unrelated staged changes; refusing to include them in a progress commit"
        )
    expected_remote = command("git", "rev-parse", "refs/remotes/origin/ce/catch-training-progress")
    command("git", "add", "docs/assets/catch-progress")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        command(
            "git",
            "commit",
            "--amend",
            "-m",
            f"chore(train): render catch checkpoint {meta['iteration']} [skip ci]",
        )
    command(
        "git",
        "push",
        f"--force-with-lease=refs/heads/ce/catch-training-progress:{expected_remote}",
        "origin",
        "HEAD:ce/catch-training-progress",
    )
    sha = command("git", "rev-parse", "HEAD")
    body = args.run / "pr-progress/body.md"
    body.write_text(pr_body(meta, sha, args.repository, args.every))
    # This is the only GitHub write: update the body, never create comments.
    command("gh", "pr", "edit", str(args.pr), "--repo", args.repository, "--body-file", str(body))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--pr", type=int)
    parser.add_argument("--repository", default="protortyp/starship")
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--session", default="starship-catch")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: sys.exit(0))
    args.run = args.run.resolve()
    if args.every < 1 or (not args.render_only and not args.pr):
        parser.error("positive --every and --pr required unless --render-only")
    work = args.run / "pr-progress"
    work.mkdir(exist_ok=True)
    lock = (work / "updater.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = work / "published.json"
    while True:
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        status = training_status(args.run, args.session)
        # A completed trainer has finished both checkpoint writes; include its final save.
        path = latest_checkpoint(args.run, minimum_age=0 if status == "finished" else 5)
        if path is not None and (
            args.render_only
            or previous.get("profile_version") != 1
            or not previous.get("diagnostic_continuation", False)
            or due(
                checkpoint_update(path),
                previous.get("iteration", 0),
                args.every,
                status != "running",
            )
            or status != previous.get("status")
        ):
            try:
                meta = refresh(args, path, status)
                if not args.render_only:
                    publish(args, meta)
                    temporary = state_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(meta) + "\n")
                    temporary.replace(state_path)
                print(json.dumps(meta), flush=True)
            except PublisherStopped:
                raise
            except Exception:
                traceback.print_exc()
                if args.once:
                    raise
                # Retry transient rendering, GitHub and network errors without marking success.
                time.sleep(30)
                continue
        if args.once or status != "running":
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
