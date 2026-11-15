"""Replay a catch checkpoint with geometry, policy-input and reward traces.

uv run python -m starship.train.catch_probe logs/<experiment>/<run>/policy_1499.pt --rerun
"""

import argparse
import csv
import dataclasses
import json
from contextlib import ExitStack
from pathlib import Path

import torch
import warp as wp

from starship.render.rerun_viewer import run_configs
from starship.sim.catch import (
    CATCH_FAILURES,
    flap_axes,
    flap_center,
    flap_half_size,
    pin,
    pin_error,
    projected_radius,
    tower_step,
)
from starship.sim.mass import mass_properties
from starship.sim.physics import Params
from starship.sim.propulsion import FlapState
from starship.sim.rigid import Body
from starship.train.actions import ARM_HEAD, CONTINUOUS, HEAD_NAMES
from starship.train.checkpoint import load_policy
from starship.train.env import RocketEnv
from starship.train.reward import REWARD_TERMS
from starship.train.rsl_env import RslRocketEnv

FIELDS = (
    "pin_left_above_support_m",
    "pin_right_above_support_m",
    "pin_alignment_error_m",
    "aft_fin_clearance_m",
    "arm_open_deg",
    "closure_time_s",
    "pin_descent_eta_s",
    "automatic_close",
    "altitude_m",
    "vertical_speed_m_s",
    "tilt_deg",
)
STATUS = {0: "flying", 1: "caught", 2: "crashed", 3: "out_of_bounds", 4: "timeout", 5: "tumbled"}


@wp.kernel
def geometry(
    p: Params,
    bodies: wp.array(dtype=Body),
    tanks: wp.array(dtype=wp.vec4),
    flaps: wp.array(dtype=FlapState),
    arms: wp.array(dtype=float),
    out: wp.array2d(dtype=float),
):
    i = wp.tid()
    b = bodies[i]
    com = mass_properties(p.mass, tanks[i]).com
    c = p.catcher
    left = pin(c, b, com, -1.0)[2] - c.height
    right = pin(c, b, com, 1.0)[2] - c.height
    out[i, 0] = left
    out[i, 1] = right
    out[i, 2] = pin_error(c, b, com)
    top = -1.0e10
    for j in range(2):
        center = flap_center(p.aero, b, com, flaps[i].angle, j)
        extent = projected_radius(
            flap_axes(p.aero, b, flaps[i].angle, j),
            flap_half_size(c, p.aero, j),
            wp.vec3(0.0, 0.0, 1.0),
        )
        top = wp.max(top, center[2] + extent)
    out[i, 3] = c.height - 1.0 - top
    out[i, 4] = arms[i] * 180.0 / wp.pi
    out[i, 5] = arms[i] / c.angular_speed
    eta = -1.0
    for side in range(2):
        sign = float(2 * side - 1)
        lever = wp.vec3(c.station - com, sign * c.pin_radius, 0.0)
        vz = (b.vel + wp.quat_rotate(b.rot, wp.cross(b.omega, lever)))[2]
        height = pin(c, b, com, sign)[2] - c.height
        if vz < -1.0e-6 and height >= 0.0:
            arrival = height / -vz
            if eta < 0.0 or arrival < eta:
                eta = arrival
    out[i, 6] = eta
    # Evaluate from fully open so this still reports a close target when already closed.
    out[i, 7] = float(
        tower_step(c, p.aero, b, com, flaps[i].angle, c.open_angle, p.dt) < c.open_angle
    )
    out[i, 8] = b.pos[2]
    out[i, 9] = b.vel[2]
    out[i, 10] = (
        wp.acos(wp.clamp(wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))[2], -1.0, 1.0))
        * 180.0
        / wp.pi
    )


def measure(sim, terminal=False):
    state = sim.terminal if terminal else sim.state
    out = wp.zeros((sim.num_envs, len(FIELDS)), dtype=float, device=sim.device)
    wp.launch(
        geometry,
        dim=sim.num_envs,
        inputs=[sim.params, state.body, state.tanks, state.flaps, state.arm_angle, out],
        device=sim.device,
    )
    return dict(zip(FIELDS, map(float, out.numpy()[0]), strict=True))


def replay(policy_path, cfg, obs_cfg, reward_cfg, seed, mode, output, recording=False):
    env = RslRocketEnv(RocketEnv(1, cfg, obs_cfg, reward_cfg, device="cpu", seed=seed))
    policy = load_policy(policy_path, obs_dim=env.env.obs_dim)
    raw_outputs = []
    hook = policy.mlp.register_forward_hook(lambda _m, _i, value: raw_outputs.append(value))
    rows = []
    rec = None
    with ExitStack() as stack:
        if recording:
            import rerun as rr
            import rerun.blueprint as rrb

            from starship.render.rerun_viewer import flap_geometry, log_scene, log_step
            from starship.sim.mass import make_mass_params

            rec = rr.RecordingStream(f"catch-probe-{mode}-{seed}")
            rec.save(str(output / f"{mode}-seed-{seed}.rrd"))
            stack.enter_context(rec)
            rr.send_blueprint(
                rrb.Blueprint(
                    rrb.Tabs(
                        rrb.Horizontal(
                            rrb.Spatial3DView(
                                name="Flight",
                                origin="/world",
                                line_grid=False,
                                eye_controls=rrb.EyeControls3D(
                                    position=(150, -180, 150), look_target=(0, 0, 90)
                                ),
                            ),
                            rrb.Vertical(
                                rrb.TimeSeriesView(
                                    name="Close decision",
                                    origin="/decision",
                                    contents=[
                                        "/decision/close_probability",
                                        "/decision/executed_close",
                                        "/decision/automatic_close",
                                    ],
                                ),
                                rrb.TimeSeriesView(
                                    name="Pin height and fin clearance (m)",
                                    origin="/state",
                                    contents=[
                                        "/state/pin_left_above_support_m",
                                        "/state/pin_right_above_support_m",
                                        "/state/aft_fin_clearance_m",
                                    ],
                                ),
                                rrb.TimeSeriesView(
                                    name="Closure and arrival estimates (s)",
                                    origin="/state",
                                    contents=["/state/closure_time_s", "/state/pin_descent_eta_s"],
                                ),
                            ),
                            name="Catch timing",
                        ),
                        rrb.Vertical(
                            rrb.TimeSeriesView(name="Raw policy inputs", origin="/inputs/raw"),
                            rrb.TimeSeriesView(
                                name="Normalized policy inputs", origin="/inputs/normalized"
                            ),
                            name="Policy inputs",
                        ),
                        rrb.Vertical(
                            rrb.TimeSeriesView(name="Engine switches", origin="/commands/engines"),
                            rrb.TimeSeriesView(
                                name="Gimbal targets (rad)", origin="/commands/gimbal"
                            ),
                            rrb.TimeSeriesView(
                                name="Throttle and roll RCS", origin="/commands/power"
                            ),
                            rrb.TimeSeriesView(name="Flap targets (rad)", origin="/commands/flaps"),
                            name="Flight commands",
                        ),
                        rrb.TimeSeriesView(name="Reward components", origin="/reward"),
                    )
                )
            )
            log_scene(cfg, 1)
            trails = [[]]
            flaps = flap_geometry(cfg)
            mass = make_mass_params(cfg.vehicle)
        obs = env.get_observations()
        total = 0.0
        first_close = None
        with torch.inference_mode():
            for step in range(env.max_episode_length):
                row = measure(env.env.sim)
                row.update(time_s=float(env.env.sim.state.time.numpy()[0]), step=step)
                raw = obs["policy"].clone()
                normalized = policy.obs_normalizer(raw).clone()
                action = policy(obs, stochastic_output=False)
                logits = raw_outputs.pop()
                row["close_probability"] = torch.sigmoid(logits[0, ARM_HEAD]).item()
                row["policy_close"] = float(action[0, ARM_HEAD])
                for j, name in enumerate(HEAD_NAMES):
                    row[f"requested_{name}"] = float(action[0, j])
                if mode == "open":
                    action[0, ARM_HEAD] = 0
                elif mode == "automatic":
                    action[0, ARM_HEAD] = row["automatic_close"]
                row["executed_close"] = float(action[0, ARM_HEAD])
                for j, name in enumerate(HEAD_NAMES):
                    row[f"executed_{name}"] = float(action[0, j])
                if first_close is None and row["executed_close"]:
                    first_close = {
                        key: row[key] for key in ("time_s", *FIELDS, "close_probability")
                    }
                for j in range(raw.shape[-1]):
                    row[f"obs_{j:02d}"] = raw[0, j].item()
                    row[f"normalized_obs_{j:02d}"] = normalized[0, j].item()
                obs, reward, done, _ = env.step(action)
                for j, (name, _, _) in enumerate(CONTINUOUS):
                    row[f"policy_target_{name}"] = float(env.squashed_actions[0, j])
                row["reward"] = reward.item()
                row["terminal_reward"] = env.terminal_reward.item()
                for key, value in zip(REWARD_TERMS, env.reward_terms[0].tolist(), strict=True):
                    row[f"reward_{key}"] = value
                total += reward.item()
                terminal = bool(done.item())
                end = measure(env.env.sim, terminal=terminal)
                row.update({f"after_{key}": value for key, value in end.items()})
                state = env.env.sim.terminal if terminal else env.env.sim.state
                row["after_time_s"] = float(state.time.numpy()[0])
                row["status"] = STATUS[int(state.status.numpy()[0])]
                row["failure"] = CATCH_FAILURES[int(state.catch_failure.numpy()[0])]
                rows.append(row)
                if rec:
                    snap = env.env.sim.snapshot()
                    if terminal:
                        final = env.env.sim.terminal_snapshot()
                        snap = dataclasses.replace(
                            snap,
                            **{
                                field.name: getattr(final, field.name)
                                for field in dataclasses.fields(snap)
                                if hasattr(final, field.name)
                            },
                        )
                    rr.set_time("sim_time", duration=row["time_s"])
                    for key in (*FIELDS, "close_probability", "executed_close"):
                        rr.log(f"decision/{key}", rr.Scalars(row[key]))
                    for name in HEAD_NAMES[:3]:
                        rr.log(f"commands/engines/{name}", rr.Scalars(row[f"executed_{name}"]))
                    for name, _, _ in CONTINUOUS:
                        group = (
                            "gimbal"
                            if name.startswith("gimbal")
                            else ("flaps" if name.startswith("flap") else "power")
                        )
                        rr.log(f"commands/{group}/{name}", rr.Scalars(row[f"policy_target_{name}"]))
                    for j in range(raw.shape[-1]):
                        rr.log(f"inputs/raw/obs_{j:02d}", rr.Scalars(raw[0, j].item()))
                        rr.log(
                            f"inputs/normalized/obs_{j:02d}", rr.Scalars(normalized[0, j].item())
                        )
                    rr.set_time("sim_time", duration=row["after_time_s"])
                    for key in FIELDS:
                        rr.log(f"state/{key}", rr.Scalars(end[key]))
                    for key in (
                        "reward",
                        "terminal_reward",
                        *(f"reward_{k}" for k in REWARD_TERMS),
                    ):
                        rr.log(f"reward/{key}", rr.Scalars(row[key]))
                    rr.log("outcome", rr.TextLog(f"{row['status']}: {row['failure']}"))
                    log_step(cfg, snap, flaps, trails, mass)
                if terminal:
                    break
        hook.remove()
        if rec:
            rec.flush()
            rec.disconnect()
        path = output / f"{mode}-seed-{seed}.csv"
        with path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return {
            "mode": mode,
            "seed": seed,
            "status": rows[-1]["status"],
            "failure": rows[-1]["failure"],
            "duration_s": rows[-1]["after_time_s"],
            "return": total,
            "first_close": first_close,
            "preclear_close_steps": sum(
                r["executed_close"] > 0.5 and r["aft_fin_clearance_m"] < 0 for r in rows
            ),
            "steps": len(rows),
            "reward_totals": {key: sum(r[f"reward_{key}"] for r in rows) for key in REWARD_TERMS},
            "final_geometry": end,
        }


def report_text(report):
    lines = [
        "# Catch decision probe",
        "",
        f"Checkpoint: `{report['policy']}`; stage: **{report['stage_name']}**.",
        "Deterministic rollouts, matched seeds. Each intervention changes only the arm command; "
        "the policy responds to the resulting state, so flight trajectories can diverge. "
        "The automatic clearance/alignment rule is sampled once per policy control step.",
        "",
        "| Mode | Seed | Outcome | Failure | Duration (s) | Return | "
        "First close (s) | Fin clearance then (m) |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for result in report["episodes"]:
        close = result["first_close"]
        when = f"{close['time_s']:.2f}" if close else "never"
        clearance = f"{close['aft_fin_clearance_m']:.2f}" if close else "—"
        lines.append(
            f"| {result['mode']} | {result['seed']} | {result['status']} | {result['failure']} | "
            f"{result['duration_s']:.2f} | {result['return']:.2f} | {when} | {clearance} |"
        )
    lines += [
        "",
        "Positive fin clearance means both aft fins are below the arm underside. A negative "
        "value is not itself a collision: the arms can be open while the fins pass. ETA is a "
        "constant-current-pin-velocity estimate, not a prediction; -1 means no descending pin.",
        "",
        "CSV rows contain pre-action geometry, close probability, requested/executed commands, "
        "continuous actuator targets in the policy frame (after tanh, before mirroring), "
        "all raw and normalized policy observations, reward components, and post-step geometry. "
        "Terminal values are captured before autoreset. Collision labels identify the first "
        "conservative swept contact detected by the simulator, not measured impact physics.",
        "",
        "The policy observes attitude (0-2), body velocity (3-5), target relative to pin center "
        "(6-8), angular velocity (9-11), COM altitude (12), fins (13-16), "
        "engines (17-22), fuel (23), "
        "entry channels (24-27), tower heading (28-30), arm opening (31). It has no explicit fin "
        "clearance, individual pin heights or closure-time in the base 32 inputs. With "
        "obs.catch_geometry=true, channels 32-37 add two pin heights, aft-fin clearance, pin "
        "alignment error, closure time and constant-velocity arrival time. Collision labels are "
        "diagnostics only. Raw position is divided by "
        f"{report.get('position_scale', 'the configured scale')}; "
        "the learned normalizer is also exported "
        "to avoid mistaking small raw values for an invisible signal.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rerun", action="store_true", help="record seed 0 of each mode")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    policy = args.policy.resolve()
    paired = policy.with_name(policy.name.replace("policy_", "model_", 1))
    if paired == policy or not paired.exists():
        parser.error("use a policy_N.pt checkpoint with its matching model_N.pt")
    model = torch.load(paired, map_location="cpu", weights_only=False)
    stage = model["infos"]["curriculum"]["stage"]
    cfg, obs, reward, name = run_configs(policy.parent, stage)
    if not (cfg.catch.enabled and cfg.catch.moving and cfg.catch.policy_controlled):
        parser.error("probe requires a policy-controlled moving-arm checkpoint")
    output = args.output or policy.parent / "catch-probe" / policy.stem
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    report = {
        "policy": str(policy),
        "stage": stage,
        "stage_name": name,
        "position_scale": obs.position,
        "episodes": [],
    }
    for mode in ("policy", "open", "automatic"):
        for seed in range(args.seeds):
            report["episodes"].append(
                replay(policy, cfg, obs, reward, seed, mode, output, args.rerun and seed == 0)
            )
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(report_text(report))
    print(report_text(report))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
