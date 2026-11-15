"""Audit a paired checkpoint's input scales and critic against complete on-policy returns.

uv run python -m starship.train.learning_audit RUN/model_2999.pt --episodes 256
"""

import argparse
import csv
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import warp as wp

from starship.sim.atmosphere import atmosphere
from starship.sim.catch import catch_center
from starship.sim.mass import mass_properties
from starship.sim.physics import Params, State
from starship.sim.rigid import gravity
from starship.train.actions import CONTINUOUS, ENGINE_HEADS, LEGACY_ACTION_DIM
from starship.train.catch_pretrain import make_runner
from starship.train.conf import load_run_config
from starship.train.diagnose import load_scalars
from starship.train.encoding import OBSERVATION_NAMES


@wp.kernel
def initial_physics(p: Params, state: State, out: wp.array2d(dtype=float)):
    i = wp.tid()
    body = state.body[i]
    mass = mass_properties(p.mass, state.tanks[i])
    out[i, 0] = catch_center(p.catcher, body, mass.com)[2] - p.catcher.height
    out[i, 1] = body.vel[2]
    out[i, 2] = mass.mass
    out[i, 3] = (
        p.propulsion.vacuum_thrust - atmosphere(body.pos[2]).pressure * p.propulsion.exit_area
    )
    out[i, 4] = gravity(body.pos[2])


def braking_estimates(physics):
    height, vz, mass, thrust, g = physics.T
    if not np.isfinite(physics).all() or np.any(mass <= 0):
        raise ValueError("braking estimates require finite physics and positive mass")
    result = {}
    for engines in (1, 2, 3):
        acceleration = engines * thrust / mass - g
        distance = np.maximum(np.maximum(-vz, 0) ** 2 - 2**2, 0) / (
            2 * np.maximum(acceleration, 1e-9)
        )
        margin = height - distance
        result[str(engines)] = {
            "deceleration_m_s2_median": float(np.median(acceleration)),
            "stopping_distance_m_median": float(np.median(distance)),
            "height_margin_m_median": float(np.median(margin)),
            "negative_margin_fraction": float(np.mean(margin < 0)),
        }
    return {
        "assumptions": "Initial-mass vertical full-thrust estimate to 2 m/s; excludes "
        "spool delay, tilt, drag, fuel burn and collisions; not a feasibility proof.",
        "engines": result,
    }


def value_metrics(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    residual = prediction - target
    variance = float(np.var(target))
    return {
        "samples": int(target.size),
        "prediction_mean": float(prediction.mean()),
        "return_mean": float(target.mean()),
        "return_std": float(target.std()),
        "bias": float(residual.mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "explained_variance": None if variance < 1e-12 else 1 - float(np.var(residual)) / variance,
    }


def channel_metrics(raw, normalized):
    rows = []
    for i in range(raw.shape[-1]):
        row = {"channel": i, "name": OBSERVATION_NAMES[i] if i < len(OBSERVATION_NAMES) else str(i)}
        for prefix, values in [("raw", raw[:, i]), ("normalized", normalized[:, i])]:
            finite = values[np.isfinite(values)]
            row[prefix + "_nonfinite"] = int(values.size - finite.size)
            if finite.size == 0:
                for name in ("min", "max", "mean", "std", "abs_p99", "abs_gt_10"):
                    row[prefix + "_" + name] = None
                continue
            for name, value in [
                ("min", finite.min()),
                ("max", finite.max()),
                ("mean", finite.mean()),
                ("std", finite.std()),
                ("abs_p99", np.quantile(np.abs(finite), 0.99)),
            ]:
                row[prefix + "_" + name] = float(value)
            row[prefix + "_abs_gt_10"] = float(np.mean(np.abs(finite) > 10))
        rows.append(row)
    return rows


def continuous_action_metrics(means, std):
    """Measure saturation and transformed +/-sigma span, not actuator response noise."""
    means = np.asarray(means, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    span = (np.tanh(means + std) - np.tanh(means - std)) / 2
    rows = []
    for i, (name, low, high) in enumerate(CONTINUOUS):
        # Half the transformed +/-sigma interval as a fraction of the full actuator range.
        half_width = span[:, i] / 2
        rows.append(
            {
                "name": name,
                "samples": len(means),
                "pre_tanh_mean_abs_p99": float(np.quantile(np.abs(means[:, i]), 0.99)),
                "pre_tanh_std": float(std[i]),
                "saturated_mean_fraction": float(np.mean(np.abs(np.tanh(means[:, i])) > 0.99)),
                "sigma_interval_half_width_range_fraction_median": float(np.median(half_width)),
                "sigma_interval_half_width_actuator_units_median": float(
                    np.median(half_width) * (high - low)
                ),
            }
        )
    return rows


def initial_tilt_metrics(initial_obs, status, values, returns):
    """Fixed physical-angle bins expose failures hidden by aggregate success."""
    tilt = np.arccos(np.clip(initial_obs[:, 0], -1, 1))
    edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.1, 1.6, np.pi])
    bins = []
    for i, (low, high) in enumerate(pairwise(edges)):
        selected = (tilt >= low) & ((tilt <= high) if i == len(edges) - 2 else (tilt < high))
        count = int(selected.sum())
        bins.append(
            {
                "lower_rad": float(low),
                "upper_rad": float(high),
                "episodes": count,
                "catch_rate": float(np.mean(status[selected] == 1)) if count else None,
                "critic_initial_states": value_metrics(values[selected], returns[selected])
                if count
                else None,
            }
        )
    return {
        "angle": "acos(up_body_1); radians; upper edge exclusive except final bin",
        "bins": bins,
    }


def discounted_returns(rewards, valid, gamma):
    result = np.where(valid, rewards, 0).copy()
    for t in range(len(result) - 2, -1, -1):
        result[t] += gamma * result[t + 1]
    return result


def audit(checkpoint, episodes, seed, output, stage=None, stochastic=False, action_seed=None):
    if action_seed is not None and not stochastic:
        raise ValueError("action_seed requires stochastic rollouts")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg = load_run_config(checkpoint.parent)
    cfg.device, cfg.num_envs, cfg.eval_envs = "cpu", episodes, episodes
    output.mkdir(parents=True, exist_ok=True)
    runner = make_runner(cfg, output, seed)
    runner.load(checkpoint)
    if stage is not None:
        if not 0 <= stage < len(runner.curriculum.stages):
            raise ValueError("stage is outside the checkpoint curriculum")
        runner.curriculum.stage = stage
        runner._apply_stage()
        runner._reset_rollout()
    actor, critic, env = runner.alg.actor.eval(), runner.alg.critic.eval(), runner.eval_env
    actor.reset()
    critic.reset()
    physics = wp.zeros((episodes, 5), dtype=float, device="cpu")
    wp.launch(
        initial_physics,
        dim=episodes,
        inputs=[env.env.sim.params, env.env.sim.state, physics],
        device="cpu",
    )
    braking = braking_estimates(physics.numpy())
    obs = env.get_observations()
    pending = torch.ones(episodes, dtype=torch.bool)
    status = torch.zeros(episodes, dtype=torch.int)
    raw, normalized, values, rewards, masks, engines = [], [], [], [], [], []
    critic_normalized = []
    action_means = []
    action_std = (
        actor.distribution.log_std.detach().clamp(*actor.distribution.log_std_range).exp().numpy()
    )
    terminal = {
        name: np.zeros((episodes, *getattr(env, "final_" + name).shape[1:]))
        for name in (
            "time",
            "pos",
            "vel",
            "omega",
            "catch_error",
            "peak_roll_rate",
            "peak_horizontal_rate",
            "flip_altitude",
            "ignition_altitude",
            "limit_causes",
            "first_limit_causes",
            "first_limit_time",
            "first_limit_altitude",
            "peak_transverse_rate",
        )
    }
    torch.manual_seed(seed if action_seed is None else action_seed)
    with torch.inference_mode():
        for _ in range(env.max_episode_length + 2):
            masks.append(pending.numpy().copy())
            raw.append(obs["policy"].numpy().copy())
            normalized.append(actor.obs_normalizer(obs["policy"]).numpy().copy())
            critic_normalized.append(critic.obs_normalizer(obs["policy"]).numpy().copy())
            values.append(runner.alg._values(critic(obs)).squeeze(-1).numpy().copy())
            action = actor(obs, stochastic_output=stochastic)
            mean = actor.distribution.mean if stochastic else action
            action_means.append(mean[:, ENGINE_HEADS:LEGACY_ACTION_DIM].numpy().copy())
            engines.append(action[:, :3].numpy().copy())
            obs, reward, done, _ = env.step(action)
            rewards.append(reward.numpy().copy())
            ended = done.bool() & pending
            status[ended] = env.final_status[ended]
            for name, data in terminal.items():
                data[ended.numpy()] = getattr(env, "final_" + name)[ended].numpy()
            pending &= ~done.bool()
            actor.reset(done)
            critic.reset(done)
            if not pending.any():
                break
    if pending.any():
        raise RuntimeError("Audit requires complete episodes; refusing truncated return targets")
    masks = np.stack(masks)
    raw, normalized, values = np.stack(raw), np.stack(normalized), np.stack(values)
    targets = discounted_returns(np.stack(rewards), masks, cfg.runner.algorithm.gamma)
    rows = channel_metrics(raw[masks], normalized[masks])
    critic_normalized = np.stack(critic_normalized)
    critic_rows = channel_metrics(raw[masks], critic_normalized[masks])
    action_means = np.stack(action_means)
    with (output / "critic_observations.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(critic_rows[0]))
        writer.writeheader()
        writer.writerows(critic_rows)
    with (output / "observations.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scalar_tags = [
        "value/explained_variance",
        "value/mean",
        "value/return_mean",
        "value/advantage_std",
        "loss/critic",
        "optim/updates",
        "policy/kl_analytic",
    ]
    cutoff = runner.total_env_steps
    scalars = {
        tag: [(step, value) for step, value in entries if step <= cutoff]
        for tag, entries in load_scalars(checkpoint.parent).items()
    }
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "seed": seed,
        "action_seed": seed if action_seed is None else action_seed,
        "episodes": episodes,
        "stage": runner.curriculum.stage,
        "stage_name": runner.curriculum.name,
        "mode": ("stochastic" if stochastic else "deterministic")
        + " policy; complete discounted Monte Carlo targets; no bootstrap",
        "stochastic": stochastic,
        "catch_rate": float((status == 1).float().mean()),
        "historical_limits": {
            "termination_enabled": bool(env.cfg.terminate_on_attitude_limits),
            "violated_episodes": int(np.count_nonzero(terminal["limit_causes"])),
            "caught_after_violation": int(
                np.sum((status.numpy() == 1) & (terminal["limit_causes"] != 0))
            ),
            "caught_without_violation": int(
                np.sum((status.numpy() == 1) & (terminal["limit_causes"] == 0))
            ),
        },
        "critic_all_states": value_metrics(values[masks], targets[masks]),
        "critic_initial_states": value_metrics(values[0], targets[0]),
        "initial_tilt": initial_tilt_metrics(raw[0], status.numpy(), values[0], targets[0]),
        "continuous_action_note": "Policy-frame target intervals from tanh(mean +/- std); "
        "not post-tanh standard deviations or physical actuator response. "
        "Saturation means abs(tanh(mean)) > 0.99.",
        "continuous_actions": continuous_action_metrics(action_means[masks], action_std),
        "initial_continuous_actions": continuous_action_metrics(action_means[0], action_std),
        "initial_braking_estimates": braking,
        "training_metrics_max_env_steps": cutoff,
        "engine_on_fraction": np.stack(engines)[masks].mean(0).tolist(),
        "observations": rows,
        "critic_observations": critic_rows,
        "terminal_status_counts": {str(int(k)): int((status == k).sum()) for k in status.unique()},
        "terminal_summary": {
            name: {
                "median": np.median(data, axis=0).tolist(),
                "min": np.min(data, axis=0).tolist(),
                "max": np.max(data, axis=0).tolist(),
            }
            for name, data in terminal.items()
        },
        "training_metrics": {
            tag: {
                "last": scalars[tag][-1][1],
                "last_200_median": float(np.median([v for _, v in scalars[tag][-200:]])),
            }
            for tag in scalar_tags
            if scalars.get(tag)
        },
    }
    np.savez_compressed(
        output / "trajectories.npz",
        obs=raw,
        normalized=normalized,
        critic_normalized=critic_normalized,
        continuous_action_means=action_means,
        continuous_action_std=action_std,
        **{"terminal_" + name: data for name, data in terminal.items()},
        values=values,
        returns=targets,
        valid=masks,
        status=status.numpy(),
    )
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Checkpoint changed during audit; use a frozen numbered checkpoint")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    runner.writer.close()
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("observations", "critic_observations")},
            indent=2,
        )
    )
    for row in rows:
        if (
            (row["normalized_abs_p99"] or 0) > 10
            or row["raw_nonfinite"]
            or row["normalized_nonfinite"]
        ):
            print(
                f"WARN observation {row['channel']}: normalized |p99|={row['normalized_abs_p99']}"
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--action-seed", type=int, help="vary action noise while retaining spawn seed"
    )
    parser.add_argument("--stage", type=int, help="explicit counterfactual curriculum stage")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.action_seed is not None and not args.stochastic:
        parser.error("--action-seed requires --stochastic")
    torch.set_num_threads(2)
    audit(
        args.checkpoint,
        args.episodes,
        args.seed,
        args.output
        or args.checkpoint.parent
        / "learning-audit"
        / (
            args.checkpoint.stem
            + (f"-stage{args.stage}" if args.stage is not None else "")
            + ("-stochastic" if args.stochastic else "")
            + (f"-actions{args.action_seed}" if args.action_seed is not None else "")
        ),
        stage=args.stage,
        stochastic=args.stochastic,
        action_seed=args.action_seed,
    )


if __name__ == "__main__":
    main()
