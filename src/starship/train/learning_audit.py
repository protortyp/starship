"""Audit a paired checkpoint's input scales and critic against complete on-policy returns.

uv run python -m starship.train.learning_audit RUN/model_2999.pt --episodes 256
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from starship.sim.atmosphere import atmosphere
from starship.sim.catch import catch_center
from starship.sim.mass import mass_properties
from starship.sim.physics import Params, State
from starship.sim.rigid import gravity
from starship.train.catch_pretrain import make_runner
from starship.train.conf import load_run_config
from starship.train.diagnose import load_scalars


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
        row = {"channel": i}
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


def discounted_returns(rewards, valid, gamma):
    result = np.where(valid, rewards, 0).copy()
    for t in range(len(result) - 2, -1, -1):
        result[t] += gamma * result[t + 1]
    return result


def audit(checkpoint, episodes, seed, output, stage=None, stochastic=False):
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg = load_run_config(checkpoint.parent)
    training_batch = cfg.num_envs * cfg.runner.num_steps_per_env
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
        initial_physics, dim=episodes, inputs=[env.env.sim.params, env.env.sim.state, physics]
    )
    braking = braking_estimates(physics.numpy())
    obs = env.get_observations()
    pending = torch.ones(episodes, dtype=torch.bool)
    status = torch.zeros(episodes, dtype=torch.int)
    raw, normalized, values, rewards, masks, engines = [], [], [], [], [], []
    torch.manual_seed(seed)
    with torch.inference_mode():
        for _ in range(env.max_episode_length + 2):
            masks.append(pending.numpy().copy())
            raw.append(obs["policy"].numpy().copy())
            normalized.append(actor.obs_normalizer(obs["policy"]).numpy().copy())
            values.append(runner.alg._values(critic(obs)).squeeze(-1).numpy().copy())
            action = actor(obs, stochastic_output=stochastic)
            engines.append(action[:, :3].numpy().copy())
            obs, reward, done, _ = env.step(action)
            rewards.append(reward.numpy().copy())
            ended = done.bool() & pending
            status[ended] = env.final_status[ended]
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
    cutoff = (runner.current_learning_iteration + 1) * training_batch
    scalars = {
        tag: [(step, value) for step, value in entries if step <= cutoff]
        for tag, entries in load_scalars(checkpoint.parent).items()
    }
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "seed": seed,
        "episodes": episodes,
        "stage": runner.curriculum.stage,
        "stage_name": runner.curriculum.name,
        "mode": ("stochastic" if stochastic else "deterministic")
        + " policy; complete discounted Monte Carlo targets; no bootstrap",
        "stochastic": stochastic,
        "catch_rate": float((status == 1).float().mean()),
        "critic_all_states": value_metrics(values[masks], targets[masks]),
        "critic_initial_states": value_metrics(values[0], targets[0]),
        "initial_braking_estimates": braking,
        "training_metrics_max_env_steps": cutoff,
        "engine_on_fraction": np.stack(engines)[masks].mean(0).tolist(),
        "observations": rows,
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
        values=values,
        returns=targets,
        valid=masks,
        status=status.numpy(),
    )
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Checkpoint changed during audit; use a frozen numbered checkpoint")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    runner.writer.close()
    print(json.dumps({k: v for k, v in report.items() if k != "observations"}, indent=2))
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
    parser.add_argument("--stage", type=int, help="explicit counterfactual curriculum stage")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
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
        ),
        stage=args.stage,
        stochastic=args.stochastic,
    )


if __name__ == "__main__":
    main()
