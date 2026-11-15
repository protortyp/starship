"""Separate stochastic return variability from initial-state critic prediction errors."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starship.train.learning_audit import audit, value_metrics


def uncertainty_metrics(returns, predictions):
    returns = np.asarray(returns, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if returns.ndim != 2 or min(returns.shape) < 2 or predictions.shape != returns.shape[1:]:
        raise ValueError("need at least two repeats and two starts, with one prediction per start")
    if not np.isfinite(returns).all() or not np.isfinite(predictions).all():
        raise ValueError("returns and predictions must be finite")
    repeats, starts = returns.shape
    mean = returns.mean(axis=0)
    within = float(returns.var(axis=0, ddof=1).mean())
    # The variance of sample means includes within-start noise divided by repeat count.
    between = max(0.0, float(mean.var(ddof=1)) - within / repeats)
    total = between + within
    return {
        "repeats": repeats,
        "starts": starts,
        "within_start_return_variance": within,
        "estimated_between_start_return_variance": between,
        "estimated_single_rollout_ev_ceiling": between / total if total > 0 else None,
        "critic_vs_individual_returns": value_metrics(
            np.broadcast_to(predictions, returns.shape).ravel(), returns.ravel()
        ),
        "critic_vs_mean_return": value_metrics(predictions, mean),
        "note": "Finite-repeat variance-component estimate, not a proven ceiling. "
        "Between-start variance is clipped at zero. Mean-return targets still contain sampling "
        "noise; this measures the current stochastic policy at spawn, not control quality.",
    }


def measure(checkpoint, output, starts=128, repeats=8, seed=819283):
    if starts < 2 or repeats < 2:
        raise ValueError("starts and repeats must be at least two")
    initial_obs = initial_values = digest = None
    returns, catches, reports = [], [], []
    for repeat in range(repeats):
        folder = output / f"repeat-{repeat}"
        report = audit(
            checkpoint, starts, seed, folder, stochastic=True, action_seed=seed + repeat + 1
        )
        with np.load(folder / "trajectories.npz") as data:
            if initial_obs is None:
                initial_obs, initial_values = data["obs"][0].copy(), data["values"][0].copy()
                digest = report["checkpoint_sha256"]
            if (
                report["checkpoint_sha256"] != digest
                or not np.array_equal(initial_obs, data["obs"][0])
                or not np.array_equal(initial_values, data["values"][0])
            ):
                raise RuntimeError(
                    "repeats must preserve the checkpoint, starts and initial values"
                )
            returns.append(data["returns"][0].copy())
            catches.append(data["status"] == 1)
        reports.append(str(folder / "report.json"))
    catch_rates = np.mean(catches, axis=0)
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "spawn_seed": seed,
        "action_seeds": list(range(seed + 1, seed + repeats + 1)),
        "matched_initial_observations_and_values": True,
        "stage": report["stage"],
        "stage_name": report["stage_name"],
        "catch_rate": float(catch_rates.mean()),
        "fraction_starts_with_mixed_outcomes": float(
            np.mean((catch_rates > 0) & (catch_rates < 1))
        ),
        "returns": uncertainty_metrics(returns, initial_values),
        "repeat_reports": reports,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2))
    print("RETURN UNCERTAINTY", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--starts", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=819283)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.starts, args.repeats) < 2:
        parser.error("starts and repeats must be at least two")
    torch.set_num_threads(2)
    measure(
        args.checkpoint,
        args.output or args.checkpoint.parent / "return-uncertainty" / args.checkpoint.stem,
        args.starts,
        args.repeats,
        args.seed,
    )


if __name__ == "__main__":
    main()
