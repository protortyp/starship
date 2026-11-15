"""Verify exact recurrent-policy continuation before using a snapshot for causal probes.

uv run python -m starship.train.branch_probe POLICY --stage 18 --stochastic --output DIR
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starship.train.branch_snapshot import BranchSnapshot
from starship.train.checkpoint import load_policy
from starship.train.conf import load_run_config
from starship.train.curriculum import Curriculum
from starship.train.env import RocketEnv
from starship.train.rsl_env import RslRocketEnv


def continuation(env, actor, stochastic):
    pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    status = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    trace = {key: [] for key in ("obs", "actions", "rewards", "done", "valid")}
    with torch.inference_mode():
        for _ in range(env.max_episode_length + 1):
            obs = env.get_observations()
            action = actor(obs, stochastic_output=stochastic)
            trace["obs"].append(obs["policy"].cpu().numpy().copy())
            trace["actions"].append(action.cpu().numpy().copy())
            trace["valid"].append(pending.cpu().numpy().copy())
            _, reward, done, _ = env.step(action)
            trace["rewards"].append(reward.cpu().numpy().copy())
            trace["done"].append(done.cpu().numpy().copy())
            ended = done & pending
            status[ended] = env.final_status[ended]
            pending &= ~done
            actor.reset(done)
            if not pending.any():
                break
    if pending.any():
        raise RuntimeError("Continuation did not reach terminal outcomes")
    return {**{k: np.stack(v) for k, v in trace.items()}, "status": status.cpu().numpy().copy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--branch-step", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.branch_step < 0:
        parser.error("episodes must be positive and branch-step nonnegative")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    cfg = load_run_config(args.policy.parent)
    sim, reward = Curriculum(cfg, cfg.curriculum).configs(args.stage)
    env = RslRocketEnv(
        RocketEnv(args.episodes, sim, OmegaConf.to_object(cfg.obs), reward, args.device, args.seed)
    )
    actor = load_policy(args.policy, args.device, obs_dim=env.env.obs_dim).eval()
    actor.reset()
    with torch.inference_mode():
        for _ in range(args.branch_step):
            obs = env.get_observations()
            _, _, done, _ = env.step(actor(obs, stochastic_output=args.stochastic))
            actor.reset(done)
            if done.any():
                raise ValueError(
                    "An episode ended before the branch; choose an earlier branch-step"
                )
    provenance = {
        "policy": str(args.policy.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "stage": args.stage,
        "seed": args.seed,
        "branch_step": args.branch_step,
        "stochastic": args.stochastic,
    }
    snapshot = BranchSnapshot.capture(env, actor, provenance=provenance)
    snapshot.save(args.output / "branch.pt")
    baseline = continuation(env, actor, args.stochastic)
    snapshot.restore(env, actor)
    replay = continuation(env, actor, args.stochastic)
    checks = {key: bool(np.array_equal(baseline[key], replay[key])) for key in baseline}
    report = {**provenance, "episodes": args.episodes, "exact_replay": checks}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise RuntimeError(f"Unchanged continuation differs: {checks}")
    np.savez_compressed(args.output / "baseline.npz", **baseline)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
