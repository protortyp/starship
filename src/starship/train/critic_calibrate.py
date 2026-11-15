"""Fit only the critic to an audited checkpoint's complete on-policy returns.

The actor and normalizers stay fixed. Episode-disjoint validation selects a checkpoint;
validate again on fresh seeds before using it for PPO. Supervised updates are not PPO steps.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from starship.train.catch_pretrain import make_runner, sequence_output
from starship.train.conf import load_run_config
from starship.train.learning_audit import value_metrics


def critic_step(critic, optimizer, obs, targets, valid):
    values = sequence_output(critic, obs).squeeze(-1)
    loss = ((values - targets)[valid] / 100).square().mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def fit(runner, data, output, updates=2000, batch=16, seed=8512, learning_rate=3e-4):
    episodes = data["valid"].shape[1]
    if not all(torch.isfinite(data[k][data["valid"]]).all() for k in ("obs", "returns")):
        raise ValueError("calibration data must be finite")
    if episodes < 4:
        raise ValueError("at least four complete episodes are required")
    torch.manual_seed(seed)
    critic = runner.alg.critic
    critic.train()
    critic.obs_normalizer.eval()
    indices = torch.randperm(episodes, device=data["obs"].device)
    held, training = indices[: episodes // 4], indices[episodes // 4 :]
    actor_before = {k: v.detach().clone() for k, v in runner.alg.actor.state_dict().items()}

    def evaluate():
        with torch.no_grad():
            values = sequence_output(critic, data["obs"][:, held]).squeeze(-1)
            mask = data["valid"][:, held]
            targets = data["returns"][:, held]
            return {
                "all_states": value_metrics(
                    values[mask].cpu().numpy(), targets[mask].cpu().numpy()
                ),
                "initial": value_metrics(values[0].cpu().numpy(), targets[0].cpu().numpy()),
            }

    history = [{"update": 0, **evaluate()}]
    with torch.no_grad():
        values = sequence_output(critic, data["obs"][:, training]).squeeze(-1)
        mask = data["valid"][:, training]
        shift = (data["returns"][:, training] - values)[mask].mean()
        layer = [m for m in critic.mlp.modules() if isinstance(m, nn.Linear)][-1]
        layer.bias.add_(shift)
    optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    best = float("inf")
    report = {
        "seed": seed,
        "train_episode_indices": training.cpu().tolist(),
        "validation_episode_indices": held.cpu().tolist(),
        "bias_correction": shift.item(),
        "ppo_iteration_unchanged": runner.current_learning_iteration,
        "actor_unchanged": True,
        "history": history,
    }
    for update in range(1, updates + 1):
        selected = training[torch.randint(len(training), (batch,), device=training.device)]
        loss = critic_step(
            critic,
            optimizer,
            data["obs"][:, selected],
            data["returns"][:, selected],
            data["valid"][:, selected],
        )
        if update % 100 == 0 or update == updates:
            row = {"update": update, "training_loss": loss, **evaluate()}
            history.append(row)
            print(json.dumps(row), flush=True)
            if row["all_states"]["rmse"] < best:
                best = row["all_states"]["rmse"]
                if not all(
                    torch.equal(v, actor_before[k])
                    for k, v in runner.alg.actor.state_dict().items()
                ):
                    raise RuntimeError("Critic calibration changed the actor")
                # Adam moments for the old critic no longer match its calibrated weights.
                # Actor moments remain intact; the standalone fitting optimizer is not PPO.
                for parameter in critic.parameters():
                    runner.alg.optimizer.state.pop(parameter, None)
                runner.save(str(output / "model_calibrated.pt"))
                report["selected_update"] = update
            (output / "calibration.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=8512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if (
        min(args.updates, args.batch) < 1
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("counts and learning rate must be positive and finite")
    provenance = json.loads((args.audit / "report.json").read_text())
    if Path(provenance["checkpoint"]).resolve() != args.checkpoint.resolve():
        parser.error("audit must come from the exact checkpoint being calibrated")
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if provenance.get("checkpoint_sha256") != digest:
        parser.error("checkpoint bytes differ or audit has no digest; rerun its learning audit")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".hydra").mkdir()
    shutil.copy2(args.checkpoint.parent / ".hydra/config.yaml", args.output / ".hydra/config.yaml")
    (args.output / "source.json").write_text(json.dumps(provenance, indent=2))
    cfg = load_run_config(args.checkpoint.parent)
    cfg.device, cfg.num_envs, cfg.eval_envs = args.device, 8, 8
    torch.set_num_threads(2)
    runner = make_runner(cfg, args.output, args.seed)
    runner.load(args.checkpoint)
    if provenance["stage"] != runner.curriculum.stage:
        parser.error("counterfactual-stage audits cannot calibrate a different checkpoint stage")
    with np.load(args.audit / "trajectories.npz") as arrays:
        data = {k: torch.from_numpy(arrays[k]).to(args.device) for k in ["obs", "returns", "valid"]}
    fit(runner, data, args.output, args.updates, args.batch, args.seed, args.learning_rate)
    runner.writer.close()


if __name__ == "__main__":
    main()
