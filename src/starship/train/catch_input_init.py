"""Prepare an isolated recurrent ground checkpoint for previously unused catch inputs.

This changes only observation channels 28:32. It does not enable catch physics,
change the curriculum, unfreeze normalization, or adopt the prepared checkpoint.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starship.train.catch_pretrain import make_runner
from starship.train.conf import load_run_config
from starship.train.runner import export_policy

CHANNELS = slice(28, 32)
INPUT_WEIGHT = "rnn.rnn.weight_ih_l0"


def prepare_checkpoint(checkpoint, actor, critic, optimizer):
    """Return a copy; reject learned channels or mismatched parameter bindings.

    Models and optimizer must already be loaded from this checkpoint. Dormant
    channels must have exactly zero normalization moments and Adam moments. We
    preserve the complete optimizer state, including its shared step counters.
    """
    models = {"actor": actor, "critic": critic}
    targets = {}
    for role, model in models.items():
        saved = checkpoint[f"{role}_state_dict"]
        live = model.state_dict()
        if saved.keys() != live.keys() or any(
            not torch.equal(value.cpu(), live[key].cpu()) for key, value in saved.items()
        ):
            raise ValueError(f"{role} must match the source checkpoint exactly")
        parameters = dict(model.named_parameters())
        if INPUT_WEIGHT not in parameters or parameters[INPUT_WEIGHT].shape[-1] != 32:
            raise ValueError("initialization requires a 32-input recurrent model")
        for name in ("_mean", "_var", "_std"):
            stats = saved[f"obs_normalizer.{name}"]
            if stats.shape != (1, 32) or not torch.equal(
                stats[..., CHANNELS], torch.zeros_like(stats[..., CHANNELS])
            ):
                raise ValueError(f"{role} catch inputs are not unused zero-moment channels")
        targets[id(parameters[INPUT_WEIGHT])] = role
    saved_groups = checkpoint["optimizer_state_dict"]["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("optimizer groups must match the source checkpoint")
    found = set()
    for old, live in zip(saved_groups, optimizer.param_groups, strict=True):
        if len(old["params"]) != len(live["params"]):
            raise ValueError("optimizer parameter counts differ")
        for index, parameter in zip(old["params"], live["params"], strict=True):
            if id(parameter) not in targets:
                continue
            role = targets[id(parameter)]
            saved_state = checkpoint["optimizer_state_dict"]["state"].get(index, {})
            live_state = optimizer.state.get(parameter, {})
            if saved_state.keys() != live_state.keys():
                raise ValueError("optimizer parameter binding differs")
            for key, value in saved_state.items():
                current = live_state[key]
                if isinstance(value, torch.Tensor):
                    if not torch.equal(value.cpu(), current.cpu()):
                        raise ValueError("optimizer parameter binding differs")
                    if value.ndim and (
                        value.shape != parameter.shape or torch.count_nonzero(value[:, CHANNELS])
                    ):
                        raise ValueError(f"{role} catch inputs have learned optimizer moments")
                elif value != current:
                    raise ValueError("optimizer parameter binding differs")
            found.add(role)
    if found != set(models):
        raise ValueError("optimizer must contain both recurrent input weights")
    result = copy.deepcopy(checkpoint)
    for role in models:
        state = result[f"{role}_state_dict"]
        state[INPUT_WEIGHT][:, CHANNELS] = 0
        state["obs_normalizer._var"][..., CHANNELS] = 1
        state["obs_normalizer._std"][..., CHANNELS] = 1
    return result


def prepare(source: Path, output: Path):
    """Write a separate candidate checkpoint and export, leaving the source untouched."""
    if output.exists():
        raise FileExistsError("use a new output directory")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original = torch.load(source, map_location="cpu", weights_only=False)
    cfg = load_run_config(source.parent)
    cfg.device = "cpu"
    cfg.num_envs = cfg.eval_envs = 1
    output.mkdir(parents=True)
    runner = make_runner(cfg, output, 957000)
    try:
        runner.load(source)
        if runner.env.cfg.catch.enabled or not runner.alg.freeze_observation_normalization:
            raise ValueError("source must be a ground policy with frozen normalization")
        prepared = prepare_checkpoint(
            original, runner.alg.actor, runner.alg.critic, runner.alg.optimizer
        )
        checkpoint = output / "model_prepared.pt"
        torch.save(prepared, checkpoint)
        # Load through the normal runner and export path, not a custom inference model.
        runner.load(checkpoint)
        export_policy(runner.alg.actor, output / "policy_prepared.pt")
        (output / ".hydra").mkdir()
        saved_cfg = load_run_config(source.parent)
        saved_cfg.checkpoint = str(checkpoint.resolve())
        OmegaConf.save(saved_cfg, output / ".hydra/config.yaml")
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("source checkpoint changed during preparation")
        report = {
            "source": str(source.resolve()),
            "source_sha256": digest,
            "prepared_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "channels": [28, 29, 30, 31],
            "optimizer_preserved": True,
            "status": "Prepared only; validate complete ground and catch rollouts before adoption.",
        }
        (output / "preparation.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        runner.writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(prepare(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
