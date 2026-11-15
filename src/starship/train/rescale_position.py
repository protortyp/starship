"""Change position encoding sensitivity while preserving initial actor/critic outputs.

uv run python -m starship.train.rescale_position RUN/model_N.pt --position 30 --output NEW_RUN
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starship.train.catch_pretrain import make_runner, sequence_output
from starship.train.conf import load_run_config


@torch.no_grad()
def rescale_model(model, multiplier):
    """For x'=a*x, absorb the normalizer epsilon change into the LSTM input columns.

    W' = W*(a*std+eps)/(a*(std+eps)); means and variances scale with the raw inputs.
    This preserves the function, but intentionally changes its optimization coordinates.
    """
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("input multiplier must be finite and positive")
    normalizer = model.obs_normalizer
    weight = model.rnn.rnn.weight_ih_l0
    old_denominator = normalizer._std[..., 6:9].clone() + normalizer.eps
    normalizer._mean[..., 6:9] *= multiplier
    normalizer._var[..., 6:9] *= multiplier**2
    normalizer._std[..., 6:9] *= multiplier
    ratio = (normalizer._std[..., 6:9] + normalizer.eps) / (multiplier * old_denominator)
    weight[:, 6:9] *= ratio
    return weight, ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--position", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.position) or args.position <= 0:
        parser.error("position scale must be finite and positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    cfg = load_run_config(args.checkpoint.parent)
    old_scale = cfg.obs.position
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".hydra").mkdir()
    original_device, original_envs, original_eval = cfg.device, cfg.num_envs, cfg.eval_envs
    cfg.device, cfg.num_envs, cfg.eval_envs = "cpu", 8, 8
    runner = make_runner(cfg, args.output, 0)
    runner.load(args.checkpoint)
    raw = runner.env.get_observations()["policy"].unsqueeze(0)
    scaled = raw.clone()
    multiplier = old_scale / args.position
    scaled[..., 6:9] *= multiplier
    report = {
        "source": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "old_position_scale": old_scale,
        "new_position_scale": args.position,
        "ppo_iteration": runner.current_learning_iteration,
        "optimizer_change": "reset moments only for transformed input weights",
    }
    with torch.no_grad():
        for name, model in [("actor", runner.alg.actor), ("critic", runner.alg.critic)]:
            model.eval()
            before = sequence_output(model, raw)
            weight, ratio = rescale_model(model, multiplier)
            runner.alg.optimizer.state.pop(weight, None)
            after = sequence_output(model, scaled)
            torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-5)
            report[name + "_max_output_error"] = (after - before).abs().max().item()
            report[name + "_column_multiplier"] = ratio.tolist()
    cfg.device, cfg.num_envs, cfg.eval_envs = original_device, original_envs, original_eval
    cfg.obs.position = args.position
    OmegaConf.save(cfg, args.output / ".hydra/config.yaml")
    runner.save_all(runner.current_learning_iteration)
    runner.writer.close()
    (args.output / "rescale.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
