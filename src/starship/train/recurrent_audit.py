"""Replay a learning audit to separate initial and late-flight LSTM saturation.

uv run python -m starship.train.recurrent_audit RUN/learning-audit/model_2999-stochastic
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from starship.train.actions import ENGINE_HEADS, LEGACY_ACTION_DIM
from starship.train.catch_pretrain import make_runner
from starship.train.conf import load_run_config
from starship.train.diagnostics import lstm_gate_stats


@torch.inference_mode()
def phase_metrics(model, obs, valid, recorded, *, kind, window_steps=20, batch=64):
    """Replay complete first episodes from zero state; exclude autoreset/padding data.

    No recurrent state or normalizer statistics are updated. Recorded actor means or
    critic values must match before the reconstructed hidden states are trusted.
    """
    if kind not in ("actor", "critic") or min(window_steps, batch) < 1:
        raise ValueError("kind must be actor/critic and window/batch must be positive")
    if obs.ndim != 3 or valid.shape != obs.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("expected observations [time, episode, channel] and boolean valid mask")
    if not valid.numel() or not valid[0].all() or (valid[1:] & ~valid[:-1]).any():
        raise ValueError("valid must describe nonempty contiguous first episodes")
    width = (LEGACY_ACTION_DIM - ENGINE_HEADS,) if kind == "actor" else ()
    if recorded.shape != (*valid.shape, *width):
        raise ValueError("recorded outputs have the wrong shape")
    if not torch.isfinite(obs[valid]).all() or not torch.isfinite(recorded[valid]).all():
        raise ValueError("valid observations and recorded outputs must be finite")
    model.eval()
    lstm = model.rnn.rnn
    if lstm.num_layers != 1 or lstm.bidirectional or lstm.batch_first:
        raise ValueError("recurrent audit supports one-layer, forward, time-first LSTMs")
    totals, counts = {}, {}
    maximum_error = 0.0
    tolerance = 1e-4 if kind == "actor" else 0.005
    for start in range(0, obs.shape[1], batch):
        observations = obs[:, start : start + batch]
        mask = valid[:, start : start + batch]
        shape = observations.shape
        normalized = model.obs_normalizer(observations.reshape(-1, shape[-1])).reshape(shape)
        hidden, _ = lstm(normalized)
        predicted = model.mlp(hidden)
        predicted = (
            predicted[..., ENGINE_HEADS:LEGACY_ACTION_DIM]
            if kind == "actor"
            else predicted.squeeze(-1)
        )
        expected = recorded[:, start : start + batch]
        maximum_error = max(maximum_error, (predicted[mask] - expected[mask]).abs().max().item())
        if not torch.allclose(predicted[mask], expected[mask], rtol=1e-5, atol=tolerance):
            raise ValueError(
                f"{kind} replay differs from saved outputs (max error {maximum_error})"
            )
        previous = torch.cat([torch.zeros_like(hidden[:1]), hidden[:-1]])
        time = torch.arange(len(observations), device=obs.device)[:, None]
        lengths = mask.sum(0)[None, :]
        phases = {
            "spawn": mask & (time == 0),
            "initial_window": mask & (time < window_steps),
            "after_initial_window": mask & (time >= window_steps),
            "terminal_window": mask & (time >= lengths - window_steps),
            "all": mask,
        }
        for phase, selected in phases.items():
            count = int(selected.sum())
            if not count:
                continue
            stats = lstm_gate_stats(lstm, normalized[selected], previous[selected])
            counts[phase] = counts.get(phase, 0) + count
            for key, value in stats.items():
                totals[phase, key] = totals.get((phase, key), 0) + value * count
    return {
        "max_absolute_output_error": maximum_error,
        "replay_matches_saved_outputs": True,
        "phases": {
            phase: {
                "transitions": count,
                **{key: totals[phase, key] / count for p, key in totals if p == phase},
            }
            for phase, count in counts.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, help="optional relocated checkpoint with same SHA"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    if min(args.window_steps, args.batch) < 1:
        parser.error("window and batch must be positive")
    provenance = json.loads((args.audit / "report.json").read_text())
    checkpoint = args.checkpoint or Path(provenance["checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != provenance.get("checkpoint_sha256"):
        parser.error("checkpoint does not match the source audit SHA")
    output = args.output or args.audit / "recurrent-health"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cfg = load_run_config(checkpoint.parent)
    cfg.device, cfg.num_envs, cfg.eval_envs = "cpu", 1, 1
    runner = make_runner(cfg, output, provenance["seed"])
    try:
        runner.load(checkpoint)
        result = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": digest,
            "source_audit": str(args.audit.resolve()),
            "window_steps": args.window_steps,
            "note": "First-episode replay from zero hidden state. Terminal and initial windows "
            "may overlap. Cell saturation means the tanh cell-candidate gate, not stored cell "
            "state. Saturation alone does not establish a cause of control failure.",
            "models": {},
        }
        with np.load(args.audit / "trajectories.npz") as data:
            obs, valid = torch.from_numpy(data["obs"]), torch.from_numpy(data["valid"])
            for name, model, key in (
                ("actor", runner.alg.actor, "continuous_action_means"),
                ("critic", runner.alg.critic, "values"),
            ):
                result["models"][name] = phase_metrics(
                    model,
                    obs,
                    valid,
                    torch.from_numpy(data[key]),
                    kind=name,
                    window_steps=args.window_steps,
                    batch=args.batch,
                )
        (output / "report.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        runner.writer.close()


if __name__ == "__main__":
    main()
