"""Controlled offline world-model comparison, with fixed probes and resumable fits.

The historical checkpoint tests duration; fresh seeds test capacity at equal fit
budgets on one device. This is a dynamics comparison, not a PPO policy benchmark.
"""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starship.sim import SimConfig, Simulator
from starship.train.actions import ACTION_DIM, ENGINE_HEADS, HEAD_NAMES
from starship.train.checkpoint import load_policy
from starship.train.encoding import ObsConfig
from starship.train.env import RocketEnv
from starship.train.model_experiment import collect, sensitivity_metrics, simulator_rollout
from starship.train.model_state import PHYSICAL_DIM, STATE_NAMES
from starship.train.reward import RewardConfig
from starship.train.rsl_env import RslRocketEnv
from starship.train.world_model import Replay, WorldModel, fit_step, physical_errors


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_replay(path: Path, device: str) -> tuple[dict, Replay]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        saved = torch.load(stream, map_location="cpu", weights_only=False)
    if saved.get("schema") != 1 or tuple(saved["state_names"]) != STATE_NAMES:
        raise ValueError("unsupported replay schema or state layout")
    if saved["args"]["stage"] not in range(9) or saved["data"]["mirror"].any():
        raise ValueError("comparison supports unmirrored calm stages only")
    return saved, Replay({k: v.to(device) for k, v in saved["data"].items()})


def phase_masks(state: torch.Tensor, next_state: torch.Tensor) -> dict[str, torch.Tensor]:
    q = state[:, 6:10]
    tilt = torch.acos((2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1])).clamp(-1, 1))
    return {
        "belly": (state[:, 2] >= 600) & (tilt >= torch.pi / 3),
        "flip": (tilt > torch.pi / 9) & (tilt < 4 * torch.pi / 9),
        "spooling": (next_state[:, 13:16] - state[:, 13:16]).abs().amax(-1) > 1e-5,
        "low": state[:, 2] < 600,
    }


def take_windows(replay, indices, count, horizon, generator) -> tuple[dict, torch.Tensor]:
    # Independent CPU generator: evaluation never advances the training sampler.
    choice = torch.randint(len(indices), (count,), generator=generator)
    selected = indices[choice.to(indices.device)]
    times = selected[:, 0][None] + torch.arange(horizon, device=indices.device)[:, None]
    envs = selected[:, 1][None].expand_as(times)
    return {k: v[times, envs] for k, v in replay.data.items()}, selected


def make_cohort(replay, prediction_count=1024, per_phase=32, seed=123, held_out=True) -> dict:
    generator = torch.Generator().manual_seed(seed)
    cohort = {}
    for horizon in (1, 8, 32):
        if held_out:
            indices = replay.windows(horizon, validation=True)
        else:
            indices = (~replay.data["done"].unfold(0, horizon, 1).any(-1)).nonzero()
        batch, selected = take_windows(replay, indices, prediction_count, horizon, generator)
        states = replay.data["state"][indices[:, 0], indices[:, 1]]
        following = replay.data["next_state"][indices[:, 0], indices[:, 1]]
        phases = phase_masks(states, following)
        probes, probe_indices, labels, missing = [], [], [], []
        for phase, mask in phases.items():
            candidates = indices[mask]
            if not len(candidates):
                missing.append(phase)
                continue
            probe, starts = take_windows(replay, candidates, per_phase, horizon, generator)
            probes.append(probe)
            probe_indices.append(starts)
            labels.extend([phase] * per_phase)
        if not probes:
            raise ValueError("no phase probes available")
        cohort[str(horizon)] = {
            "prediction": batch,
            "prediction_indices": selected,
            "probe": {k: torch.cat([p[k] for p in probes], dim=1) for k in probes[0]},
            "probe_indices": torch.cat(probe_indices),
            "phases": labels,
            "missing_phases": missing,
        }
    return cohort


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def make_reference(cohort, config, epsilons=(0.01, 0.003)) -> dict:
    reference = {}
    for horizon, entry in cohort.items():
        batch = entry["probe"]
        state, actions = batch["state"][0].contiguous(), batch["action"].contiguous()
        sim = Simulator(len(state), config, device=str(state.device), seed=73, auto_reset=False)
        exact, alive = simulator_rollout(sim, state, actions)
        row = {
            "replay_errors": physical_errors(exact, batch["next_state"][-1, :, :PHYSICAL_DIM]),
            "replay_alive": int(alive.sum()),
            "epsilon": {},
        }
        for eps in epsilons:
            differences, masks = [], []
            for head in range(ENGINE_HEADS, ACTION_DIM):
                direction = torch.zeros_like(actions)
                direction[0, :, head] = eps
                plus, plus_alive = simulator_rollout(sim, state, actions + direction)
                minus, minus_alive = simulator_rollout(sim, state, actions - direction)
                differences.append((plus - minus) / (2 * eps))
                masks.append(plus_alive & minus_alive)
            row["epsilon"][str(eps)] = {
                "derivative": torch.stack(differences),
                "alive": torch.stack(masks),
            }
        reference[horizon] = row
    return reference


def evaluate(model, cohort, reference) -> dict:
    report = {}
    model.eval()
    for horizon, entry in cohort.items():
        prediction = entry["prediction"]
        state = prediction["state"][0, :, :PHYSICAL_DIM]
        target = prediction["next_state"][-1, :, :PHYSICAL_DIM]
        with torch.no_grad():
            predicted = model.rollout(state, prediction["action"])[-1]
            if not torch.isfinite(predicted).all():
                report[horizon] = {
                    "prediction": {"status": "nonfinite rollout"},
                    "normalized_mse": None,
                    "sensitivities": {"status": "not evaluated"},
                }
                continue
            row = {
                "prediction": physical_errors(predicted, target),
                "normalized_mse": ((predicted - target) / model.delta_scale).square().mean().item(),
            }
        probe = entry["probe"]
        actions = probe["action"]
        derivatives = []
        for head in range(ENGINE_HEADS, ACTION_DIM):
            direction = torch.zeros_like(actions)
            direction[0, :, head] = 1
            _, derivative = torch.autograd.functional.jvp(
                lambda a, start=probe["state"][0, :, :PHYSICAL_DIM]: model.rollout(start, a)[-1],
                actions,
                direction,
            )
            derivatives.append(derivative.detach() / model.scale)
        predicted = torch.stack(derivatives)
        row["sensitivities"] = {}
        labels = entry["phases"]
        for eps, truth in reference[horizon]["epsilon"].items():
            actual = truth["derivative"] / model.scale
            heads = {}
            for j, head in enumerate(HEAD_NAMES[ENGINE_HEADS:]):
                if not torch.isfinite(predicted[j]).all():
                    heads[head] = {"status": "nonfinite model derivatives"}
                    continue
                heads[head] = {
                    "all": sensitivity_metrics(actual[j], predicted[j], truth["alive"][j])
                }
                for phase in sorted(set(labels)):
                    mask = torch.tensor([p == phase for p in labels], device=actions.device)
                    heads[head][phase] = sensitivity_metrics(
                        actual[j][mask], predicted[j][mask], truth["alive"][j][mask]
                    )
            row["sensitivities"][eps] = heads
        row["missing_phases"] = entry["missing_phases"]
        report[horizon] = row
    model.train()
    return report


def checkpoint(path, model, optimizer, generator, updates, fit_s, metadata):
    temp = path.with_suffix(".tmp")
    torch.save(
        {
            "schema": 2,
            "width": model.width,
            "depth": model.depth,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler_rng": generator.get_state(),
            "updates": updates,
            "fit_s": fit_s,
            "metadata": metadata,
        },
        temp,
    )
    temp.replace(path)


def restore(path, model, optimizer, generator, metadata):
    saved = torch.load(path, map_location=model.scale.device, weights_only=False)
    if saved.get("schema") == 2 and saved["metadata"] != metadata:
        raise ValueError("checkpoint provenance differs from requested comparison")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    if "sampler_rng" in saved:
        generator.set_state(saved["sampler_rng"].cpu())
    return saved.get("updates", metadata.get("historical_updates", 5000)), saved.get("fit_s")


def run_case(
    name,
    model,
    replay,
    cohort,
    reference,
    out,
    metadata,
    seed,
    updates,
    batch,
    budget_s=None,
    resume=None,
):
    case = out / name
    case.mkdir(exist_ok=True)
    generator = torch.Generator(device=model.scale.device).manual_seed(10000 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    done, previous_fit_s = 0, 0.0
    if resume:
        done, previous_fit_s = restore(resume, model, optimizer, generator, metadata)
    indices = replay.windows(8)
    report_path = case / "metrics.json"
    report = {
        "case": name,
        "metadata": metadata,
        "seed": seed,
        "width": model.width,
        "depth": model.depth,
        "parameters": sum(p.numel() for p in model.parameters()),
        "resumed_from_updates": done,
        "previous_device_fit_s": previous_fit_s,
        "fit_s": 0.0,
        "rows": [],
        "complete": False,
    }
    fit_s = 0.0
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (
            report["metadata"] != metadata
            or report["width"] != model.width
            or report["depth"] != model.depth
        ):
            raise ValueError("existing case has different provenance or architecture")
        last = report["rows"][-1]
        done, fit_s = restore(
            case / f"model_{last['updates']}.pt", model, optimizer, generator, metadata
        )
        if report["complete"]:
            return report, model
    start = time.monotonic()
    loss_sum, loss_count = 0.0, 0
    time_marks = [t for t in (120, 300) if t > fit_s] if budget_s is not None else []

    def record():
        nonlocal loss_sum, loss_count
        tick = time.monotonic()
        metrics = evaluate(model, cohort, reference)
        row = {
            "updates": done,
            "fit_s": fit_s,
            "train_loss": loss_sum / loss_count if loss_count else None,
            "validation": metrics,
            "eval_s": time.monotonic() - tick,
        }
        report["rows"].append(row)
        report["fit_s"] = fit_s
        checkpoint(case / f"model_{done}.pt", model, optimizer, generator, done, fit_s, metadata)
        write_json(report_path, report)
        print(
            json.dumps(
                {
                    "case": name,
                    "updates": done,
                    "fit_s": round(fit_s, 2),
                    "loss": row["train_loss"],
                    "validation_8": metrics["8"]["prediction"],
                }
            ),
            flush=True,
        )
        loss_sum, loss_count = 0.0, 0

    if not report["rows"]:
        record()
    while done < updates and (budget_s is None or fit_s < budget_s):
        tick = time.monotonic()
        loss = fit_step(model, optimizer, replay.sample(indices, batch, 8, generator))
        fit_s += time.monotonic() - tick
        done += 1
        loss_sum += loss
        loss_count += 1
        reached_time = bool(time_marks and fit_s >= time_marks[0])
        if reached_time:
            time_marks.pop(0)
        if done in (10000, 25000, 50000) or reached_time:
            record()
    if report["rows"][-1]["updates"] != done:
        record()
    report.update(
        complete=True,
        wall_s=time.monotonic() - start,
        stop_reason="fit budget" if budget_s and fit_s >= budget_s else "update limit",
    )
    write_json(report_path, report)
    return report, model


def main():
    study_start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration-updates", type=int, default=50000)
    parser.add_argument("--fit-budget", type=float, default=300)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--large-width", type=int, default=2048)
    parser.add_argument("--large-depth", type=int, default=3)
    parser.add_argument("--prediction-count", type=int, default=1024)
    parser.add_argument("--per-phase", type=int, default=32)
    parser.add_argument("--duration-only", action="store_true")
    parser.add_argument("--continue-run", action="store_true")
    parser.add_argument("--test-policy", type=Path)
    args = parser.parse_args()
    if (
        args.duration_updates < 5000
        or min(args.fit_budget, args.prediction_count, args.per_phase) <= 0
    ):
        parser.error("invalid training budget or evaluation size")
    args.out.mkdir(parents=True, exist_ok=args.continue_run)
    saved, replay = load_replay(args.replay, args.device)
    metadata = {
        "replay_sha256": file_hash(args.replay),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "validation_seed": 123,
        "batch": 512,
        "learning_rate": 3e-4,
        "fit_horizon": 8,
        "precision": "float32; TF32 disabled",
        "historical_updates": saved["args"]["updates"],
        "evaluation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("world_model.py").read_bytes()
        ).hexdigest(),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = {
        **metadata,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__,
        "device": str(args.device),
        "gpu": torch.cuda.get_device_name() if str(args.device).startswith("cuda") else None,
    }
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if any(previous[k] != protocol[k] for k in (*metadata, "gpu", "torch")):
            raise ValueError("continuation requires unchanged inputs, software and hardware type")
        for key in (
            "duration_updates",
            "fit_budget",
            "seeds",
            "large_width",
            "large_depth",
            "prediction_count",
            "per_phase",
            "duration_only",
        ):
            if previous["args"][key] != protocol["args"][key]:
                raise ValueError(f"continuation cannot change protocol argument {key}")
    else:
        write_json(protocol_path, protocol)
    sim = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(SimConfig), saved["sim"]))
    cache_path = args.out / "validation.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location=args.device, weights_only=False)
        if cached["metadata"] != metadata:
            raise ValueError("validation cache provenance mismatch")
        cohort, reference = cached["cohort"], cached["reference"]
    else:
        cohort = make_cohort(replay, args.prediction_count, args.per_phase)
        reference = make_reference(cohort, sim)
        torch.save(
            {
                "cohort": to_device(cohort, "cpu"),
                "reference": to_device(reference, "cpu"),
                "metadata": metadata,
            },
            cache_path,
        )
    template = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    write_json(args.out / "timing.json", {"preparation_s": time.monotonic() - study_start})
    duration = WorldModel(template["width"], template.get("depth", 2)).to(args.device)
    duration_report, _ = run_case(
        "duration",
        duration,
        replay,
        cohort,
        reference,
        args.out,
        metadata,
        0,
        args.duration_updates,
        512,
        resume=args.checkpoint,
    )
    reports = [duration_report]
    write_json(args.out / "summary.json", reports)
    if not args.duration_only:
        for seed in (int(s) for s in args.seeds.split(",")):
            # Alternate order across seeds; no simultaneous fits on the device.
            specs = [("small", 256, 2), ("large", args.large_width, args.large_depth)]
            if seed % 2:
                specs.reverse()
            for label, width, depth in specs:
                torch.manual_seed(seed)
                model = WorldModel(width, depth).to(args.device)
                for key in ("scale", "center", "delta_scale"):
                    getattr(model, key).copy_(template["model"][key])
                report, _ = run_case(
                    f"{label}_seed{seed}",
                    model,
                    replay,
                    cohort,
                    reference,
                    args.out,
                    metadata,
                    seed,
                    args.duration_updates,
                    512,
                    budget_s=args.fit_budget,
                )
                reports.append(report)
                write_json(args.out / "summary.json", reports)
    if args.test_policy:
        # Generate the test trajectories only after all scheduled fits are complete.
        if file_hash(args.test_policy) != saved["policy_sha256"]:
            raise ValueError("test policy must match the data-collection policy")
        torch.manual_seed(717)
        reward = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(RewardConfig), saved["reward"])
        )
        env = RslRocketEnv(
            RocketEnv(
                64,
                sim,
                ObsConfig(mirror=False),
                reward,
                device=args.device,
                seed=717,
                record_model_state=True,
            )
        )
        test_data = collect(load_policy(args.test_policy, args.device), env, 2048, 0.1)
        test_cohort = make_cohort(
            Replay(test_data), args.prediction_count, args.per_phase, seed=718, held_out=False
        )
        test_reference = make_reference(test_cohort, sim)
        torch.save(
            {
                "cohort": to_device(test_cohort, "cpu"),
                "reference": to_device(test_reference, "cpu"),
                "collection_seed": 717,
                "metadata": metadata,
            },
            args.out / "test.pt",
        )
        test_reports = {}
        for report in reports:
            model = WorldModel(report["width"], report["depth"]).to(args.device)
            # Fixed endpoint comparison, without choosing checkpoints using test outcomes.
            step = report["rows"][-1]["updates"]
            state = torch.load(
                args.out / report["case"] / f"model_{step}.pt",
                map_location=args.device,
                weights_only=False,
            )
            model.load_state_dict(state["model"])
            test_reports[report["case"]] = evaluate(model, test_cohort, test_reference)
        write_json(args.out / "test_results.json", test_reports)
    timing_path = args.out / "timing.json"
    timing = json.loads(timing_path.read_text())
    timing["invocation_wall_s"] = time.monotonic() - study_start
    write_json(timing_path, timing)
    print("comparison complete", flush=True)


if __name__ == "__main__":
    main()
