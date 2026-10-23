"""Collect and validate calm-flight world-model dynamics before policy optimization.

uv run starship-world-model --policy logs/.../policy_best.pt --device cuda:0
No policy weights are changed. Each invocation owns its output directory.
"""

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import torch
import warp as wp
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

import starship.train.conf  # noqa: F401
from starship.sim import Simulator
from starship.train.actions import (
    ACTION_DIM,
    ENGINE_HEADS,
    HEAD_NAMES,
    make_action_ranges,
    map_actions_kernel,
)
from starship.train.checkpoint import load_policy
from starship.train.curriculum import Curriculum
from starship.train.env import RocketEnv
from starship.train.model_state import (
    PHYSICAL_DIM,
    STATE_DIM,
    STATE_NAMES,
    capture_state,
    restore_state,
)
from starship.train.rsl_env import RslRocketEnv
from starship.train.world_model import Replay, WorldModel, fit_step, physical_errors


def collect(policy, env: RslRocketEnv, steps: int, noise: float) -> dict[str, torch.Tensor]:
    raw = env.env
    data = {key: [] for key in ("state", "next_state", "action", "done", "reward", "mirror")}
    policy.reset()
    obs = env.get_observations()
    with torch.no_grad():
        for _ in range(steps):
            # Model data are in physical coordinates: turn off reflection when constructing env.
            action = policy(obs)
            action[:, ENGINE_HEADS:] += noise * torch.randn_like(action[:, ENGINE_HEADS:])
            obs, reward, done, _ = env.step(action)
            data["state"].append(wp.to_torch(raw.model_before).clone())
            data["next_state"].append(wp.to_torch(raw.model_after).clone())
            data["action"].append(action.clone())
            data["done"].append(done.clone())
            data["reward"].append(reward.clone())
            data["mirror"].append(env.final_mirrored.clone())
            policy.reset(done)
    return {key: torch.stack(value) for key, value in data.items()}


def simulator_rollout(
    sim: Simulator, state: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay physical actions from captured state without auto-reset or policy feedback."""
    with (
        wp.ScopedStream(wp.stream_from_torch(str(state.device)))
        if state.is_cuda
        else wp.ScopedDevice("cpu")
    ):
        wp.launch(
            restore_state,
            dim=sim.num_envs,
            inputs=[wp.from_torch(state.contiguous()), sim.state],
            device=sim.device,
        )
        mirror = wp.zeros(sim.num_envs, dtype=int, device=sim.device)
        ranges = make_action_ranges(sim.device)
        squashed = wp.zeros(
            (sim.num_envs, ACTION_DIM - ENGINE_HEADS), dtype=float, device=sim.device
        )
        out = wp.zeros((sim.num_envs, STATE_DIM), dtype=float, device=sim.device)
        for action in actions:
            wp.launch(
                map_actions_kernel,
                dim=sim.num_envs,
                inputs=[wp.from_torch(action.contiguous()), mirror, ranges, sim.commands, squashed],
                device=sim.device,
            )
            sim.step()
        wp.launch(capture_state, dim=sim.num_envs, inputs=[sim.state, out], device=sim.device)
        result = wp.to_torch(out).clone()
        alive = wp.to_torch(sim.state.status).clone() == 0
    return result[:, :PHYSICAL_DIM], alive


def sensitivity_metrics(actual, predicted, alive) -> dict:
    """Keep flight dynamics separate from the easier actuator response."""
    result = {}
    for name, channels in (
        ("translation", slice(0, 6)),
        ("rotation", slice(6, 13)),
        ("actuators", slice(13, 39)),
        ("tanks", slice(39, 43)),
    ):
        a, p = actual[:, channels], predicted[:, channels]
        null = alive & (a.norm(dim=-1) <= 1e-7)
        null_metrics = {
            "null_samples": int(null.sum()),
            "null_predicted_norm_mean": p[null].norm(dim=-1).mean().item() if null.any() else None,
            "absolute_rmse": (p[alive] - a[alive]).square().mean().sqrt().item()
            if alive.any()
            else None,
        }
        keep = alive & (a.norm(dim=-1) > 1e-7)
        if keep.any():
            a, p = a[keep], p[keep]
            cosine = torch.nn.functional.cosine_similarity(a, p, dim=-1)
            result[name] = {
                **null_metrics,
                "samples": int(keep.sum()),
                "cosine_mean": cosine.mean().item(),
                "positive_alignment_fraction": (cosine > 0).float().mean().item(),
                "relative_l2": ((p - a).norm() / a.norm().clamp_min(1e-8)).item(),
            }
        else:
            result[name] = {
                **null_metrics,
                "samples": 0,
                "status": "no nonterminal measurable sensitivity",
            }
    return result


def gradient_report(model, batch, config, eps: float) -> dict:
    """Compare analytic model JVPs with simulator central differences per control head.

    Perturb only the first action of a logged open-loop sequence. This measures
    dynamics sensitivities, not reward gradients or closed-loop policy gradients.
    """
    state = batch["state"][0].contiguous()
    actions = batch["action"].contiguous()
    horizon, count = actions.shape[:2]
    sim = Simulator(count, config, device=str(state.device), seed=73, auto_reset=False)
    result = {}
    for head in range(ENGINE_HEADS, ACTION_DIM):
        direction = torch.zeros_like(actions)
        direction[0, :, head] = 1.0
        _, derivative = torch.autograd.functional.jvp(
            lambda a: model.rollout(state[:, :PHYSICAL_DIM], a)[-1],
            actions,
            direction,
        )
        plus, alive_plus = simulator_rollout(sim, state, actions + eps * direction)
        minus, alive_minus = simulator_rollout(sim, state, actions - eps * direction)
        actual = (plus - minus) / (2 * eps) / model.scale
        predicted = derivative / model.scale
        result[HEAD_NAMES[head]] = sensitivity_metrics(actual, predicted, alive_plus & alive_minus)
    # Check that restored state and command mapping reproduce the recorded trajectory.
    exact, alive = simulator_rollout(sim, state, actions)
    target = batch["next_state"][-1, :, :PHYSICAL_DIM]
    return {
        "horizon": horizon,
        "epsilon": eps,
        "heads": result,
        "exact_replay_alive": int(alive.sum()),
        "exact_replay_errors": physical_errors(exact, target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fit-horizon", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-samples", type=int, default=128)
    args = parser.parse_args()
    if args.stage < 0 or args.stage > 8:
        parser.error("initial model supports calm stages 0..8 only")
    if (
        args.envs < 2
        or args.steps < 32
        or min(args.updates, args.batch, args.width, args.probe_samples) < 1
    ):
        parser.error("need >=2 envs, >=32 steps and positive training/probe sizes")
    if not 1 <= args.fit_horizon <= args.steps or args.noise < 0:
        parser.error("invalid fit horizon or exploration noise")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("train")
    curriculum = Curriculum(cfg, cfg.curriculum)
    sim_cfg, reward_cfg = curriculum.configs(args.stage)
    obs_cfg = OmegaConf.to_object(cfg.obs)
    obs_cfg.mirror = False
    env = RslRocketEnv(
        RocketEnv(
            args.envs,
            sim_cfg,
            obs_cfg,
            reward_cfg,
            device=args.device,
            seed=args.seed,
            record_model_state=True,
        )
    )
    policy = load_policy(args.policy, args.device)
    start = time.monotonic()
    data = collect(policy, env, args.steps, args.noise)
    collect_s = time.monotonic() - start
    torch.save(
        {
            "schema": 1,
            "state_names": STATE_NAMES,
            "data": {k: v.cpu() for k, v in data.items()},
            "sim": dataclasses.asdict(sim_cfg),
            "reward": dataclasses.asdict(reward_cfg),
            "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        args.out / "replay.pt",
    )
    replay = Replay(data)
    model = WorldModel(args.width).to(args.device)
    train_indices = replay.windows(args.fit_horizon)
    calibration = replay.sample(train_indices, min(16384, len(train_indices)), 1)
    model.calibrate(
        calibration["state"][0, :, :PHYSICAL_DIM], calibration["next_state"][0, :, :PHYSICAL_DIM]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    report = {
        "schema": 1,
        "stage": args.stage,
        "seed": args.seed,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "collect_s": collect_s,
        "transitions": args.envs * args.steps,
        "terminal_transitions": int(data["done"].sum()),
        "policy_updated": False,
        "scope": "calm-flight dynamics; reward/termination and policy gradients not validated",
    }
    fit_start = time.monotonic()
    losses = []
    for update in range(args.updates):
        loss = fit_step(
            model, optimizer, replay.sample(train_indices, args.batch, args.fit_horizon)
        )
        losses.append(loss)
        if update % 100 == 0:
            print(f"update {update}/{args.updates} loss={loss:.5g}", flush=True)
    report["fit_s"] = time.monotonic() - fit_start
    report["final_train_loss"] = sum(losses[-100:]) / len(losses[-100:])
    torch.save(
        {"model": model.state_dict(), "width": args.width, "optimizer": optimizer.state_dict()},
        args.out / "world_model.pt",
    )
    errors = {}
    with torch.no_grad():
        for horizon in (1, 8, 32):
            batch = replay.sample(replay.windows(horizon, validation=True), 1024, horizon)
            state = batch["state"][0, :, :PHYSICAL_DIM]
            target = batch["next_state"][-1, :, :PHYSICAL_DIM]
            predicted = model.rollout(state, batch["action"])[-1]
            errors[str(horizon)] = {
                "model": physical_errors(predicted, target),
                "identity": physical_errors(state, target),
            }
    report["held_out_errors"] = errors
    report["gradient_probes"] = []
    for horizon in (1, 8, 32):
        batch = replay.sample(replay.windows(horizon, validation=True), args.probe_samples, horizon)
        # Both perturbation sizes use the same states/actions, so their agreement is meaningful.
        for eps in (0.01, 0.003):
            report["gradient_probes"].append(gradient_report(model, batch, sim_cfg, eps))
    report["total_s"] = time.monotonic() - start
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
