"""Health report for a training run, written for humans and LLM agents.

    uv run starship-diagnose scratch_v8              # latest run of an experiment under logs/
    uv run starship-diagnose logs/scratch_v8/<time>  # a specific run directory
    uv run starship-diagnose scratch_v8 --probe      # also run the best policy on the current stage
    uv run starship-diagnose scratch_v8 --json       # machine-readable

Reads the run's TensorBoard events, curriculum.json and Hydra config. Every finding that needs
attention is a line starting with WARN, so `grep WARN` gives the to-do list.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Thresholds for WARN lines.
DORMANT_MAX = 0.20
DEAD_MAX = 0.05
RANK_MIN_FRACTION = 0.25
GATE_SATURATION_MAX = 0.30
VANISHING_RATIO = 0.10
RIGID_UPDATE_RATIO = 1e-5
COLLAPSE_DROP = 0.30
ENGINE_ENTROPY_MIN = 0.05


def find_run(target: str) -> Path:
    path = Path(target)
    if (path / ".hydra").exists():
        return path
    runs = sorted((Path("logs") / target).glob("*/.hydra"))
    if not runs:
        raise SystemExit(f"no run found for {target!r} (looked in {path} and logs/{target}/*)")
    return runs[-1].parent


def load_scalars(run: Path) -> dict[str, list[tuple[float, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    return {t: [(s.step, s.value) for s in ea.Scalars(t)] for t in ea.Tags()["scalars"]}


def last(scalars, tag, default=float("nan")):
    return scalars[tag][-1][1] if scalars.get(tag) else default


def trend(values: list[tuple[float, float]]) -> tuple[float, float, float]:
    v = [x for _, x in values]
    return v[0], v[len(v) // 2], v[-1]


def run_summary(run: Path, s: dict, report: dict) -> list[str]:
    lines = [f"## Run {run}"]
    it, steps = last(s, "train/iteration"), last(s, "train/env_steps")
    stage = last(s, "curriculum/stage", 0)
    lines.append(f"- iteration {it:.0f}, {steps / 1e6:.0f}M env steps, stage {stage:.0f}")
    events = run / "curriculum.json"
    promotions = json.loads(events.read_text()) if events.exists() else []
    for e in promotions:
        lines.append(
            f"- promoted to stage {e['stage']} ({e['name']}) at iteration {e['iteration']}"
        )
    rollbacks = last(s, "train/rollbacks", 0)
    if rollbacks:
        lines.append(f"- rollbacks: {rollbacks:.0f}")
    evals = [v for _, v in s.get("eval/env/rate/landed", [])]
    if evals:
        best = max(evals)
        lines.append(
            f"- eval landed: last {evals[-1]:.2f}, best {best:.2f}, last 8 "
            + " ".join(f"{v:.2f}" for v in evals[-8:])
        )
        if best - evals[-1] > COLLAPSE_DROP:
            lines.append(f"- WARN eval collapsed: {evals[-1]:.2f} vs best {best:.2f}")
    target = [v for _, v in s.get("eval/target/rate/landed", [])]
    if target:
        lines.append(f"- target eval landed: last {target[-1]:.2f}, best {max(target):.2f}")
    entropy = last(s, "explore/entropy")
    lines.append(f"- policy entropy {entropy:.2f} (3 Bernoulli + 8 Gaussian heads; no fixed max)")
    for i in range(3):
        engine_entropy = last(s, f"explore/entropy_head/engine{i}")
        if engine_entropy < ENGINE_ENTROPY_MIN:
            lines.append(
                f"- WARN engine{i} entropy {engine_entropy:.2f} < {ENGINE_ENTROPY_MIN}: "
                "engine exploration has nearly stopped"
            )
    report["summary"] = {
        "iteration": it,
        "env_steps": steps,
        "stage": last(s, "curriculum/stage", 0),
        "promotions": promotions,
        "rollbacks": rollbacks,
        "eval_landed": evals[-8:],
        "target_landed": target[-8:],
        "entropy": entropy,
    }
    return lines


def plasticity_section(s: dict, report: dict) -> list[str]:
    tags = sorted(t for t in s if t.startswith("plasticity/"))
    if not tags:
        return [
            "## Plasticity",
            "- no plasticity/* metrics (runner.algorithm.diagnostics_interval=0?)",
        ]
    lines = ["## Plasticity (first / middle / last)"]
    findings = {}
    for tag in tags:
        first, mid, end = trend(s[tag])
        findings[tag] = {"first": first, "middle": mid, "last": end}
        name = tag.removeprefix("plasticity/")
        warn = None
        if name.endswith("/dormant") and end > DORMANT_MAX:
            warn = f"{end:.0%} dormant units"
        elif name.endswith("/dead") and end > DEAD_MAX:
            warn = f"{end:.0%} dead units"
        elif name.endswith("_saturated") and end > GATE_SATURATION_MAX:
            warn = f"{end:.0%} saturated gates"
        elif "/grad_norm/" in name and first > 0 and end / first < VANISHING_RATIO:
            warn = f"gradient norm fell to {end / first:.1%} of its start (vanishing)"
        elif "/update_ratio/" in name and end < RIGID_UPDATE_RATIO:
            warn = f"update ratio {end:.1e}: layer has stopped changing"
        elif name.endswith("features/effective_rank"):
            width = last(s, tag.replace("effective_rank", "width"), 0)
            if width and end / width < RANK_MIN_FRACTION:
                warn = f"feature rank {end:.0f}/{width:.0f}: representation collapsed"
        if name.endswith(("/mean_abs", "/width")):
            continue
        prefix = f"- WARN {name}: {warn} | " if warn else f"- {name}: "
        lines.append(f"{prefix}{first:.3g} / {mid:.3g} / {end:.3g}")
    report["plasticity"] = findings
    return lines


def probe_section(run: Path, report: dict, episodes: int, device: str | None) -> list[str]:
    """Run the best policy deterministically on the run's current curriculum stage."""
    import torch
    from omegaconf import OmegaConf

    from starship.train.actions import ENGINE_HEADS
    from starship.train.conf import load_run_config
    from starship.train.curriculum import Curriculum
    from starship.train.diagnostics import model_diagnostics
    from starship.train.env import RocketEnv
    from starship.train.metrics import EpisodeTracker
    from starship.train.rsl_env import RslRocketEnv

    cfg = load_run_config(run)
    sim, reward = OmegaConf.to_object(cfg.sim), OmegaConf.to_object(cfg.reward)
    stage_name = "no curriculum"
    if cfg.curriculum.enabled:
        curriculum = Curriculum(cfg, cfg.curriculum)
        events = run / "curriculum.json"
        if events.exists():
            history = json.loads(events.read_text())
            if history:
                curriculum.stage = history[-1]["stage"]
        sim, reward = curriculum.configs()
        stage_name = f"stage {curriculum.stage} ({curriculum.name})"
    policy_path = run / "policy_best.pt"
    if not policy_path.exists():
        policy_path = run / "policy.pt"
    obs_cfg = OmegaConf.to_object(cfg.obs)
    env = RslRocketEnv(RocketEnv(episodes, sim, obs_cfg, reward, device=device, seed=123))
    policy = torch.load(policy_path, map_location=env.device, weights_only=False).eval()
    policy.reset()
    tracker = EpisodeTracker(env)
    obs = env.get_observations()
    pending = torch.ones(episodes, dtype=torch.bool, device=env.device)
    undecided, samples = [], []
    with torch.no_grad():
        for step in range(env.max_episode_length):
            # Sample inputs across time: identical spawns would otherwise give a batch of
            # identical observations, which reads as dead units.
            if step % 10 == 0 and policy.rnn.hidden_state is not None:
                h = tuple(x.clone() for x in policy.rnn.hidden_state)
                samples.append((obs["policy"].clone(), h))
            # One forward pass updates the distribution; the deterministic action is engines on
            # where p > 0.5 and the Gaussian means for the continuous heads.
            policy(obs, stochastic_output=True)
            mean = policy.distribution.mean
            engine_p = mean[:, :ENGINE_HEADS]
            undecided.append(((engine_p > 0.1) & (engine_p < 0.9))[pending].float().mean().item())
            actions = torch.cat([(engine_p > 0.5).float(), mean[:, ENGINE_HEADS:]], dim=-1)
            obs, _, dones, _ = env.step(actions)
            tracker.step(dones, record=pending)
            policy.reset(dones)
            pending &= ~dones.bool()
            if not pending.any():
                break
    live = {}
    if samples:
        from tensordict import TensorDict

        batch = torch.cat([o for o, _ in samples])[:8192]
        hidden = tuple(torch.cat([h[i] for _, h in samples], dim=1)[:, :8192] for i in range(2))
        live = model_diagnostics(
            policy, TensorDict({"policy": batch}, batch_size=[len(batch)]), hidden
        )
    stats = {k.removeprefix("probe/"): v for k, v in tracker.summarize("probe").items()}
    lines = [f"## Probe: {policy_path.name} on {stage_name}, {episodes} deterministic episodes"]
    for name in ("landed", "crashed", "tumbled", "out_of_bounds", "timeout"):
        lines.append(f"- {name} {stats.get(f'rate/{name}', 0.0):.3f}")
    if undecided:
        lines.append(f"- undecided engine choices (0.1 < p < 0.9): {np.mean(undecided):.0%}")
    for key, value in sorted(stats.items()):
        if key.startswith("crashed/breaks_"):
            lines.append(f"- crashes breaking {key.removeprefix('crashed/breaks_')}: {value:.0%}")
    for key in ("vz", "vh", "tilt_deg", "rate_dps", "miss"):
        if f"crashed/{key}" in stats:
            lines.append(f"- crash mean {key}: {stats[f'crashed/{key}']:.1f}")
    for k, v in live.items():
        if k.endswith(("/dormant", "/dead")) and v > (
            DORMANT_MAX if k.endswith("dormant") else DEAD_MAX
        ):
            lines.append(f"- WARN live {k} = {v:.0%}")
    lines.append(
        "- live: "
        + ", ".join(f"{k}={v:.3g}" for k, v in live.items() if not k.endswith("mean_abs"))
    )
    report["probe"] = {
        "policy": str(policy_path),
        "stage": stage_name,
        "episodes": episodes,
        "stats": stats,
        "live_diagnostics": live,
    }
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", help="experiment name under logs/ or a run directory")
    parser.add_argument(
        "--probe", action="store_true", help="run the best policy on the current stage"
    )
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()

    run = find_run(args.run)
    scalars = load_scalars(run)
    report: dict = {"run": str(run)}
    lines = run_summary(run, scalars, report)
    lines += plasticity_section(scalars, report)
    if args.probe:
        lines += probe_section(run, report, args.episodes, args.device)
    report["warnings"] = [re.sub(r"^- ", "", line) for line in lines if line.startswith("- WARN")]
    if args.json:
        print(json.dumps(report, indent=2, default=float))
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
