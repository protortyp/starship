"""Find how many parallel envs a training config can use on this machine.

For each env count, runs a few real training iterations (same config as starship-train, given as
Hydra overrides) and measures throughput and peak device memory. Stops at out-of-memory or once
throughput stops improving, and writes a JSON report to logs/bench/.

    uv run starship-bench model=lstm runner.num_steps_per_env=32
    uv run starship-bench --envs 4096,16384,65536 --iters 3 model=mlp
"""

import argparse
import gc
import json
import platform
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import warp as wp
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from starship.train.runner import StarshipRunner
from starship.train.train import make_env

DEFAULT_ENVS = [2**k for k in range(10, 19)]  # 1024 ... 262144


def is_oom(error: Exception) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def device_memory_gb(device: torch.device | str) -> tuple[float, float]:
    """(used, total) for the whole device, which covers both Warp's and torch's allocations."""
    device = torch.device(device)
    if device.type != "cuda":
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1e9, total / 1e9


def measure(cfg, num_envs: int, warmup: int, iters: int) -> dict:
    env = make_env(cfg, num_envs, cfg.seed)
    runner_cfg = OmegaConf.to_container(cfg.runner, resolve=True)
    runner_cfg["save_interval"] = 10**9
    with tempfile.TemporaryDirectory() as tmp:
        runner = StarshipRunner(env, runner_cfg, tmp, str(env.device), verbose=False)
        runner.learn(warmup)
        samples = []
        peak_used = 0.0
        for _ in range(iters):
            runner.learn(1)
            samples.append(runner.last_metrics)
            peak_used = max(peak_used, device_memory_gb(env.device)[0])
        runner.writer.close()

    def mean(key):
        return sum(s[key] for s in samples) / len(samples)

    used, total = device_memory_gb(env.device)
    return {
        "num_envs": num_envs,
        "batch_size": num_envs * cfg.runner.num_steps_per_env,
        "collect_fps": mean("time/collect_fps"),
        "train_fps": mean("time/train_fps"),
        "collect_s": mean("time/collect_s"),
        "update_s": mean("time/update_s"),
        "iteration_s": mean("time/iteration_s"),
        "mem_used_gb": max(peak_used, used),
        "mem_total_gb": total,
    }


def release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        wp.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--envs", default=None, help="comma-separated env counts to try")
    parser.add_argument("--warmup", type=int, default=1, help="iterations before measuring")
    parser.add_argument("--iters", type=int, default=3, help="measured iterations per size")
    parser.add_argument(
        "--min-gain", type=float, default=0.05, help="stop when train fps improves less than this"
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides of the training config")
    args = parser.parse_args()

    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("train", overrides=args.overrides)
    sizes = [int(n) for n in args.envs.split(",")] if args.envs else DEFAULT_ENVS
    device = torch.device(wp.device_to_torch(wp.get_device(cfg.device)))
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    print(f"benchmarking on {device} ({gpu}), overrides: {' '.join(args.overrides) or 'none'}")
    print(
        f"{'envs':>8} {'batch':>9} {'collect fps':>12} {'train fps':>11} "
        f"{'collect s':>10} {'update s':>9} {'mem GB':>12}"
    )

    results, stop_reason = [], "all sizes measured"
    for n in sizes:
        try:
            r = measure(cfg, n, args.warmup, args.iters)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            if not is_oom(error):
                raise
            stop_reason = f"out of memory at {n} envs"
            release(device)
            break
        results.append(r)
        mem = f"{r['mem_used_gb']:.1f}/{r['mem_total_gb']:.1f}" if r["mem_total_gb"] else "-"
        print(
            f"{n:>8} {r['batch_size']:>9} {r['collect_fps']:>12,.0f} {r['train_fps']:>11,.0f} "
            f"{r['collect_s']:>10.2f} {r['update_s']:>9.2f} {mem:>12}",
            flush=True,
        )
        release(device)
        if len(results) >= 2 and r["train_fps"] < results[-2]["train_fps"] * (1 + args.min_gain):
            stop_reason = f"train fps gain below {args.min_gain:.0%} at {n} envs"
            break

    if not results:
        raise SystemExit(f"no size could be measured ({stop_reason})")
    best = max(results, key=lambda r: r["train_fps"])
    print(f"\nstopped: {stop_reason}")
    print(f"best: {best['num_envs']} envs at {best['train_fps']:,.0f} train fps")

    out = Path("logs/bench") / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "device": str(device),
        "gpu": gpu,
        "overrides": args.overrides,
        "stop_reason": stop_reason,
        "best_num_envs": best["num_envs"],
        "results": results,
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
