"""Raw simulator throughput: control and physics steps per second, no policy.

uv run python -m starship.sim.bench --device cuda:0
"""

import argparse
import time

import numpy as np
import warp as wp

from starship.sim import SimConfig, Simulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--envs", default="4096,16384,65536,262144,1048576")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    cfg = SimConfig()
    for n in (int(x) for x in args.envs.split(",")):
        sim = Simulator(n, cfg, device=args.device, auto_reset=True)
        rng = np.random.default_rng(0)
        # Engines and flaps active so every code path is exercised.
        sim.set_commands(
            engine_on=rng.integers(0, 2, (n, 3)),
            throttle=rng.uniform(0.4, 1.0, (n, 3)),
            gimbal2=rng.uniform(-0.2, 0.2, (n, 3)),
            gimbal3=rng.uniform(-0.2, 0.2, (n, 3)),
            flaps=rng.uniform(0.0, 1.4, (n, 4)),
        )
        for _ in range(3):
            sim.step()
        wp.synchronize_device(sim.device)
        start = time.perf_counter()
        for _ in range(args.steps):
            sim.step()
        wp.synchronize_device(sim.device)
        rate = n * args.steps / (time.perf_counter() - start)
        print(
            f"envs {n:>9,}  control steps/s {rate / 1e6:8.2f}M  "
            f"physics steps/s {rate * cfg.substeps / 1e6:8.1f}M",
            flush=True,
        )


if __name__ == "__main__":
    main()
