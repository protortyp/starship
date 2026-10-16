"""Headless demo: run the base scenario across many parallel envs and print telemetry."""

import argparse
import math
import time

import numpy as np
import warp as wp

from starship.control import HeuristicController, IdleController, RandomController
from starship.sim import FLYING, LEFT, MAIN, RIGHT, STATUS_NAMES, Simulator, StateSnapshot

CONTROLLERS = {
    "heuristic": HeuristicController,
    "random": lambda _cfg: RandomController(),
    "idle": lambda _cfg: IdleController(),
}


def describe_action(action: int) -> str:
    return "".join(
        c if action & bit else "." for c, bit in (("L", LEFT), ("M", MAIN), ("R", RIGHT))
    )


def telemetry(s: StateSnapshot, env: int, t: float, action: int) -> str:
    x, y = s.pos[env]
    vx, vy = s.vel[env]
    return (
        f"t={t:6.1f}s  x={x:8.1f}  y={y:7.1f}  vx={vx:7.1f}  vy={vy:7.1f}  "
        f"theta={math.degrees(s.theta[env]):7.1f}deg  omega={s.omega[env]:6.2f}  "
        f"thrusters={describe_action(action)}  {STATUS_NAMES[int(s.status[env])]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--controller", choices=CONTROLLERS, default="heuristic")
    parser.add_argument("--device", default=None, help="Warp device, e.g. cpu or cuda:0")
    parser.add_argument("--print-every", type=float, default=2.0, help="sim seconds")
    args = parser.parse_args()

    sim = Simulator(args.envs, device=args.device)
    controller = CONTROLLERS[args.controller](sim.config)
    cfg = sim.config
    print(f"{args.envs} envs on {sim.device}, controller={args.controller}, env 0 telemetry:")

    print_interval = max(1, round(args.print_every / cfg.policy_dt))
    wp.synchronize()
    start = time.perf_counter()
    steps = 0
    for k in range(cfg.max_steps):
        s = sim.snapshot()
        if not (s.status == FLYING).any():
            break
        actions = controller(s)
        if k % print_interval == 0:
            print(telemetry(s, 0, k * cfg.policy_dt, int(actions[0])))
        sim.step(actions)
        steps += 1
    wp.synchronize()
    elapsed = time.perf_counter() - start

    s = sim.snapshot()
    print(telemetry(s, 0, s.steps[0] * cfg.policy_dt, 0))
    print("\noutcomes:")
    for code, name in STATUS_NAMES.items():
        count = int((s.status == code).sum())
        if count:
            print(f"  {name:14s} {count:7d}  ({100 * count / args.envs:5.1f}%)")
    landed = s.status == 1
    if landed.any():
        print(f"  mean touchdown speed (landed): {np.abs(s.vel[landed, 1]).mean():.2f} m/s")
    env_steps = steps * args.envs
    print(
        f"\n{steps} policy steps x {args.envs} envs in {elapsed:.2f}s "
        f"(includes host-side controller): {env_steps / elapsed:,.0f} env-steps/s"
    )


if __name__ == "__main__":
    main()
