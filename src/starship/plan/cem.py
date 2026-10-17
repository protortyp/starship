"""Open-loop trajectory search with the cross-entropy method.

The whole descent is a piecewise-linear schedule of control knots. Each iteration samples
thousands of schedules, flies them all in parallel from the same initial state and refits the
sampling distribution to the best ones. Scores come only from how a flight ends, so no reward
shaping is involved: it answers whether (and how) a landing is reachable at all.

    uv run python -m starship.plan.cem --device cuda:0 --samples 8192 --iterations 60
"""

import argparse
import math
import time
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from starship.sim import CRASHED, FLYING, LANDED, TIMEOUT, SimConfig, Simulator
from starship.sim.physics import Commands

# Knot parameters: engines lit (0..3, rounded), throttle, gimbal towards +b2 and +b3 (shared by
# the lit engines) and the four flap angles.
KNOT_DIM = 8
LOW = np.array([0.0, 0.4, -0.26, -0.26, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
HIGH = np.array([3.0, 1.0, 0.26, 0.26, 1.48, 1.48, 1.48, 1.48], dtype=np.float32)
vec8 = wp.types.vector(length=KNOT_DIM, dtype=float)


@wp.kernel
def schedule_kernel(
    knots: wp.array3d(dtype=float),  # (samples, knots, KNOT_DIM)
    knot_interval: float,
    t: float,
    c: Commands,
):
    i = wp.tid()
    count = knots.shape[1]
    f = wp.clamp(t / knot_interval, 0.0, float(count - 1))
    k = wp.min(int(f), count - 2)
    a = f - float(k)
    u = vec8()
    for d in range(8):
        u[d] = wp.lerp(knots[i, k, d], knots[i, k + 1, d], a)
    lit = int(u[0] + 0.5)
    on = wp.vec3(0.0)
    for e in range(3):
        if e < lit:
            on[e] = 1.0
    c.engine_on[i] = on
    c.throttle[i] = wp.vec3(u[1])
    c.gimbal2[i] = wp.vec3(u[2])
    c.gimbal3[i] = wp.vec3(u[3])
    c.flaps[i] = wp.vec4(u[4], u[5], u[6], u[7])
    c.rcs[i] = 0.0


@dataclass
class Outcome:
    cost: np.ndarray
    status: np.ndarray
    vertical_speed: np.ndarray
    horizontal_speed: np.ndarray
    tilt_deg: np.ndarray
    miss: np.ndarray
    fuel_left: np.ndarray
    time: np.ndarray


def score(sim: Simulator) -> Outcome:
    """Lower is better. Landings beat every crash; crashes are ranked by how far they miss each
    touchdown limit, flights still airborne at the horizon by their height."""
    s = sim.snapshot()
    x, y, z, w = s.rot.T
    axis_z = 2.0 * (x * z - w * y)
    tilt = np.degrees(np.arccos(np.clip(axis_z, -1, 1)))
    vz = -s.vel[:, 2]
    vh = np.hypot(s.vel[:, 0], s.vel[:, 1])
    miss = np.hypot(s.pos[:, 0], s.pos[:, 1])
    fuel = s.tanks.sum(axis=1)
    lim = sim.config.landing
    excess = (
        np.maximum(vz - lim.max_vertical_speed, 0)
        + 2 * np.maximum(vh - lim.max_horizontal_speed, 0)
        + 0.5 * np.maximum(tilt - math.degrees(lim.max_tilt), 0)
        + 0.1 * np.maximum(miss - lim.pad_radius, 0)
    )
    cost = np.where(s.status == LANDED, -1000.0 - fuel / 1000.0, excess)
    airborne = (s.status == FLYING) | (s.status == TIMEOUT)
    cost = np.where(airborne, 500.0 + s.pos[:, 2], cost)
    cost = np.where(~airborne & (s.status != LANDED) & (s.status != CRASHED), 2000.0, cost)
    return Outcome(cost, s.status, vz, vh, tilt, miss, fuel, s.time)


class CrossEntropyPlanner:
    def __init__(
        self,
        config: SimConfig,
        samples: int,
        horizon: float = 80.0,
        knot_interval: float = 2.0,
        elite_fraction: float = 0.05,
        device: str | None = None,
        seed: int = 0,
    ):
        cfg = replace(config, max_time=horizon)
        self.sim = Simulator(samples, cfg, device=device, seed=seed)
        self.samples = samples
        self.knots = int(horizon / knot_interval) + 1
        self.knot_interval = knot_interval
        self.control_dt = cfg.dt * cfg.substeps
        self.steps = int(horizon / self.control_dt)
        self.elites = max(int(samples * elite_fraction), 8)
        self.rng = np.random.default_rng(seed)
        self.mean = np.tile((LOW + HIGH) / 2, (self.knots, 1)).astype(np.float32)
        self.mean[:, 0] = 0.0  # engines off until the search lights them
        self.std = np.tile((HIGH - LOW) / 2, (self.knots, 1)).astype(np.float32)

    def rollout(self, knots: np.ndarray, start: Simulator, index: int = 0) -> Outcome:
        sim = self.sim
        sim.copy_from(start, np.full(self.samples, index, dtype=np.int32))
        plan = wp.array(knots, dtype=float, device=sim.device)
        for k in range(self.steps):
            wp.launch(
                schedule_kernel,
                dim=self.samples,
                inputs=[plan, self.knot_interval, k * self.control_dt, sim.commands],
                device=sim.device,
            )
            sim.step()
            if k % 50 == 49 and not (sim.state.status.numpy() == FLYING).any():
                break
        return score(sim)

    def iterate(self, start: Simulator, index: int = 0) -> tuple[Outcome, np.ndarray]:
        noise = self.rng.standard_normal((self.samples, self.knots, KNOT_DIM)).astype(np.float32)
        knots = np.clip(self.mean + self.std * noise, LOW, HIGH)
        knots[0] = self.mean  # always re-evaluate the current mean
        out = self.rollout(knots, start, index)
        elite = np.argsort(out.cost)[: self.elites]
        self.mean = 0.3 * self.mean + 0.7 * knots[elite].mean(axis=0)
        self.std = np.maximum(0.3 * self.std + 0.7 * knots[elite].std(axis=0), 0.02 * (HIGH - LOW))
        return out, knots[elite[0]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--altitude", type=float, default=3000.0)
    parser.add_argument("--horizon", type=float, default=80.0)
    parser.add_argument("--knot-interval", type=float, default=2.0)
    parser.add_argument("--out", default="out/cem_best.npy")
    args = parser.parse_args()

    cfg = SimConfig()
    cfg.spawn = replace(cfg.spawn, altitude=args.altitude)
    start = Simulator(1, cfg, device=args.device, seed=0)
    planner = CrossEntropyPlanner(
        cfg, args.samples, args.horizon, args.knot_interval, device=args.device
    )
    best_cost, best = math.inf, None
    for it in range(args.iterations):
        t0 = time.perf_counter()
        out, top = planner.iterate(start)
        i = int(np.argmin(out.cost))
        if out.cost[i] < best_cost:
            best_cost, best = float(out.cost[i]), top
        landed = float(np.mean(out.status == LANDED))
        print(
            f"it {it:3d}  best {out.cost[i]:9.1f}  landed {landed:5.1%}  "
            f"vz {out.vertical_speed[i]:6.1f}  vh {out.horizontal_speed[i]:5.1f}  "
            f"tilt {out.tilt_deg[i]:5.1f}  miss {out.miss[i]:6.1f}  "
            f"fuel {out.fuel_left[i] / 1000:5.1f} t  "
            f"t {out.time[i]:5.1f} s  ({time.perf_counter() - t0:.1f} s)",
            flush=True,
        )
    if best is not None:
        np.save(args.out, best)
        print(f"saved best schedule to {args.out}")


if __name__ == "__main__":
    main()
