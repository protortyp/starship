"""Stream a simulation run to Rerun: a rocket-following view, a world overview, and telemetry.

Run `uv run starship-rerun` to spawn the Rerun viewer, or `--save run.rrd` to write a recording
that can be opened later with `rerun run.rrd`.

Rerun's 2D views are y-down, so world y is negated when logging.
"""

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from starship.control import HeuristicController, IdleController, RandomController
from starship.render.geometry import hull_outline, plumes
from starship.sim import (
    CRASHED,
    FLYING,
    LANDED,
    LEFT,
    MAIN,
    OUT_OF_BOUNDS,
    RIGHT,
    TIMEOUT,
    SimConfig,
    Simulator,
    StateSnapshot,
)

STATUS_COLORS = {
    FLYING: (205, 210, 218),
    LANDED: (90, 220, 120),
    CRASHED: (240, 90, 80),
    OUT_OF_BOUNDS: (240, 170, 60),
    TIMEOUT: (240, 170, 60),
}
TRACKED_COLOR = (90, 170, 255)
FLAME_COLOR = (255, 150, 40)
GROUND_HALF_WIDTH = 5000.0
TRAIL_LOG_EVERY = 5


def flip(points: np.ndarray) -> np.ndarray:
    return np.asarray(points) * np.array([1.0, -1.0])


def blueprint(cfg: SimConfig) -> rrb.Blueprint:
    spawn_y = cfg.spawn.y
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                # Rooted at "/", this view sees the world through the transform that keeps the
                # tracked rocket at the origin.
                rrb.Spatial2DView(
                    name="follow (env 0)",
                    origin="/",
                    contents=["+ world/**", "- world/markers"],
                    visual_bounds=rrb.VisualBounds2D(x_range=[-90, 90], y_range=[-60, 60]),
                ),
                rrb.Spatial2DView(
                    name="overview",
                    origin="world",
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[-0.6 * spawn_y, 0.6 * spawn_y], y_range=[-1.05 * spawn_y, 50]
                    ),
                ),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="altitude [m]", origin="plots/altitude"),
                rrb.TimeSeriesView(name="velocity [m/s]", origin="plots/velocity"),
                rrb.TimeSeriesView(name="attitude [deg, deg/s]", origin="plots/attitude"),
                rrb.TimeSeriesView(name="thrusters", origin="plots/thrusters"),
                rrb.TimeSeriesView(name="outcomes [envs]", origin="plots/outcomes"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def log_static_scene(cfg: SimConfig) -> None:
    for name, color in (
        ("flying", STATUS_COLORS[FLYING]),
        ("landed", STATUS_COLORS[LANDED]),
        ("crashed", STATUS_COLORS[CRASHED]),
    ):
        rr.log(f"plots/outcomes/{name}", rr.SeriesLines(colors=[color], names=[name]), static=True)
    rr.log(
        "world/ground",
        rr.LineStrips2D(
            [[[-GROUND_HALF_WIDTH, 0.0], [GROUND_HALF_WIDTH, 0.0]]], colors=[(110, 100, 85)]
        ),
        static=True,
    )
    half = cfg.landing.pad_half_width
    rr.log(
        "world/pad",
        rr.Boxes2D(mins=[[-half, 0.0]], sizes=[[2 * half, 1.5]], colors=[(240, 200, 60)]),
        static=True,
    )


def log_step(cfg: SimConfig, s: StateSnapshot, actions: np.ndarray, trail: list) -> None:
    r = cfg.rocket
    pos0 = s.pos[0]
    rr.log("world", rr.Transform3D(translation=[-pos0[0], pos0[1], 0.0]))

    outlines = hull_outline(s.pos, s.theta, r)
    closed = np.concatenate([outlines, outlines[:, :1]], axis=1)
    colors = [STATUS_COLORS[int(st)] for st in s.status]
    colors[0] = TRACKED_COLOR if s.status[0] == FLYING else colors[0]
    rr.log("world/rockets", rr.LineStrips2D([flip(c) for c in closed], colors=colors, radii=0.25))

    flames = []
    for i in np.flatnonzero(s.status == FLYING):
        for tri in plumes(s.pos[i], float(s.theta[i]), int(actions[i]), r):
            flames.append(flip(np.vstack([tri, tri[:1]])))
    rr.log("world/flames", rr.LineStrips2D(flames, colors=FLAME_COLOR, radii=0.35))
    # Fixed screen-size markers keep every rocket visible in the kilometer-scale overview.
    rr.log(
        "world/markers",
        rr.Points2D(flip(s.pos), colors=colors, radii=rr.Radius.ui_points(4.0)),
    )

    trail.append(pos0.copy())
    if len(trail) % TRAIL_LOG_EVERY == 1:
        rr.log(
            "world/trail", rr.LineStrips2D([flip(np.array(trail))], colors=TRACKED_COLOR, radii=0.4)
        )

    theta = float(s.theta[0])
    bottom = (
        pos0[1] - 0.5 * r.length * abs(math.cos(theta)) - 0.5 * r.diameter * abs(math.sin(theta))
    )
    rr.log("plots/altitude", rr.Scalars(bottom))
    rr.log("plots/velocity/vx", rr.Scalars(s.vel[0, 0]))
    rr.log("plots/velocity/vy", rr.Scalars(s.vel[0, 1]))
    rr.log("plots/attitude/tilt", rr.Scalars(math.degrees(theta)))
    rr.log("plots/attitude/spin", rr.Scalars(math.degrees(s.omega[0])))
    a0 = int(actions[0]) if s.status[0] == FLYING else 0
    # Offsets stack the three on/off signals so they don't overlap in one plot.
    rr.log("plots/thrusters/left", rr.Scalars(2.4 + 0.8 * bool(a0 & LEFT)))
    rr.log("plots/thrusters/main", rr.Scalars(1.2 + 0.8 * bool(a0 & MAIN)))
    rr.log("plots/thrusters/right", rr.Scalars(0.0 + 0.8 * bool(a0 & RIGHT)))
    rr.log("plots/outcomes/flying", rr.Scalars(int((s.status == FLYING).sum())))
    rr.log("plots/outcomes/landed", rr.Scalars(int((s.status == LANDED).sum())))
    rr.log("plots/outcomes/crashed", rr.Scalars(int((s.status == CRASHED).sum())))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--controller", choices=("heuristic", "random", "idle"), default="heuristic"
    )
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--spawn-noise", type=float, default=0.0, help="spawn randomization scale")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", default=None, help="Warp device, e.g. cpu or cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, default=None, help="write an .rrd instead of spawning")
    args = parser.parse_args()

    cfg = replace(SimConfig(), spawn=SimConfig().spawn.with_noise(args.spawn_noise))
    sim = Simulator(args.envs, cfg, device=args.device, seed=args.seed)
    controller = {
        "heuristic": lambda: HeuristicController(cfg),
        "random": lambda: RandomController(args.seed),
        "idle": IdleController,
    }[args.controller]()

    rr.init("starship", spawn=args.save is None, default_blueprint=blueprint(cfg))
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save, default_blueprint=blueprint(cfg))
    log_static_scene(cfg)

    t = 0.0
    for episode in range(args.episodes):
        if episode > 0:
            sim.reset()
        trail: list = []
        s = sim.snapshot()
        while (s.status == FLYING).any():
            actions = controller(s)
            rr.set_time("sim_time", duration=t)
            rr.set_time("episode", sequence=episode)
            log_step(cfg, s, actions, trail)
            sim.step(actions)
            s = sim.snapshot()
            t += cfg.policy_dt
        rr.set_time("sim_time", duration=t)
        log_step(cfg, s, np.zeros(args.envs, dtype=np.int32), trail)
        landed = int((s.status == LANDED).sum())
        print(f"episode {episode}: {landed}/{args.envs} landed")


if __name__ == "__main__":
    main()
