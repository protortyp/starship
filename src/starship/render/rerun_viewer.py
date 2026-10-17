"""Stream batched 6-DOF flights to Rerun: rockets with moving flaps and plumes, trajectories and
telemetry.

    uv run starship-rerun                         # 8 passive belly-flops from 3 km, live viewer
    uv run starship-rerun --save out/run.rrd      # record to a file instead
    uv run starship-rerun --schedule out/cem_best.npz   # replay a planned trajectory
"""

import argparse
import math

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from starship.sim import FLYING, SimConfig, Simulator
from starship.sim.aero import make_aero_params
from starship.sim.mass import make_mass_params
from starship.sim.mass import sample as mass_sample

HULL_COLOR = (205, 205, 210)
FLAP_COLOR = (60, 60, 70)
PLUME_COLOR = (255, 150, 40)
SEGMENTS = 24


def hull_mesh(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Surface of revolution of the hull profile in body coordinates from the engine plane."""
    h = cfg.vehicle.hull
    r0 = h.diameter / 2
    base = h.length - h.nose_length
    ogive = (h.nose_length**2 + r0**2) / (2 * r0)
    xs = [0.0, base, *(base + h.nose_length * np.linspace(0.1, 1.0, 8))]
    rs = [
        r0 if x <= base else max(math.sqrt(ogive**2 - (x - base) ** 2) + r0 - ogive, 0.0)
        for x in xs
    ]
    phi = np.linspace(0, 2 * np.pi, SEGMENTS, endpoint=False)
    verts = np.array(
        [(x, r * np.sin(a), r * np.cos(a)) for x, r in zip(xs, rs, strict=True) for a in phi]
    )
    tris = []
    for i in range(len(xs) - 1):
        for j in range(SEGMENTS):
            a, b = i * SEGMENTS + j, i * SEGMENTS + (j + 1) % SEGMENTS
            tris += [(a, b, a + SEGMENTS), (b, b + SEGMENTS, a + SEGMENTS)]
    return verts, np.array(tris)


def flap_geometry(cfg: SimConfig) -> list[dict]:
    """Hinge point, hinge axis and open/folded span directions, as the aero model uses them."""
    p = make_aero_params(cfg.vehicle, device="cpu")
    out = []
    for i, f in enumerate(cfg.vehicle.flaps):
        out.append(
            {
                "hinge": p.hinge.numpy()[i],
                "axis": p.hinge_axis.numpy()[i],
                "open": p.span_open.numpy()[i],
                "folded": p.span_folded.numpy()[i],
                "chord": f.area / f.span,
                "span": f.span,
            }
        )
    return out


def flap_quad(f: dict, angle: float) -> np.ndarray:
    span = math.cos(angle) * f["open"] + math.sin(angle) * f["folded"]
    half = 0.5 * f["chord"] * f["axis"]
    out = f["span"] * span
    return np.array(
        [f["hinge"] - half, f["hinge"] + half, f["hinge"] + half + out, f["hinge"] - half + out]
    )


def body_axis_z(rot: np.ndarray) -> np.ndarray:
    """World z component of the body axis b1 for (N, 4) quaternions (x, y, z, w)."""
    x, y, z, w = rot.T
    return 2.0 * (x * z - w * y)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                # Rooted at the rocket's own transform, so the camera rides along with it.
                rrb.Spatial3DView(name="follow (rocket 0)", origin="/world/rocket/0"),
                rrb.Spatial3DView(name="flights", origin="/world"),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="altitude (m)", origin="/telemetry/altitude"),
                rrb.TimeSeriesView(name="vertical speed (m/s)", origin="/telemetry/vz"),
                rrb.TimeSeriesView(name="pitch from vertical (deg)", origin="/telemetry/pitch"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def log_scene(cfg: SimConfig, n: int) -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    pad = cfg.landing.pad_radius
    ring = [(pad * math.cos(a), pad * math.sin(a), 0.0) for a in np.linspace(0, 2 * np.pi, 49)]
    rr.log("world/pad", rr.LineStrips3D([ring], colors=[(240, 200, 40)]), static=True)
    g = 3000.0
    rr.log(
        "world/ground",
        rr.Mesh3D(
            vertex_positions=[(-g, -g, -0.5), (g, -g, -0.5), (g, g, -0.5), (-g, g, -0.5)],
            triangle_indices=[(0, 1, 2), (0, 2, 3)],
            albedo_factor=(70, 90, 70),
        ),
        static=True,
    )
    verts, tris = hull_mesh(cfg)
    for i in range(n):
        rr.log(
            f"world/rocket/{i}/hull",
            rr.Mesh3D(vertex_positions=verts, triangle_indices=tris, albedo_factor=HULL_COLOR),
            static=True,
        )


def log_step(cfg, sim, flaps, trails, mass_params) -> None:
    s = sim.snapshot()
    coms = mass_sample(mass_params, s.tanks, device="cpu")[:, 1]
    pitch = np.degrees(np.arccos(np.clip(body_axis_z(s.rot), -1, 1)))
    engine = cfg.vehicle.engine
    for i in range(sim.num_envs):
        if s.status[i] != FLYING and trails[i] is None:
            continue
        q = s.rot[i]
        origin = s.pos[i] - rotate(q, np.array([coms[i], 0.0, 0.0]))
        rr.log(
            f"world/rocket/{i}",
            rr.Transform3D(translation=origin, quaternion=rr.Quaternion(xyzw=q)),
        )
        quads = [flap_quad(f, s.flaps[i, k]) for k, f in enumerate(flaps)]
        rr.log(
            f"world/rocket/{i}/flaps",
            rr.Mesh3D(
                vertex_positions=np.concatenate(quads),
                triangle_indices=[
                    (4 * k + a, 4 * k + b, 4 * k + c)
                    for k in range(4)
                    for a, b, c in ((0, 1, 2), (0, 2, 3))
                ],
                albedo_factor=FLAP_COLOR,
            ),
        )
        lit = s.spool[i] > 0.01
        centers = [
            (
                -0.5 * 30 * s.spool[i, k],
                engine.pivot_radius * math.sin(a),
                engine.pivot_radius * math.cos(a),
            )
            for k, a in enumerate(engine.azimuths)
        ]
        rr.log(
            f"world/rocket/{i}/plumes",
            rr.Cylinders3D(
                lengths=[30 * s.spool[i, k] for k in range(3) if lit[k]],
                radii=[0.6] * int(lit.sum()),
                centers=[c for c, on in zip(centers, lit, strict=True) if on],
                rotation_axis_angles=[rr.RotationAxisAngle(axis=(0, 1, 0), degrees=90)]
                * int(lit.sum()),
                colors=[PLUME_COLOR],
            ),
        )
        trails[i].append(s.pos[i].copy())
        rr.log(f"world/trail/{i}", rr.LineStrips3D([trails[i]], colors=[(120, 160, 255)]))
        rr.log(f"telemetry/altitude/{i}", rr.Scalars(s.pos[i, 2]))
        rr.log(f"telemetry/vz/{i}", rr.Scalars(s.vel[i, 2]))
        rr.log(f"telemetry/pitch/{i}", rr.Scalars(pitch[i]))
        if s.status[i] != FLYING:
            trails[i] = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--altitude", type=float, default=3000.0)
    parser.add_argument("--offset-noise", type=float, default=300.0)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save", default=None, help="write an .rrd file instead of spawning a viewer"
    )
    parser.add_argument("--schedule", default=None, help="replay a starship.plan.cem schedule")
    args = parser.parse_args()

    cfg = SimConfig()
    cfg.spawn.altitude = args.altitude
    cfg.spawn.offset_noise = args.offset_noise
    cfg.spawn.pitch_noise = math.radians(5.0)
    schedule = None
    if args.schedule:
        # The planner flies one env from the default spawn with seed 0; match it exactly.
        cfg.spawn.offset_noise = cfg.spawn.pitch_noise = 0.0
        args.envs, args.seed = 1, 0
    sim = Simulator(args.envs, cfg, device=args.device, seed=args.seed)
    if args.schedule:
        from starship.plan.cem import ScheduleReplay

        schedule = ScheduleReplay(args.schedule, sim)
    rr.init("starship", default_blueprint=blueprint())
    if args.save:
        rr.save(args.save, default_blueprint=blueprint())
    else:
        rr.spawn()
    # Sent as the active blueprint: the viewer otherwise restores whatever layout it last saved
    # for this application id.
    rr.send_blueprint(blueprint())
    log_scene(cfg, args.envs)
    flaps = flap_geometry(cfg)
    mass_params = make_mass_params(cfg.vehicle)
    trails = [[] for _ in range(args.envs)]
    steps = int(args.seconds / (cfg.dt * cfg.substeps))
    for k in range(steps):
        rr.set_time("sim_time", duration=k * cfg.dt * cfg.substeps)
        log_step(cfg, sim, flaps, trails, mass_params)
        if all(t is None for t in trails):
            break
        if schedule is not None:
            schedule.apply(k * cfg.dt * cfg.substeps)
        sim.step()


if __name__ == "__main__":
    main()
