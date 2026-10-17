"""Stream batched 6-DOF flights to Rerun: rockets with moving flaps and plumes, trajectories and
telemetry.

    uv run starship-rerun                         # 8 passive belly-flops from 3 km, live viewer
    uv run starship-rerun --save out/run.rrd      # record to a file instead
    uv run starship-rerun --schedule out/cem_best.npz   # replay a planned trajectory
    uv run starship-rerun --policy logs/<run>/policy_best.pt --stage 0   # fly a trained policy
    uv run starship-rerun --progress logs/<run> --stage 2   # checkpoints side by side
"""

import argparse
import math
from pathlib import Path

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
                # /world/chase moves with rocket 0 but never rotates, so this camera follows the
                # rocket while its pitch and roll stay visible against the world.
                rrb.Spatial3DView(
                    name="chase (rocket 0)",
                    origin="/world/chase",
                    contents="/world/**",
                    line_grid=False,
                    eye_controls=rrb.EyeControls3D(
                        position=(90.0, -90.0, 30.0), look_target=(0.0, 0.0, 0.0)
                    ),
                ),
                rrb.Spatial3DView(
                    name="flights",
                    origin="/world",
                    eye_controls=rrb.EyeControls3D(
                        position=(2500.0, -2500.0, 1500.0), look_target=(0.0, 0.0, 1000.0)
                    ),
                ),
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


def log_step(cfg, s, flaps, trails, mass_params, first=0, shift=(0.0, 0.0, 0.0), label=None):
    """Log rockets 0..len(trails)-1 of snapshot `s` as entities first.. onwards, moved by
    `shift`; a rocket whose trail is None is finished and stays where it was last logged."""
    n = len(trails)
    shift = np.asarray(shift, dtype=float)
    coms = mass_sample(mass_params, s.tanks[:n], device="cpu")[:, 1]
    pitch = np.degrees(np.arccos(np.clip(body_axis_z(s.rot), -1, 1)))
    engine = cfg.vehicle.engine
    # After rocket 0's episode ends its env may respawn elsewhere; the camera stays where it ended.
    if first == 0 and trails[0] is not None:
        rr.log("world/chase", rr.Transform3D(translation=s.pos[0] + shift))
    for i in range(n):
        if trails[i] is None:
            continue
        q = s.rot[i]
        origin = s.pos[i] + shift - rotate(q, np.array([coms[i], 0.0, 0.0]))
        e = first + i
        rr.log(
            f"world/rocket/{e}",
            rr.Transform3D(translation=origin, quaternion=rr.Quaternion(xyzw=q)),
        )
        quads = [flap_quad(f, s.flaps[i, k]) for k, f in enumerate(flaps)]
        rr.log(
            f"world/rocket/{e}/flaps",
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
            f"world/rocket/{e}/plumes",
            rr.Cylinders3D(
                lengths=[30 * s.spool[i, k] for k in range(3) if lit[k]],
                radii=[0.6] * int(lit.sum()),
                centers=[c for c, on in zip(centers, lit, strict=True) if on],
                rotation_axis_angles=[rr.RotationAxisAngle(axis=(0, 1, 0), degrees=90)]
                * int(lit.sum()),
                colors=[PLUME_COLOR],
            ),
        )
        trails[i].append(s.pos[i] + shift)
        rr.log(f"world/trail/{e}", rr.LineStrips3D([trails[i]], colors=[(120, 160, 255)]))
        name = e if label is None else f"{label}/{i}"
        rr.log(f"telemetry/altitude/{name}", rr.Scalars(s.pos[i, 2]))
        rr.log(f"telemetry/vz/{name}", rr.Scalars(s.vel[i, 2]))
        rr.log(f"telemetry/pitch/{name}", rr.Scalars(pitch[i]))
        if s.status[i] != FLYING:
            trails[i] = None


def run_configs(run: Path, stage: int | None):
    """Sim, obs and reward configs of a training run at a curriculum stage (default: last)."""
    from omegaconf import OmegaConf

    from starship.train.conf import TrainConfig
    from starship.train.curriculum import Curriculum

    run_cfg = OmegaConf.merge(
        OmegaConf.structured(TrainConfig), OmegaConf.load(run / ".hydra/config.yaml")
    )
    sim_cfg, reward_cfg = OmegaConf.to_object(run_cfg.sim), OmegaConf.to_object(run_cfg.reward)
    name = "no curriculum"
    if run_cfg.curriculum.enabled:
        curriculum = Curriculum(run_cfg, run_cfg.curriculum)
        stage = len(curriculum.stages) - 1 if stage is None else stage
        sim_cfg, reward_cfg = curriculum.configs(stage)
        name = curriculum.stages[stage]["name"]
    return sim_cfg, OmegaConf.to_object(run_cfg.obs), reward_cfg, name


def fly_progress(args) -> None:
    """Fly evenly spaced checkpoints of a run on identical spawns, one lane per checkpoint."""
    import torch

    from starship.train.env import RocketEnv
    from starship.train.rsl_env import RslRocketEnv

    run = Path(args.progress)
    snapshots = sorted(run.glob("policy_[0-9]*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    if not snapshots:
        raise SystemExit(f"no policy_<it>.pt checkpoints in {run}")
    picks = np.unique(np.linspace(0, len(snapshots) - 1, args.checkpoints).round().astype(int))
    chosen = [snapshots[i] for i in picks][::-1]  # newest first: lane 0 is the chase camera's
    sim_cfg, obs_cfg, reward_cfg, name = run_configs(run, args.stage)
    print(f"stage: {name}; checkpoints: {', '.join(f.stem for f in chosen)}")
    lanes = []
    for f in chosen:
        env = RslRocketEnv(
            RocketEnv(args.envs, sim_cfg, obs_cfg, reward_cfg, device=args.device, seed=args.seed)
        )
        policy = torch.load(f, map_location=env.device, weights_only=False).eval()
        policy.reset()
        label = f"it{f.stem.split('_')[1]}"
        lanes.append({"env": env, "policy": policy, "obs": env.get_observations(), "label": label,
                      "trails": [[] for _ in range(args.envs)], "outcomes": []})  # fmt: skip
    spacing = 600.0
    layout = blueprint()
    rr.init("starship", default_blueprint=layout)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save, default_blueprint=layout)
    else:
        rr.spawn()
    rr.send_blueprint(layout)
    log_scene(sim_cfg, args.envs * len(lanes))
    for j, lane in enumerate(lanes):
        rr.log(
            f"world/labels/{lane['label']}",
            rr.Points3D([(j * spacing, -250.0, 0.0)], labels=[lane["label"]], radii=8.0),
            static=True,
        )
    flaps = flap_geometry(sim_cfg)
    mass_params = make_mass_params(sim_cfg.vehicle)
    dt = sim_cfg.dt * sim_cfg.substeps
    with torch.inference_mode():
        for k in range(lanes[0]["env"].max_episode_length):
            rr.set_time("sim_time", duration=k * dt)
            for j, lane in enumerate(lanes):
                env, trails = lane["env"], lane["trails"]
                if all(t is None for t in trails):
                    continue
                log_step(
                    sim_cfg,
                    env.env.sim.snapshot(),
                    flaps,
                    trails,
                    mass_params,
                    first=j * args.envs,
                    shift=(j * spacing, 0.0, 0.0),
                    label=lane["label"],
                )
                lane["obs"], _, dones, _ = env.step(
                    lane["policy"](lane["obs"], stochastic_output=False)
                )
                for i in torch.nonzero(dones).flatten().tolist():
                    if trails[i] is not None:
                        trails[i] = None
                        lane["outcomes"].append(int(env.final_status[i]))
                lane["policy"].reset(dones)
            if all(all(t is None for t in lane["trails"]) for lane in lanes):
                break
    names = {1: "landed", 2: "crashed", 3: "out", 4: "timeout", 5: "tumbled"}
    for lane in lanes:
        o = lane["outcomes"]
        print(
            f"{lane['label']:>8}: " + ", ".join(f"{names[c]} {o.count(c)}" for c in sorted(set(o)))
        )


def _subset(snapshot, idx):
    from dataclasses import fields, replace

    return replace(snapshot, **{f.name: getattr(snapshot, f.name)[idx] for f in fields(snapshot)})


def _rollout(path, cfgs, args, record=None):
    """Fly a policy deterministically once per env; with `record` (env indices), log those."""
    import torch

    from starship.train.env import RocketEnv
    from starship.train.rsl_env import RslRocketEnv

    sim_cfg, obs_cfg, reward_cfg = cfgs
    env = RslRocketEnv(
        RocketEnv(args.envs, sim_cfg, obs_cfg, reward_cfg, device=args.device, seed=args.seed)
    )
    policy = torch.load(path, map_location=env.device, weights_only=False).eval()
    policy.reset()
    obs = env.get_observations()
    status = np.zeros(args.envs, dtype=int)
    pending = np.ones(args.envs, dtype=bool)
    if record is not None:
        flaps = flap_geometry(sim_cfg)
        mass_params = make_mass_params(sim_cfg.vehicle)
        trails = [[] for _ in record]
    dt = sim_cfg.dt * sim_cfg.substeps
    with torch.inference_mode():
        for k in range(env.max_episode_length):
            if record is not None:
                rr.set_time("sim_time", duration=k * dt)
                log_step(
                    sim_cfg, _subset(env.env.sim.snapshot(), record), flaps, trails, mass_params
                )
            obs, _, dones, _ = env.step(policy(obs, stochastic_output=False))
            done = dones.cpu().numpy().astype(bool) & pending
            status[done] = env.final_status.cpu().numpy()[done]
            if record is not None:
                for j, i in enumerate(record):
                    if done[i]:
                        trails[j] = None
            pending &= ~done
            policy.reset(dones)
            if not pending.any():
                break
    return status


def fly_policy(args) -> None:
    """Run a policy on its run's config (curriculum stage `args.stage`, default the last one)."""
    path = Path(args.policy)
    sim_cfg, obs_cfg, reward_cfg, name = run_configs(path.parent, args.stage)
    cfgs = (sim_cfg, obs_cfg, reward_cfg)
    print(f"stage: {name}")
    names = {1: "landed", 2: "crashed", 3: "out of bounds", 4: "timeout", 5: "tumbled"}
    if args.only_landed:
        # Same seed, same spawns, deterministic policy: a second pass repeats every outcome.
        status = _rollout(path, cfgs, args)
        record = np.nonzero(status == 1)[0][: args.show]
        print(f"{len(status)} episodes: {(status == 1).sum()} landed; recording {len(record)}")
        if not len(record):
            return
    else:
        record = np.arange(min(args.show, args.envs))
    layout = blueprint()
    rr.init("starship", default_blueprint=layout)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save, default_blueprint=layout)
    else:
        rr.spawn()
    rr.send_blueprint(layout)
    log_scene(sim_cfg, len(record))
    status = _rollout(path, cfgs, args, record=record)
    counts = {names[c]: int((status == c).sum()) for c in sorted(set(status.tolist())) if c}
    print(f"{len(status)} episodes: {counts}")


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
    parser.add_argument("--policy", default=None, help="fly an exported policy deterministically")
    parser.add_argument("--stage", type=int, default=None, help="curriculum stage for --policy")
    parser.add_argument("--only-landed", action="store_true", help="record only landing episodes")
    parser.add_argument("--show", type=int, default=16, help="rockets to record with --policy")
    parser.add_argument(
        "--progress", default=None, help="run dir: fly its checkpoints side by side"
    )
    parser.add_argument("--checkpoints", type=int, default=8, help="checkpoints for --progress")
    args = parser.parse_args()
    if args.progress:
        fly_progress(args)
        return
    if args.policy:
        fly_policy(args)
        return

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
    layout = blueprint()
    rr.init("starship", default_blueprint=layout)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save, default_blueprint=layout)
    else:
        rr.spawn()
    # Sent as the active blueprint: the viewer otherwise restores whatever layout it last saved
    # for this application id.
    rr.send_blueprint(layout)
    log_scene(cfg, args.envs)
    flaps = flap_geometry(cfg)
    mass_params = make_mass_params(cfg.vehicle)
    trails = [[] for _ in range(args.envs)]
    steps = int(args.seconds / (cfg.dt * cfg.substeps))
    for k in range(steps):
        rr.set_time("sim_time", duration=k * cfg.dt * cfg.substeps)
        log_step(cfg, sim.snapshot(), flaps, trails, mass_params)
        if all(t is None for t in trails):
            break
        if schedule is not None:
            schedule.apply(k * cfg.dt * cfg.substeps)
        sim.step()


if __name__ == "__main__":
    main()
