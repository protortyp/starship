"""Export one landing of a trained policy as a 60 fps animation track for offline renderers.

    uv run python -m starship.render.export --policy logs/final_run/policy_best.pt --stage 4 \
        --seed 3 --out out/landing.json [--usd out/landing.usda]

Flies one deterministic episode per seed on the CPU (trying `--tries` seeds in a row until one
lands), records the state after every control step and resamples it to `--fps`. The JSON holds
the static vehicle geometry the renderer needs (hull profile, flap hinges, engine positions) and
per-frame tracks: world position of the engine-plane origin, body quaternion, flap angles,
per-engine spool, throttle and gimbals, altitude, velocity and status. Frames and axes follow
`starship.sim.config`: world z up, body b1 engines -> nose, b2 starboard, b3 belly.

The optional USDA file is written by hand (no pxr dependency): hull mesh, flap plates, engine
nozzle anchors and the camera-friendly time-sampled transforms, 60 samples per second.
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from starship.sim import FLYING, LANDED, SimConfig
from starship.sim.mass import make_mass_params
from starship.sim.mass import sample as mass_sample

STATUS_NAMES = {
    0: "flying",
    1: "landed",
    2: "crashed",
    3: "out_of_bounds",
    4: "timeout",
    5: "tumbled",
}
ENGINE_FIELDS = ("spool", "throttle", "gimbal2", "gimbal3")


@dataclass
class Samples:
    """Control-step samples of one episode; arrays have one row per sample."""

    dt: float
    time: np.ndarray  # (K,)
    com_position: np.ndarray  # (K, 3) world, centre of mass
    com_station: np.ndarray  # (K,) station of the centre of mass along b1
    velocity: np.ndarray  # (K, 3)
    quaternion: np.ndarray  # (K, 4) x, y, z, w body -> world
    omega: np.ndarray  # (K, 3) body frame
    flaps: np.ndarray  # (K, 4) rad
    engine: dict[str, np.ndarray] = field(default_factory=dict)  # each (K, 3)
    status: int = FLYING

    @property
    def landed(self) -> bool:
        return self.status == LANDED


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors (..., 3) by quaternions (..., 4) in x, y, z, w order."""
    u, w = q[..., :3], q[..., 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def hull_profile(cfg: SimConfig, nose_points: int = 24) -> list[list[float]]:
    """(station, radius) pairs from the engine plane to the nose tip."""
    h = cfg.vehicle.hull
    r0 = h.diameter / 2
    base = h.length - h.nose_length
    ogive = (h.nose_length**2 + r0**2) / (2 * r0)
    xs = [0.0, base, *(base + h.nose_length * np.linspace(0.0, 1.0, nose_points + 1)[1:])]
    out = []
    for x in xs:
        r = r0 if x <= base else max(math.sqrt(ogive**2 - (x - base) ** 2) + r0 - ogive, 0.0)
        out.append([float(x), float(r)])
    return out


def vehicle_geometry(cfg: SimConfig) -> dict:
    """Static geometry in body coordinates (station along b1, then b2, b3) for the renderer."""
    from starship.render.rerun_viewer import flap_geometry

    v = cfg.vehicle
    e = v.engine
    exit_diameter = 2.0 * math.sqrt(e.exit_area / math.pi)
    engines = [
        {
            "azimuth": float(a),
            "radius": e.pivot_radius,
            "pivot_station": e.pivot_station,
            # Nozzle exit on the engine plane at r*(cos(phi) b3 + sin(phi) b2).
            "exit": [0.0, e.pivot_radius * math.sin(a), e.pivot_radius * math.cos(a)],
            "exit_diameter": exit_diameter,
        }
        for a in e.azimuths
    ]
    flaps = []
    for f, g in zip(v.flaps, flap_geometry(cfg), strict=True):
        flaps.append(
            {
                "azimuth": f.azimuth,
                "cant": f.cant,
                "hinge_bottom": f.hinge_bottom,
                "hinge_top": f.hinge_top,
                "area": f.area,
                "span": f.span,
                "chord": g["chord"],
                "hinge": [float(x) for x in g["hinge"]],
                "axis": [float(x) for x in g["axis"]],
                "open": [float(x) for x in g["open"]],
                "folded": [float(x) for x in g["folded"]],
            }
        )
    return {
        "length": v.hull.length,
        "diameter": v.hull.diameter,
        "nose_length": v.hull.nose_length,
        "skirt_length": v.hull.skirt_length,
        "hull_profile": hull_profile(cfg),
        "engines": engines,
        "flaps": flaps,
        "flap_order": ["aft_starboard", "aft_port", "forward_starboard", "forward_port"],
        "gimbal_limit": e.gimbal_limit,
        "flap_max_angle": v.flap_actuator.max_angle,
        "pad_radius": cfg.landing.pad_radius,
    }


def run_configs(policy: Path, stage: int | None):
    from starship.render.rerun_viewer import run_configs as _run_configs

    sim_cfg, obs_cfg, reward_cfg, name = _run_configs(policy.parent, stage)
    return (sim_cfg, obs_cfg, reward_cfg), name


def load_policy(path: Path, device: str = "cpu"):
    import torch

    policy = torch.load(path, map_location=device, weights_only=False).eval()
    policy.reset()
    return policy


def record_episode(policy, cfgs, seed: int, device: str = "cpu") -> Samples:
    """Fly env 0 once with a deterministic policy and sample its state after every control step.

    `policy(obs, stochastic_output=False)` returns actions and `policy.reset(dones)` clears its
    recurrent state, as rsl_rl actors do. The final sample is the terminal state the simulator
    recorded at the end of the episode (the env itself respawns in the same step).
    """
    import torch

    from starship.train.env import RocketEnv
    from starship.train.rsl_env import RslRocketEnv

    sim_cfg, obs_cfg, reward_cfg = cfgs
    env = RslRocketEnv(RocketEnv(1, sim_cfg, obs_cfg, reward_cfg, device=device, seed=seed))
    sim = env.env.sim
    mass_params = make_mass_params(sim_cfg.vehicle)
    dt = sim_cfg.dt * sim_cfg.substeps
    rows: list[dict] = []

    def row(pos, vel, rot, omega, tanks, engine, flaps):
        com = float(mass_sample(mass_params, tanks[None], device=device)[0, 1])
        return {
            "com_position": np.asarray(pos, dtype=float),
            "com_station": com,
            "velocity": np.asarray(vel, dtype=float),
            "quaternion": np.asarray(rot, dtype=float),
            "omega": np.asarray(omega, dtype=float),
            "flaps": np.asarray(flaps, dtype=float),
            "engine": {k: np.asarray(engine[k], dtype=float) for k in ENGINE_FIELDS},
        }

    def engine_state():
        e = sim.state.engine.numpy()[0]
        return {k: np.array(e[k]) for k in ENGINE_FIELDS}

    obs = env.get_observations()
    status = FLYING
    with torch.inference_mode():
        for _ in range(env.max_episode_length):
            s = sim.snapshot()
            engine = engine_state()
            rows.append(
                row(s.pos[0], s.vel[0], s.rot[0], s.omega[0], s.tanks[0], engine, s.flaps[0])
            )
            obs, _, dones, _ = env.step(policy(obs, stochastic_output=False))
            if bool(dones[0]):
                t = sim.terminal_snapshot()
                status = int(t.status[0])
                # The terminal record has no actuator state; the last flying one is 0.1 s old.
                rows.append(
                    row(
                        t.pos[0],
                        t.vel[0],
                        t.rot[0],
                        t.omega[0],
                        t.tanks[0],
                        engine,
                        rows[-1]["flaps"],
                    )
                )
                break
            policy.reset(dones)
    k = len(rows)
    return Samples(
        dt=dt,
        time=np.arange(k) * dt,
        com_position=np.stack([r["com_position"] for r in rows]),
        com_station=np.array([r["com_station"] for r in rows]),
        velocity=np.stack([r["velocity"] for r in rows]),
        quaternion=np.stack([r["quaternion"] for r in rows]),
        omega=np.stack([r["omega"] for r in rows]),
        flaps=np.stack([r["flaps"] for r in rows]),
        engine={f: np.stack([r["engine"][f] for r in rows]) for f in ENGINE_FIELDS},
        status=status,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(max=1.0)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    small = sin < 1e-6
    w0 = np.where(small, 1.0 - u, np.sin((1.0 - u) * theta) / np.where(small, 1.0, sin))
    w1 = np.where(small, u, np.sin(u * theta) / np.where(small, 1.0, sin))
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _hermite(p0, p1, v0, v1, h, u):
    u2, u3 = u * u, u * u * u
    return (
        (2 * u3 - 3 * u2 + 1) * p0
        + (u3 - 2 * u2 + u) * h * v0
        + (-2 * u3 + 3 * u2) * p1
        + (u3 - u2) * h * v1
    )


def resample(samples: Samples, fps: float = 60.0) -> dict:
    """Interpolate the control-step samples to `fps`: cubic Hermite positions using the recorded
    velocities, slerp attitudes, linear actuators. Returns column arrays, one row per frame."""
    k = len(samples.time)
    if k < 2:
        raise ValueError("an episode needs at least two samples")
    t_end = samples.time[-1]
    n = math.floor(t_end * fps + 1e-6) + 1
    t = np.arange(n) / fps
    idx = np.clip(np.searchsorted(samples.time, t, side="right") - 1, 0, k - 2)
    h = samples.dt
    u = ((t - samples.time[idx]) / h).clip(0.0, 1.0)
    uc = u[:, None]

    com = _hermite(
        samples.com_position[idx],
        samples.com_position[idx + 1],
        samples.velocity[idx],
        samples.velocity[idx + 1],
        h,
        uc,
    )
    quat = _slerp(samples.quaternion[idx], samples.quaternion[idx + 1], uc)
    # Consistent sign across frames keeps renderer-side interpolation from flipping.
    for i in range(1, n):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]

    def lerp(a):
        a = np.asarray(a)
        shape = (-1,) + (1,) * (a.ndim - 1)
        w = u.reshape(shape)
        return (1 - w) * a[idx] + w * a[idx + 1]

    com_station = lerp(samples.com_station)
    axis = rotate(quat, np.array([1.0, 0.0, 0.0]))
    base = com - axis * com_station[:, None]
    # Velocity of the base point: v_com + omega x r, with omega taken to the world frame.
    omega_world = rotate(quat, lerp(samples.omega))
    velocity = lerp(samples.velocity) + np.cross(omega_world, -axis * com_station[:, None])
    status = np.full(n, FLYING, dtype=int)
    status[-1] = samples.status
    engine = {f: lerp(samples.engine[f]) for f in ENGINE_FIELDS}
    thrust = engine["spool"] * engine["throttle"]
    return {
        "time": t,
        "position": base,
        "quaternion": quat,
        "com_position": com,
        "com_station": com_station,
        "velocity": velocity,
        "speed": np.linalg.norm(velocity, axis=1),
        "altitude": base[:, 2],
        "flaps": lerp(samples.flaps),
        "spool": engine["spool"],
        "throttle": engine["throttle"],
        "gimbal2": engine["gimbal2"],
        "gimbal3": engine["gimbal3"],
        "thrust": thrust,
        "status": status,
    }


def build_export(samples: Samples, cfg: SimConfig, fps: float = 60.0, meta: dict | None = None):
    frames = resample(samples, fps)
    out = {
        "meta": {
            "fps": fps,
            "control_dt": samples.dt,
            "frames": len(frames["time"]),
            "duration": float(frames["time"][-1]),
            "status": STATUS_NAMES.get(samples.status, str(samples.status)),
            "landed": bool(samples.landed),
            "frames_world": "z up, pad at the origin, metres, seconds",
            "frames_body": "b1 engines -> nose, b2 starboard, b3 belly; quaternion x, y, z, w",
            **(meta or {}),
        },
        "vehicle": vehicle_geometry(cfg),
        "frames": {k: np.round(v, 6).tolist() for k, v in frames.items()},
    }
    return out


def write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path


def _usd_quat(q) -> str:
    x, y, z, w = q
    return f"({w:.6f}, {x:.6f}, {y:.6f}, {z:.6f})"


def write_usda(data: dict, path: str | Path, segments: int = 48) -> Path:
    """Hand-written USDA: the ship as an Xform with time-sampled translate/orient, a hull mesh,
    hinged flap plates and one Xform per engine nozzle (gimbal applied), 1 frame per time code."""
    v, f = data["vehicle"], data["frames"]
    fps = data["meta"]["fps"]
    n = len(f["time"])
    lines = [
        "#usda 1.0",
        "(",
        f"    startTimeCode = 0\n    endTimeCode = {n - 1}\n    timeCodesPerSecond = {fps}",
        '    upAxis = "Z"\n    metersPerUnit = 1\n    defaultPrim = "Starship"',
        ")",
        "",
        'def Xform "Starship" (kind = "component")',
        "{",
        "    double3 xformOp:translate.timeSamples = {",
    ]
    lines += [
        f"        {i}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})," for i, p in enumerate(f["position"])
    ]
    lines += ["    }", "    quatd xformOp:orient.timeSamples = {"]
    lines += [f"        {i}: {_usd_quat(q)}," for i, q in enumerate(f["quaternion"])]
    lines += [
        "    }",
        '    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
        "",
    ]
    # Hull as a surface of revolution about b1 (local x).
    profile = v["hull_profile"]
    phis = [2 * math.pi * j / segments for j in range(segments)]
    points = [(x, r * math.sin(a), r * math.cos(a)) for x, r in profile for a in phis]
    counts, indices = [], []
    for i in range(len(profile) - 1):
        for j in range(segments):
            a, b = i * segments + j, i * segments + (j + 1) % segments
            counts.append(4)
            indices += [a, a + segments, b + segments, b]
    lines += [
        '    def Mesh "Hull"',
        "    {",
        f"        int[] faceVertexCounts = {counts}",
        f"        int[] faceVertexIndices = {indices}",
        "        point3f[] points = ["
        + ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in points)
        + "]",
        '        uniform token subdivisionScheme = "none"',
        "    }",
    ]
    for k, fl in enumerate(v["flaps"]):
        hinge, axis = fl["hinge"], fl["axis"]
        chord, span = fl["chord"], fl["span"]
        op, fo = np.array(fl["open"]), np.array(fl["folded"])
        # The plate is modelled in a local frame: hinge at the origin, local x along the hinge,
        # local y along the open span. The hinge angle rotates it about local x.
        ex = np.array(axis)
        ey = op
        ez = np.cross(ex, ey)
        # Folding goes towards `folded`; the sign of the rotation about the hinge follows that.
        sign = 1.0 if np.dot(np.cross(ex, ey), fo) > 0 else -1.0
        rows = np.stack([ex, ey, ez], axis=1)  # columns: local axes in body coordinates
        m = [
            (rows[0, 0], rows[1, 0], rows[2, 0], 0.0),
            (rows[0, 1], rows[1, 1], rows[2, 1], 0.0),
            (rows[0, 2], rows[1, 2], rows[2, 2], 0.0),
            (hinge[0], hinge[1], hinge[2], 1.0),
        ]
        mat = ", ".join("(" + ", ".join(f"{c:.6f}" for c in r) + ")" for r in m)
        lines += [
            f'    def Xform "Flap{k}"',
            "    {",
            f"        matrix4d xformOp:transform = ({mat})",
            "        double xformOp:rotateX.timeSamples = {",
        ]
        lines += [
            f"            {i}: {math.degrees(sign * ang[k]):.4f},"
            for i, ang in enumerate(f["flaps"])
        ]
        lines += [
            "        }",
            '        uniform token[] xformOpOrder = ["xformOp:transform", "xformOp:rotateX"]',
            '        def Mesh "Plate"',
            "        {",
            "            int[] faceVertexCounts = [4]",
            "            int[] faceVertexIndices = [0, 1, 2, 3]",
            f"            point3f[] points = [({-chord / 2:.4f}, 0, 0), ({chord / 2:.4f}, 0, 0),"
            f" ({chord / 2:.4f}, {span:.4f}, 0), ({-chord / 2:.4f}, {span:.4f}, 0)]",
            "        }",
            "    }",
        ]
    for k, e in enumerate(v["engines"]):
        ex = e["exit"]
        lines += [
            f'    def Xform "Engine{k}"',
            "    {",
            f"        double3 xformOp:translate = ({ex[0]:.4f}, {ex[1]:.4f}, {ex[2]:.4f})",
            # Thrust tilts towards +b2 (gimbal2) and +b3 (gimbal3): rotate about b3 / -b2.
            "        double3 xformOp:rotateXYZ.timeSamples = {",
        ]
        lines += [
            f"            {i}: (0, {-math.degrees(g3[k]):.4f}, {math.degrees(g2[k]):.4f}),"
            for i, (g2, g3) in enumerate(zip(f["gimbal2"], f["gimbal3"], strict=True))
        ]
        lines += [
            "        }",
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]',
            "        custom double thrust.timeSamples = {",
        ]
        lines += [f"            {i}: {th[k]:.4f}," for i, th in enumerate(f["thrust"])]
        lines += ["        }", "    }"]
    lines += ["}", ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def export_landing(
    policy_path: Path,
    stage: int | None,
    seed: int,
    tries: int = 1,
    fps: float = 60.0,
    device: str = "cpu",
) -> tuple[dict, Samples, int]:
    """Try seeds seed..seed+tries-1 until an episode lands (or return the last one)."""
    cfgs, stage_name = run_configs(policy_path, stage)
    policy = load_policy(policy_path, device)
    samples = None
    for s in range(seed, seed + tries):
        policy.reset()
        samples = record_episode(policy, cfgs, s, device)
        print(f"seed {s}: {STATUS_NAMES[samples.status]} after {samples.time[-1]:.1f} s")
        if samples.landed:
            seed = s
            break
    assert samples is not None
    meta = {"policy": str(policy_path), "stage": stage_name, "seed": seed}
    return build_export(samples, cfgs[0], fps, meta), samples, seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", required=True, help="policy_best.pt inside its run directory")
    parser.add_argument("--stage", type=int, default=None, help="curriculum stage (default: last)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tries", type=int, default=8, help="seeds to try until one lands")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="out/landing.json")
    parser.add_argument("--usd", default=None, help="also write this .usda file")
    args = parser.parse_args()
    data, samples, seed = export_landing(
        Path(args.policy), args.stage, args.seed, args.tries, args.fps, args.device
    )
    if not samples.landed:
        print("warning: no seed landed; exporting the last episode anyway")
    out = write_json(data, args.out)
    m = data["meta"]
    print(f"wrote {out}: {m['frames']} frames, {m['duration']:.2f} s, seed {seed}")
    if args.usd:
        print(f"wrote {write_usda(data, args.usd)}")


if __name__ == "__main__":
    main()
