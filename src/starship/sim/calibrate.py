"""Fit a held belly-flop to OCR telemetry with batched cross-entropy search.

Example: python -m starship.sim.calibrate --flight IFT-5 --data-dir /path/to/telemetry
Only the effective ballistic coefficient is strongly constrained without attitude/flap data.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from starship.sim.config import SimConfig
from starship.sim.physics import Commands, Params, State, advance, attitude, spawn
from starship.sim.simulator import Simulator, make_params
from starship.sim.telemetry import Telemetry, load_telemetry

DEFAULT_BOUNDS = np.array([[0.5, 1.8], [0.5, 1.5], [100_000, 200_000], [40, 90]], dtype=float)


@dataclass
class FitParameters:
    crossflow_drag_scale: float = 1.0
    flap_force_scale: float = 1.0
    dry_mass: float = 160_000.0
    descent_angle_deg: float = 70.0  # below horizontal, velocity aligned with the nose heading


@wp.kernel
def _rollout(
    params: wp.array(dtype=Params),
    commands: Commands,
    state: State,
    controls_per_sample: int,
    samples: int,
    result: wp.array2d(dtype=wp.vec2),
):
    i = wp.tid()
    p = params[i]
    spawn(p, state, i)
    b = state.body[i]
    # Common heading and drift remove the randomized training spawn as a fitting nuisance.
    b.rot = attitude(wp.pi / 2.0, 0.0)
    b.vel = wp.vec3(p.spawn.horizontal_speed, 0.0, p.spawn.vertical_speed)
    state.body[i] = b
    result[i, 0] = wp.vec2(b.pos[2], wp.length(b.vel))
    for j in range(1, samples):
        for _ in range(controls_per_sample):
            if state.status[i] == 0:
                b = state.body[i]
                axis_z = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))[2]
                u = wp.clamp((wp.pi / 3.0) * (axis_z + b.omega[1]), -0.7, 0.7)
                aft = 52.0 * wp.pi / 180.0 - u
                fwd = 20.0 * wp.pi / 180.0 + u
                commands.flaps[i] = wp.vec4(aft, aft, fwd, fwd)
                advance(p, state, commands, i)
        b = state.body[i]
        result[i, j] = wp.vec2(b.pos[2], wp.length(b.vel))


def simulation_config(data: Telemetry, candidate: FitParameters, dt: float = 0.02) -> SimConfig:
    cfg = SimConfig(dt=dt, substeps=round(0.1 / dt))
    cfg.max_time = float(data.t[-1] + 5)
    cfg.max_offset = 100_000.0
    cfg.spawn.altitude = float(data.altitude_m[0])
    angle = math.radians(candidate.descent_angle_deg)
    cfg.spawn.vertical_speed = -float(data.speed_mps[0]) * math.sin(angle)
    cfg.spawn.horizontal_speed = float(data.speed_mps[0]) * math.cos(angle)
    cfg.spawn.main_fill = 0.0
    cfg.vehicle.aero.crossflow_drag_scale = candidate.crossflow_drag_scale
    cfg.vehicle.aero.flap_force_scale = candidate.flap_force_scale
    cfg.vehicle.mass.dry_mass = candidate.dry_mass
    return cfg


def simulate(
    data: Telemetry, candidates: np.ndarray, *, device: str = "cpu", dt: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Return (candidate, telemetry sample, altitude/speed) and completion mask.

    Each Warp thread runs one candidate through the production `advance` function. Parameters
    retain their host owners until the launch completes because they contain device pointers.
    """
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    if candidates.shape[1] != 4 or not np.isfinite(candidates).all():
        raise ValueError("candidates must have four finite parameters")
    if dt <= 0 or not np.isclose(0.1 / dt, round(0.1 / dt)):
        raise ValueError("dt must divide the 0.1 second control interval")
    if len(data.t) < 2 or data.t[0] != 0 or np.any(np.diff(data.t) <= 0):
        raise ValueError("telemetry time must start at zero and strictly increase")
    configs = [simulation_config(data, FitParameters(*row), dt) for row in candidates]
    sim = Simulator(len(candidates), configs[0], device=device)
    owners = [make_params(cfg, device) for cfg in configs]
    params = wp.array(owners, dtype=Params, device=device)
    clock = np.arange(math.ceil(data.t[-1]) + 1, dtype=float)
    result = wp.empty((len(candidates), len(clock)), dtype=wp.vec2, device=device)
    wp.launch(
        _rollout,
        dim=len(candidates),
        inputs=[params, sim.commands, sim.state, 10, len(clock), result],
        device=device,
    )
    host = result.numpy()
    prediction = np.empty((len(candidates), len(data.t), 2))
    for i in range(len(candidates)):
        for channel in range(2):
            prediction[i, :, channel] = np.interp(data.t, clock, host[i, :, channel])
    # A frozen crashed/tumbled trajectory must never be rewarded for matching the last samples.
    end_time = sim.state.time.numpy()
    terminated = sim.state.status.numpy() != 0
    complete = (~terminated | (end_time >= data.t[-1])) & np.isfinite(prediction).all(axis=(1, 2))
    prediction[terminated[:, None] & (data.t[None, :] > end_time[:, None])] = np.nan
    return prediction, complete


def rmse(data: Telemetry, prediction: np.ndarray) -> np.ndarray:
    """Altitude interval RMSE (m), speed RMSE (m/s); repaired channels do not contribute."""
    prediction = np.asarray(prediction)
    h = np.maximum(abs(prediction[..., 0] - data.altitude_m) - data.altitude_half_width, 0)
    v = prediction[..., 1] - data.speed_mps
    valid = ~data.repaired
    if np.any(valid.sum(axis=0) < 2):
        raise ValueError("each telemetry channel needs two unrepaired samples")
    return np.stack(
        [
            np.sqrt(np.mean(h[..., valid[:, 0]] ** 2, axis=-1)),
            np.sqrt(np.mean(v[..., valid[:, 1]] ** 2, axis=-1)),
        ],
        axis=-1,
    )


def fit(
    data: Telemetry,
    *,
    device: str = "cpu",
    population: int = 48,
    iterations: int = 10,
    seed: int = 7,
    bounds: np.ndarray | None = None,
    dt: float = 0.02,
    verbose: bool = False,
) -> tuple[FitParameters, np.ndarray]:
    """Bounded Gaussian CEM, with elitism; equal bounds fix an unidentifiable parameter.

    The objective weights 500 m of interval error like 10 m/s of speed error. These are explicit
    fitting scales, not estimates of independent OCR noise (adjacent frames are correlated).
    """
    bounds = np.array(DEFAULT_BOUNDS if bounds is None else bounds, dtype=float)
    if (
        bounds.shape != (4, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 1] < bounds[:, 0])
    ):
        raise ValueError("bounds must be four finite (lower, upper) pairs")
    if population < 4 or iterations < 1:
        raise ValueError("need population >= 4 and iterations >= 1")
    rng = np.random.default_rng(seed)
    mean, std = np.full(4, 0.5), np.full(4, 0.3)
    best, best_loss, best_errors = mean.copy(), float("inf"), None
    for iteration in range(iterations):
        unit = np.clip(rng.normal(mean, std, (population, 4)), 0, 1)
        unit[0] = best
        candidates = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
        predictions, complete = simulate(data, candidates, device=device, dt=dt)
        errors = rmse(data, predictions)
        loss = np.sum((errors / [500, 10]) ** 2, axis=1)
        loss[~complete | ~np.isfinite(loss)] = np.inf
        order = np.argsort(loss)
        if loss[order[0]] < best_loss:
            best_loss = float(loss[order[0]])
            best = unit[order[0]].copy()
            best_errors = errors[order[0]].copy()
        finite = order[np.isfinite(loss[order])]
        if len(finite) < 2:
            if verbose:
                print(f"iteration {iteration + 1}: fewer than two completed candidates", flush=True)
            continue
        elite = unit[finite[: max(2, population // 5)]]
        mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
        std = np.maximum(0.25 * std + 0.75 * elite.std(axis=0), 0.015)
        if verbose:
            print(
                f"iteration {iteration + 1}: altitude={best_errors[0]:.1f} m, "
                f"speed={best_errors[1]:.3f} m/s ({complete.sum()}/{population} completed)",
                flush=True,
            )
    if best_errors is None:
        raise RuntimeError("no candidate completed the descent window")
    return FitParameters(*(bounds[:, 0] + best * (bounds[:, 1] - bounds[:, 0]))), best_errors


def write_comparison(path: Path, data: Telemetry, prediction: np.ndarray) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "frame_time_s",
                "altitude_m",
                "speed_mps",
                "sim_altitude_m",
                "sim_speed_mps",
                "altitude_repaired",
                "speed_repaired",
                "sim_valid",
            ]
        )
        for i, t in enumerate(data.t):
            writer.writerow(
                [
                    t,
                    t + data.start_time,
                    data.altitude_m[i],
                    data.speed_mps[i],
                    *prediction[i],
                    *data.repaired[i].astype(int),
                    int(np.isfinite(prediction[i]).all()),
                ]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flight", choices=["IFT-4", "IFT-5", "IFT-6"], default="IFT-5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("telemetry"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()
    datasets = {
        flight: load_telemetry(args.data_dir / f"{flight}_Telemetry.csv", corrected=not args.raw)
        for flight in ("IFT-4", "IFT-5", "IFT-6")
    }
    started = perf_counter()
    fitted, _ = fit(
        datasets[args.flight],
        device=args.device,
        population=args.population,
        iterations=args.iterations,
        seed=args.seed,
        dt=args.dt,
        verbose=True,
    )
    fit_seconds = perf_counter() - started
    print(json.dumps(asdict(fitted), indent=2), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "fit_flight": args.flight,
        "fit_seconds": fit_seconds,
        "parameters": asdict(fitted),
        "flights": {},
        "seed": args.seed,
        "population": args.population,
        "iterations": args.iterations,
        "dt": args.dt,
        "corrected": not args.raw,
        "bounds": DEFAULT_BOUNDS.tolist(),
        "altitude_half_width_m": 500,
        "header_propellant_kg": 23000,
        "main_propellant_kg": 0,
    }
    for flight, data in datasets.items():
        prediction, complete = simulate(
            data, [list(asdict(fitted).values())], device=args.device, dt=args.dt
        )
        valid = np.isfinite(prediction[0]).all(axis=1)
        scored = replace(
            data,
            t=data.t[valid],
            altitude_m=data.altitude_m[valid],
            speed_mps=data.speed_mps[valid],
            repaired=data.repaired[valid],
        )
        errors = rmse(scored, prediction[:, valid])[0]
        scope = "full_window" if complete[0] else "valid_prefix_only"
        report["flights"][flight] = {
            "altitude_rmse_m": float(errors[0]),
            "speed_rmse_mps": float(errors[1]),
            "completed": bool(complete[0]),
            "rmse_scope": scope,
            "valid_duration_s": float(scored.t[-1]),
            "start_frame_time_s": data.start_time,
            "duration_s": float(data.t[-1]),
            "burn_frame_time_s": data.burn_time,
            "repaired_fraction": data.repaired.mean(axis=0).tolist(),
        }
        write_comparison(args.out / f"calibration_{args.flight}_{flight}.csv", data, prediction[0])
        print(
            f"{flight}: altitude interval RMSE={errors[0]:.1f} m, speed RMSE={errors[1]:.3f} m/s; "
            f"completed={complete[0]}, scope={scope}",
            flush=True,
        )
    (args.out / f"calibration_{args.flight}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
