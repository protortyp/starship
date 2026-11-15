"""Batched 6-DOF Starship simulator on a Warp device."""

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from starship.sim.aero import make_aero_params
from starship.sim.catch import CatchParams
from starship.sim.config import SimConfig
from starship.sim.hull import make_hull_params
from starship.sim.mass import make_mass_params
from starship.sim.physics import (
    Commands,
    LandingParams,
    Params,
    State,
    Terminal,
    allocate,
    gather_kernel,
    init_rng_kernel,
    make_spawn_params,
    reset_kernel,
    step_kernel,
)
from starship.sim.propulsion import make_propulsion_params
from starship.sim.wind import init_wind_kernel, make_wind_params


def make_params(config: SimConfig, device) -> Params:
    c = config.catch
    p_c = CatchParams()
    for name in c.__dataclass_fields__:
        value = getattr(c, name)
        if not math.isfinite(value):
            raise ValueError(f"catch {name} must be finite")
        setattr(
            p_c,
            name,
            int(value)
            if name in ("enabled", "moving", "policy_controlled", "flap_collisions")
            else value,
        )
    if c.enabled and not (
        0 < c.station < config.vehicle.hull.length
        and c.height > c.station
        and 0 < c.tolerance <= c.arm_half_width
        and c.height_tolerance > 0
        and c.pin_radius > config.vehicle.hull.diameter / 2 + c.arm_half_width
        and c.arm_half_length > abs(c.arm_center_x) + c.tolerance
        and c.tower_half_width > 0
        and c.tower_height > c.height
        and c.tower_x + c.tower_half_width < -config.vehicle.hull.diameter / 2
    ):
        raise ValueError("invalid catch geometry or capture tolerances")
    if c.enabled and (
        not 0 < c.open_angle < math.pi / 2
        or c.angular_speed <= 0
        or not 0 < c.close_max_tilt < math.pi / 2
        or c.close_max_offset <= 0
        or c.close_clearance < 0
        or c.flap_half_thickness <= 0
        or (c.moving and not c.flap_collisions)
    ):
        raise ValueError("moving catch requires valid motion limits and flap collisions")
    if c.policy_controlled and not (c.enabled and c.moving and c.flap_collisions):
        raise ValueError("policy arm control requires moving catch and flap collisions")
    v = config.vehicle
    p = Params()
    p.catcher = p_c
    p.dt = config.dt
    p.substeps = config.substeps
    p.max_steps = max(1, round(config.max_time / (config.dt * config.substeps)))
    p.max_altitude = config.max_altitude
    p.max_offset = config.max_offset
    p.max_pitch = config.max_pitch
    p.max_rate = config.max_rate
    p.max_roll_rate = config.max_roll_rate
    p.terminate_on_attitude_limits = int(config.terminate_on_attitude_limits)
    p.hull = make_hull_params(v.hull)
    for name in LandingParams.vars:
        setattr(p.landing, name, getattr(config.landing, name))
    p.spawn = make_spawn_params(v, config.spawn)
    p.mass = make_mass_params(v)
    p.propulsion = make_propulsion_params(v, config.dt)
    p.aero = make_aero_params(v, device=device)
    p.wind = make_wind_params(config.wind, v.hull.diameter)
    return p


@dataclass
class Snapshot:
    """Host copy of the per-env state."""

    pos: np.ndarray  # (N, 3) centre of mass
    vel: np.ndarray
    rot: np.ndarray  # (N, 4) quaternion x, y, z, w, body to world
    omega: np.ndarray  # (N, 3) body frame
    tanks: np.ndarray  # (N, 4) in TANK_ORDER
    spool: np.ndarray  # (N, 3)
    flaps: np.ndarray  # (N, 4) rad
    time: np.ndarray
    steps: np.ndarray
    status: np.ndarray
    catch_failure: np.ndarray
    arm_angle: np.ndarray  # symmetric arm opening angle, radians
    wind: np.ndarray  # (N, 3) air velocity at the vehicle, world


@dataclass
class TerminalSnapshot:
    catch_failure: np.ndarray
    arm_angle: np.ndarray
    flaps: np.ndarray
    spool: np.ndarray
    done: np.ndarray
    status: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    rot: np.ndarray
    omega: np.ndarray
    tanks: np.ndarray
    time: np.ndarray
    steps: np.ndarray


def _bodies(arr: wp.array) -> dict[str, np.ndarray]:
    host = arr.numpy()
    return {k: np.array(host[k]) for k in ("pos", "vel", "rot", "omega")}


class Simulator:
    """N independent rockets. Commands are per-env arrays (fields of `Commands`), held for one
    control step of `substeps` physics steps.

    When an episode ends, the step that ended it sets `terminal.done` and records the final state
    in `terminal`. With `auto_reset`, the env then respawns within the same kernel launch.
    """

    def __init__(
        self,
        num_envs: int,
        config: SimConfig | None = None,
        device: str | None = None,
        seed: int = 0,
        auto_reset: bool = False,
    ):
        self.num_envs = num_envs
        self.device = wp.get_device(device)
        self.auto_reset = auto_reset
        self.config = config or SimConfig()
        self.params = make_params(self.config, self.device)

        self.state = allocate(State, num_envs, self.device)
        self.terminal = allocate(Terminal, num_envs, self.device)
        self.commands = allocate(Commands, num_envs, self.device)
        wp.launch(init_rng_kernel, dim=num_envs, inputs=[seed, self.state.rng], device=self.device)
        wp.launch(
            init_wind_kernel,
            dim=num_envs,
            inputs=[seed ^ 0x57494E44, self.state.wind],
            device=self.device,
        )
        self.reset()

    def set_config(self, config: SimConfig) -> None:
        """Takes effect for later steps and respawns; running episodes continue."""
        self.config = config
        self.params = make_params(config, self.device)

    def reset(self, mask: wp.array | np.ndarray | None = None) -> None:
        if mask is None:
            mask = np.ones(self.num_envs, dtype=np.int32)
        if isinstance(mask, np.ndarray):
            mask = wp.array(mask.astype(np.int32), dtype=wp.int32, device=self.device)
        wp.launch(
            reset_kernel,
            dim=self.num_envs,
            inputs=[self.params, mask, self.state],
            device=self.device,
        )

    def copy_from(self, other: "Simulator", sources) -> None:
        """Set env i to the state of `other`'s env sources[i] (other may be this simulator).
        Random streams stay per env, so copies of one state diverge under noise."""
        if not isinstance(sources, wp.array):
            sources = wp.array(
                np.asarray(sources, dtype=np.int32), dtype=wp.int32, device=self.device
            )
        src = other.state
        if other is self:
            src = State()
            for name in State.vars:
                setattr(src, name, wp.clone(getattr(self.state, name)))
        wp.launch(
            gather_kernel, dim=self.num_envs, inputs=[sources, src, self.state], device=self.device
        )

    def set_commands(self, **commands) -> None:
        """Copy per-env commands, e.g. set_commands(engine_on=(N, 3) array, flaps=(N, 4) array).
        Scalars or single rows broadcast to all envs; omitted fields keep their values."""
        for name, value in commands.items():
            target = getattr(self.commands, name)
            if isinstance(value, wp.array):
                target.assign(value)
                continue
            width = getattr(target.dtype, "_length_", 1)
            shape = (self.num_envs,) if width == 1 else (self.num_envs, width)
            host = np.broadcast_to(np.asarray(value, dtype=np.float32), shape)
            target.assign(
                wp.array(np.ascontiguousarray(host), dtype=target.dtype, device=self.device)
            )

    def step(self, **commands) -> None:
        if commands:
            self.set_commands(**commands)
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[self.params, int(self.auto_reset), self.commands, self.state, self.terminal],
            device=self.device,
        )

    def snapshot(self) -> Snapshot:
        s = self.state
        engine, flaps = s.engine.numpy(), s.flaps.numpy()
        return Snapshot(
            **_bodies(s.body),
            tanks=s.tanks.numpy().copy(),
            spool=np.array(engine["spool"]),
            flaps=np.array(flaps["angle"]),
            time=s.time.numpy().copy(),
            steps=s.steps.numpy().copy(),
            status=s.status.numpy().copy(),
            wind=np.array(s.wind.numpy()["wind"]),
            catch_failure=s.catch_failure.numpy().copy(),
            arm_angle=s.arm_angle.numpy().copy(),
        )

    def terminal_snapshot(self) -> TerminalSnapshot:
        t = self.terminal
        return TerminalSnapshot(
            catch_failure=t.catch_failure.numpy().copy(),
            arm_angle=t.arm_angle.numpy().copy(),
            flaps=np.array(t.flaps.numpy()["angle"]),
            spool=np.array(t.engine.numpy()["spool"]),
            done=t.done.numpy().copy(),
            status=t.status.numpy().copy(),
            **_bodies(t.body),
            tanks=t.tanks.numpy().copy(),
            time=t.time.numpy().copy(),
            steps=t.steps.numpy().copy(),
        )
