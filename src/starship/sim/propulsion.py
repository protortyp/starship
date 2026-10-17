"""Engines, flap actuators, roll RCS and propellant draw.

Functions are pure: per-env actuator state goes in and the advanced state comes out. Engine
vectors (wp.vec3) hold one component per engine, in the order of `EngineConfig.azimuths`.

Gimbal sign convention: gimbal2 > 0 tilts the thrust vector of an engine towards +b2. The engines
sit aft of the centre of mass (r ~ -L b1), so the torque r x F ~ -L F tan(g2) b3 is negative about
b3, which swings the nose towards -b2 (and the tail towards +b2). Likewise gimbal3 > 0 gives a
positive torque about b2 and swings the nose towards -b3.
"""

import math

import numpy as np
import warp as wp

from starship.sim.config import G0, VehicleConfig
from starship.sim.types import Wrench, body_point


@wp.struct
class PropulsionParams:
    vacuum_thrust: float
    exit_area: float
    mdot_full: float  # kg/s per engine at full throttle
    min_throttle: float
    throttle_step: float  # max throttle change per step
    spool_up_step: float
    spool_down_step: float
    gimbal_limit: float
    gimbal_step: float  # max gimbal angle change per step
    gimbal_alpha: float  # first-order lag blend per step
    pivot_station: float
    pivot_radius: float
    azimuth: wp.vec3
    flap_min: float
    flap_max: float
    flap_step: float
    flap_alpha: float
    rcs_torque: float  # roll torque at full command, N m
    oxidizer_fraction: float


@wp.struct
class EngineState:
    spool: wp.vec3  # 0..1 fraction of rated thrust reached since ignition
    throttle: wp.vec3
    gimbal2: wp.vec3  # rad, thrust tilt towards +b2
    gimbal3: wp.vec3  # rad, thrust tilt towards +b3
    gimbal2_filtered: wp.vec3
    gimbal3_filtered: wp.vec3


@wp.struct
class FlapState:
    angle: wp.vec4  # rad, order as default_flaps()
    filtered: wp.vec4


def _lag_alpha(dt: float, tau: float) -> float:
    # Exact discretisation of a first-order lag, stable for any dt / tau.
    return 1.0 - math.exp(-dt / tau) if tau > 0.0 else 1.0


def make_propulsion_params(cfg: VehicleConfig, dt: float) -> PropulsionParams:
    e, f = cfg.engine, cfg.flap_actuator
    if e.count != 3 or len(e.azimuths) != 3:
        raise ValueError("propulsion models exactly three gimballed engines")
    p = PropulsionParams()
    p.vacuum_thrust = e.vacuum_thrust
    p.exit_area = e.exit_area
    p.mdot_full = e.vacuum_thrust / (e.vacuum_isp * G0)
    p.min_throttle = e.min_throttle
    p.throttle_step = e.throttle_rate * dt
    p.spool_up_step = dt / e.ignition_delay
    p.spool_down_step = dt / e.shutdown_time
    p.gimbal_limit = e.gimbal_limit
    p.gimbal_step = e.gimbal_rate * dt
    p.gimbal_alpha = _lag_alpha(dt, e.gimbal_lag)
    p.pivot_station = e.pivot_station
    p.pivot_radius = e.pivot_radius
    p.azimuth = wp.vec3(*e.azimuths)
    p.flap_min = f.min_angle
    p.flap_max = f.max_angle
    p.flap_step = f.rate * dt
    p.flap_alpha = _lag_alpha(dt, f.lag)
    p.rcs_torque = cfg.rcs.thrust * cfg.rcs.arm * cfg.rcs.pairs
    p.oxidizer_fraction = e.mixture_ratio / (1.0 + e.mixture_ratio)
    return p


@wp.func
def _actuate(
    position: float, filtered: float, cmd: float, alpha: float, step: float, lo: float, hi: float
) -> wp.vec2:
    """Clamp -> first-order lag -> rate limit -> clamp; returns (filtered, position)."""
    f = filtered + alpha * (wp.clamp(cmd, lo, hi) - filtered)
    x = position + wp.clamp(f - position, -step, step)
    return wp.vec2(f, wp.clamp(x, lo, hi))


@wp.func
def engine_step(
    p: PropulsionParams,
    s: EngineState,
    engine_on: wp.vec3,
    throttle_cmd: wp.vec3,
    gimbal2_cmd: wp.vec3,
    gimbal3_cmd: wp.vec3,
    has_propellant: bool,
) -> EngineState:
    out = EngineState()
    out.spool = s.spool
    out.throttle = s.throttle
    out.gimbal2 = s.gimbal2
    out.gimbal3 = s.gimbal3
    out.gimbal2_filtered = s.gimbal2_filtered
    out.gimbal3_filtered = s.gimbal3_filtered
    lim = p.gimbal_limit
    for i in range(3):
        spool = s.spool[i] - p.spool_down_step
        if engine_on[i] > 0.5 and has_propellant:
            spool = s.spool[i] + p.spool_up_step
        out.spool[i] = wp.clamp(spool, 0.0, 1.0)
        target = wp.clamp(throttle_cmd[i], p.min_throttle, 1.0)
        out.throttle[i] = s.throttle[i] + wp.clamp(
            target - s.throttle[i], -p.throttle_step, p.throttle_step
        )
        g2 = _actuate(
            s.gimbal2[i],
            s.gimbal2_filtered[i],
            gimbal2_cmd[i],
            p.gimbal_alpha,
            p.gimbal_step,
            -lim,
            lim,
        )
        g3 = _actuate(
            s.gimbal3[i],
            s.gimbal3_filtered[i],
            gimbal3_cmd[i],
            p.gimbal_alpha,
            p.gimbal_step,
            -lim,
            lim,
        )
        out.gimbal2_filtered[i] = g2[0]
        out.gimbal2[i] = g2[1]
        out.gimbal3_filtered[i] = g3[0]
        out.gimbal3[i] = g3[1]
    return out


@wp.func
def engine_wrench(p: PropulsionParams, s: EngineState, pressure: float, com: float) -> Wrench:
    """Thrust force and torque about the centre of mass; ambient pressure (Pa) reduces thrust."""
    force = wp.vec3(0.0)
    torque = wp.vec3(0.0)
    for i in range(3):
        thrust = wp.max(
            s.spool[i] * (s.throttle[i] * p.vacuum_thrust - pressure * p.exit_area), 0.0
        )
        d = wp.normalize(wp.vec3(1.0, wp.tan(s.gimbal2[i]), wp.tan(s.gimbal3[i])))
        f = thrust * d
        force += f
        torque += wp.cross(body_point(p.pivot_station, p.pivot_radius, p.azimuth[i], com), f)
    out = Wrench()
    out.force = force
    out.torque = torque
    return out


@wp.func
def engine_mass_flow(p: PropulsionParams, s: EngineState) -> float:
    return p.mdot_full * wp.dot(s.spool, s.throttle)


@wp.func
def flap_step(p: PropulsionParams, s: FlapState, cmd: wp.vec4) -> FlapState:
    out = FlapState()
    out.angle = s.angle
    out.filtered = s.filtered
    for i in range(4):
        a = _actuate(
            s.angle[i], s.filtered[i], cmd[i], p.flap_alpha, p.flap_step, p.flap_min, p.flap_max
        )
        out.filtered[i] = a[0]
        out.angle[i] = a[1]
    return out


@wp.func
def rcs_wrench(p: PropulsionParams, cmd: float) -> Wrench:
    """Roll torque about b1; the thruster pairs are balanced, so no net force."""
    out = Wrench()
    out.force = wp.vec3(0.0)
    out.torque = wp.vec3(wp.clamp(cmd, -1.0, 1.0) * p.rcs_torque, 0.0, 0.0)
    return out


@wp.func
def _take(available: float, demand: float) -> float:
    return wp.min(wp.max(available, 0.0), demand)


@wp.func
def draw_propellant(p: PropulsionParams, tanks: wp.vec4, mdot: float, dt: float) -> wp.vec4:
    """Remove mdot * dt from (main_lox, main_ch4, header_lox, header_ch4); the header tanks feed
    first, any shortfall comes from the main tanks."""
    burned = wp.max(mdot, 0.0) * dt
    ox = burned * p.oxidizer_fraction
    fuel = burned - ox
    ox_header = _take(tanks[2], ox)
    fuel_header = _take(tanks[3], fuel)
    ox_main = _take(tanks[0], ox - ox_header)
    fuel_main = _take(tanks[1], fuel - fuel_header)
    return wp.vec4(
        tanks[0] - ox_main, tanks[1] - fuel_main, tanks[2] - ox_header, tanks[3] - fuel_header
    )


# Host helpers for tests: single-env rollouts and evaluations.


@wp.kernel
def _engine_rollout_kernel(
    p: PropulsionParams,
    s0: EngineState,
    cmd: wp.array(dtype=wp.vec3),  # engine_on, throttle, gimbal2, gimbal3
    has_propellant: bool,
    out: wp.array(dtype=EngineState),
):
    s = s0
    for k in range(out.shape[0]):
        s = engine_step(p, s, cmd[0], cmd[1], cmd[2], cmd[3], has_propellant)
        out[k] = s


@wp.kernel
def _engine_eval_kernel(
    p: PropulsionParams,
    s: EngineState,
    pressure: float,
    com: float,
    out: wp.array(dtype=wp.vec3),
):
    w = engine_wrench(p, s, pressure, com)
    out[0] = w.force
    out[1] = w.torque
    out[2] = wp.vec3(engine_mass_flow(p, s), 0.0, 0.0)


@wp.kernel
def _flap_rollout_kernel(
    p: PropulsionParams, s0: FlapState, cmd: wp.vec4, out: wp.array(dtype=FlapState)
):
    s = s0
    for k in range(out.shape[0]):
        s = flap_step(p, s, cmd)
        out[k] = s


@wp.kernel
def _misc_kernel(
    p: PropulsionParams,
    rcs_cmd: float,
    tanks: wp.vec4,
    mdot: float,
    dt: float,
    torque: wp.array(dtype=wp.vec3),
    tanks_out: wp.array(dtype=wp.vec4),
):
    torque[0] = rcs_wrench(p, rcs_cmd).torque
    tanks_out[0] = draw_propellant(p, tanks, mdot, dt)


def engine_state(spool=0.0, throttle=1.0, gimbal2=0.0, gimbal3=0.0) -> EngineState:
    """Engine state with the same values on all three engines; the lag filters start settled."""
    s = EngineState()
    s.spool = wp.vec3(spool)
    s.throttle = wp.vec3(throttle)
    s.gimbal2 = s.gimbal2_filtered = wp.vec3(gimbal2)
    s.gimbal3 = s.gimbal3_filtered = wp.vec3(gimbal3)
    return s


def engine_rollout(
    p,
    s0,
    engine_on,
    throttle_cmd,
    gimbal2_cmd,
    gimbal3_cmd,
    steps,
    has_propellant=True,
    device=None,
):
    """Run engine_step for `steps` steps with constant per-engine commands; returns a dict of
    (steps, 3) arrays, one per EngineState field."""
    cmd = wp.array(
        [
            wp.vec3(*np.broadcast_to(c, 3))
            for c in (engine_on, throttle_cmd, gimbal2_cmd, gimbal3_cmd)
        ],
        dtype=wp.vec3,
        device=device,
    )
    out = wp.empty(steps, dtype=EngineState, device=device)
    wp.launch(
        _engine_rollout_kernel, dim=1, inputs=[p, s0, cmd, has_propellant, out], device=device
    )
    arr = out.numpy()
    return {name: arr[name] for name in arr.dtype.names}


def engine_eval(p, s, pressure, com, device=None):
    """Returns (force, torque, mass flow) for one engine state."""
    out = wp.zeros(3, dtype=wp.vec3, device=device)
    wp.launch(_engine_eval_kernel, dim=1, inputs=[p, s, pressure, com, out], device=device)
    force, torque, mdot = out.numpy()
    return force, torque, float(mdot[0])


def flap_rollout(p, angle, cmd, steps, device=None):
    """Run flap_step from a settled state at `angle`; returns (steps, 4) angles and filters."""
    s0 = FlapState()
    s0.angle = s0.filtered = wp.vec4(*np.broadcast_to(angle, 4))
    out = wp.empty(steps, dtype=FlapState, device=device)
    wp.launch(
        _flap_rollout_kernel,
        dim=1,
        inputs=[p, s0, wp.vec4(*np.broadcast_to(cmd, 4)), out],
        device=device,
    )
    arr = out.numpy()
    return arr["angle"], arr["filtered"]


def rcs_and_draw(p, rcs_cmd, tanks, mdot, dt, device=None):
    """Returns (rcs torque, tanks after draw_propellant)."""
    torque = wp.zeros(1, dtype=wp.vec3, device=device)
    tanks_out = wp.zeros(1, dtype=wp.vec4, device=device)
    wp.launch(
        _misc_kernel,
        dim=1,
        inputs=[p, rcs_cmd, wp.vec4(*tanks), mdot, dt, torque, tanks_out],
        device=device,
    )
    return torque.numpy()[0], tanks_out.numpy()[0]
