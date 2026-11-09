"""Batched 6-DOF step: actuators, mass, aerodynamics, engines and ground contact per substep.

Each env runs `substeps` physics steps per control step with its commands held. An episode ends
when the hull touches the ground (landed or crashed), leaves the bounds, or times out; with
auto-reset the final state goes to the terminal buffers and the env respawns in the same launch.
Aerodynamics act on the air-relative velocity (`starship.sim.wind`); ground speed is the state.
"""

import warp as wp

from starship.sim.aero import AeroParams, aero_wrench
from starship.sim.atmosphere import atmosphere
from starship.sim.config import G0
from starship.sim.flight_stats import FlightStats, track_flight, track_motion
from starship.sim.hull import HullParams, lowest_point
from starship.sim.mass import MassParams, mass_properties
from starship.sim.propulsion import (
    EngineState,
    FlapState,
    PropulsionParams,
    burn_fraction,
    draw_propellant,
    engine_mass_flow,
    engine_step,
    engine_wrench,
    flap_step,
    rcs_wrench,
)
from starship.sim.rigid import Body, gravity, integrate, orientation
from starship.sim.wind import (
    WindParams,
    WindState,
    flap_gain,
    sample_wind,
    unsteady_step,
    wind_step,
)
from starship.sim.wrench import add_wrench

FLYING = 0
LANDED = 1
CRASHED = 2
OUT_OF_BOUNDS = 3
TIMEOUT = 4
TUMBLED = 5


@wp.struct
class SpawnParams:
    altitude: float
    altitude_noise: float
    offset: float
    offset_noise: float
    vertical_speed: float
    vertical_speed_noise: float
    horizontal_speed: float
    horizontal_speed_noise: float
    entry: int
    speed: float
    speed_noise: float
    flight_path_angle: float
    flight_path_angle_noise: float
    heading: float
    heading_noise: float
    entry_drag_area: float
    pitch: float
    pitch_noise: float
    angular_rate_noise: float
    flap_angle: float
    lit_throttle: float
    tanks: wp.vec4


@wp.struct
class LandingParams:
    pad_radius: float
    max_vertical_speed: float
    max_horizontal_speed: float
    max_tilt: float
    max_angular_rate: float


@wp.struct
class Params:
    dt: float
    substeps: int
    max_steps: int
    max_altitude: float
    max_offset: float
    max_pitch: float  # from vertical; beyond it the vehicle counts as tumbling
    max_rate: float
    max_roll_rate: float
    hull: HullParams
    landing: LandingParams
    spawn: SpawnParams
    mass: MassParams
    propulsion: PropulsionParams
    aero: AeroParams
    wind: WindParams


@wp.struct
class Commands:
    engine_on: wp.array(dtype=wp.vec3)
    throttle: wp.array(dtype=wp.vec3)
    gimbal2: wp.array(dtype=wp.vec3)
    gimbal3: wp.array(dtype=wp.vec3)
    flaps: wp.array(dtype=wp.vec4)
    rcs: wp.array(dtype=float)


@wp.struct
class State:
    body: wp.array(dtype=Body)
    stats: wp.array(dtype=FlightStats)
    engine: wp.array(dtype=EngineState)
    flaps: wp.array(dtype=FlapState)
    tanks: wp.array(dtype=wp.vec4)
    time: wp.array(dtype=float)
    steps: wp.array(dtype=wp.int32)
    status: wp.array(dtype=wp.int32)
    # Per-env random stream, advanced on every respawn.
    rng: wp.array(dtype=wp.uint32)
    # Wind sample and filter states, with their own stream inside.
    wind: wp.array(dtype=WindState)


@wp.struct
class Terminal:
    """Final state of the episode that ended on the last step (valid where done is set)."""

    done: wp.array(dtype=wp.int32)
    status: wp.array(dtype=wp.int32)
    body: wp.array(dtype=Body)
    stats: wp.array(dtype=FlightStats)
    tanks: wp.array(dtype=wp.vec4)
    time: wp.array(dtype=float)
    steps: wp.array(dtype=wp.int32)
    wind: wp.array(dtype=WindState)


@wp.func
def entry_range(
    altitude: float, speed: float, angle: float, mass: float, drag_area: float
) -> float:
    """Point-mass, no-thrust footprint to the 300 m flip corridor, with quadratic drag."""
    z = altitude
    vx = speed * wp.cos(angle)
    vz = -speed * wp.sin(angle)
    x = float(0.0)  # noqa: UP018 - Warp requires a dynamic loop accumulator.
    for _ in range(480):
        if z <= 300.0:
            break
        drag = 0.5 * atmosphere(z).density * drag_area * wp.length(wp.vec2(vx, vz)) / mass
        dt = wp.min(0.5, (z - 300.0) / wp.max(-vz, 1.0))
        # Semi-implicit drag avoids a velocity reversal at large density or speed.
        vx = vx / (1.0 + drag * dt)
        vz = (vz - gravity(z) * dt) / (1.0 + drag * dt)
        x += vx * dt
        z += vz * dt
    return x


@wp.func
def spawn(p: Params, s: State, tid: int):
    sp = p.spawn
    rng = s.rng[tid]
    alt = sp.altitude + sp.altitude_noise * wp.randf(rng, -1.0, 1.0)
    offset = sp.offset + sp.offset_noise * wp.randf(rng, -1.0, 1.0)
    bearing = wp.randf(rng, 0.0, 2.0 * wp.pi)
    speed = sp.horizontal_speed + sp.horizontal_speed_noise * wp.randf(rng, -1.0, 1.0)
    drift = wp.randf(rng, 0.0, 2.0 * wp.pi)
    vz = sp.vertical_speed + sp.vertical_speed_noise * wp.randf(rng, -1.0, 1.0)
    pitch = sp.pitch + sp.pitch_noise * wp.randf(rng, -1.0, 1.0)
    heading = wp.randf(rng, 0.0, 2.0 * wp.pi)
    w = sp.angular_rate_noise
    b = Body()
    b.pos = wp.vec3(offset * wp.cos(bearing), offset * wp.sin(bearing), alt)
    b.vel = wp.vec3(speed * wp.cos(drift), speed * wp.sin(drift), vz)
    b.rot = orientation(pitch, heading)
    b.omega = wp.vec3(wp.randf(rng, -w, w), wp.randf(rng, -w, w), wp.randf(rng, -w, w))
    if sp.entry != 0:
        speed = sp.speed + sp.speed_noise * wp.randf(rng, -1.0, 1.0)
        angle = sp.flight_path_angle + sp.flight_path_angle_noise * wp.randf(rng, -1.0, 1.0)
        heading = bearing + sp.heading + sp.heading_noise * wp.randf(rng, -1.0, 1.0)
        reach = entry_range(
            alt, speed, angle, mass_properties(p.mass, sp.tanks).mass, sp.entry_drag_area
        )
        b.pos = wp.vec3(
            -reach * wp.cos(bearing) - offset * wp.sin(bearing),
            -reach * wp.sin(bearing) + offset * wp.cos(bearing),
            alt,
        )
        b.vel = speed * wp.vec3(
            wp.cos(angle) * wp.cos(heading), wp.cos(angle) * wp.sin(heading), -wp.sin(angle)
        )
        # In still air, pitch=90 degrees means b3 along the flow, b1 at 90-degree AoA.
        b.rot = orientation(angle + pitch - 0.5 * wp.pi, heading)
    s.body[tid] = b
    a = track_motion(FlightStats(), b, wp.vec3(0.0), 0.0)
    a.flip_altitude = -1.0
    a.ignition_altitude = -1.0
    if sp.lit_throttle > 0.0:
        a.ignition_altitude = b.pos[2]
    a.start_offset = wp.length(wp.vec2(b.pos[0], b.pos[1]))
    s.stats[tid] = a

    e = EngineState()
    e.throttle = wp.vec3(p.propulsion.min_throttle)
    if sp.lit_throttle > 0.0:
        e.spool = wp.vec3(1.0)
        e.throttle = wp.vec3(sp.lit_throttle)
    s.engine[tid] = e
    f = FlapState()
    f.angle = wp.vec4(sp.flap_angle)
    f.filtered = f.angle
    s.flaps[tid] = f
    s.tanks[tid] = sp.tanks
    s.time[tid] = 0.0
    s.steps[tid] = 0
    s.status[tid] = FLYING
    s.rng[tid] = rng
    s.wind[tid] = sample_wind(p.wind, s.wind[tid].rng, alt)


@wp.func
def touchdown_status(ld: LandingParams, b: Body, com: float) -> int:
    axis = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))
    base = b.pos - axis * com
    horizontal_speed = wp.sqrt(b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1])
    miss = wp.sqrt(base[0] * base[0] + base[1] * base[1])
    tilt = wp.acos(wp.clamp(axis[2], -1.0, 1.0))
    rate = wp.length(b.omega)
    if (
        -b.vel[2] <= ld.max_vertical_speed
        and horizontal_speed <= ld.max_horizontal_speed
        and tilt <= ld.max_tilt
        and rate <= ld.max_angular_rate
        and miss <= ld.pad_radius
    ):
        return LANDED
    return CRASHED


@wp.func
def advance(p: Params, s: State, c: Commands, tid: int):
    b = s.body[tid]
    a = s.stats[tid]
    e = s.engine[tid]
    f = s.flaps[tid]
    tanks = s.tanks[tid]
    ws = s.wind[tid]
    t = s.time[tid]
    status = int(FLYING)
    for _ in range(p.substeps):
        mp = mass_properties(p.mass, tanks)
        air = atmosphere(b.pos[2])
        oxidizer = tanks[0] + tanks[2]
        fuel = tanks[1] + tanks[3]
        e = engine_step(
            p.propulsion,
            e,
            c.engine_on[tid],
            c.throttle[tid],
            c.gimbal2[tid],
            c.gimbal3[tid],
            oxidizer > 0.0 and fuel > 0.0,
        )
        f = flap_step(p.propulsion, f, c.flaps[tid])
        v_air_world = b.vel
        if p.wind.active != 0:
            ws = wind_step(p.wind, ws, b.pos, b.vel, p.dt)
            v_air_world = b.vel - ws.wind
        v_air = wp.quat_rotate_inv(b.rot, v_air_world)
        shed = wp.vec2(0.0)
        gain = wp.vec4(1.0)
        if p.wind.unsteady != 0:
            ws = unsteady_step(p.wind, ws, v_air, p.dt)
            shed = wp.vec2(p.wind.shedding_amplitude, ws.shed_phase)
            gain = flap_gain(ws)
        w = aero_wrench(p.aero, v_air, b.omega, mp.com, f.angle, air, shed, gain)
        mdot = engine_mass_flow(p.propulsion, e)
        supply = burn_fraction(p.propulsion, tanks, mdot, p.dt)
        thrust = engine_wrench(p.propulsion, e, air.pressure, mp.com)
        thrust.force = thrust.force * supply
        thrust.torque = thrust.torque * supply
        w = add_wrench(w, thrust)
        w = add_wrench(w, rcs_wrench(p.propulsion, c.rcs[tid]))
        force = wp.quat_rotate(b.rot, w.force) - wp.vec3(0.0, 0.0, gravity(b.pos[2]) * mp.mass)
        b = integrate(b, force, w.torque, mp.mass, mp.inertia, p.dt)
        # Match the integrated angular step and use the same air-relative flow as aerodynamics.
        a = track_motion(a, b, v_air, p.dt)
        air_speed_sq = wp.dot(v_air, v_air)
        a = track_flight(
            a,
            b,
            0.5 * air.density * air_speed_sq,
            wp.sqrt(air_speed_sq) / air.speed_of_sound,
            wp.length(w.force) / (mp.mass * G0),
            wp.length(thrust.force) > 0.0,
        )
        tanks = draw_propellant(p.propulsion, tanks, mdot * supply, p.dt)
        t += p.dt
        if lowest_point(p.hull, b, mp.com) <= 0.0:
            status = touchdown_status(p.landing, b, mp.com)
            break
        horizontal = wp.sqrt(b.pos[0] * b.pos[0] + b.pos[1] * b.pos[1])
        if b.pos[2] > p.max_altitude or horizontal > p.max_offset:
            status = OUT_OF_BOUNDS
            break
        axis_z = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))[2]
        transverse = wp.sqrt(b.omega[1] * b.omega[1] + b.omega[2] * b.omega[2])
        if (
            axis_z < wp.cos(p.max_pitch)
            or transverse > p.max_rate
            or wp.abs(b.omega[0]) > p.max_roll_rate
        ):
            status = TUMBLED
            break
    if status == FLYING and s.steps[tid] + 1 >= p.max_steps:
        status = TIMEOUT
    s.body[tid] = b
    s.stats[tid] = a
    s.engine[tid] = e
    s.flaps[tid] = f
    s.tanks[tid] = tanks
    s.wind[tid] = ws
    s.time[tid] = t
    s.steps[tid] = s.steps[tid] + 1
    s.status[tid] = status


@wp.kernel
def step_kernel(p: Params, auto_reset: int, c: Commands, s: State, term: Terminal):
    tid = wp.tid()
    term.done[tid] = 0
    if s.status[tid] != FLYING:
        return
    advance(p, s, c, tid)
    if s.status[tid] != FLYING:
        term.done[tid] = 1
        term.status[tid] = s.status[tid]
        term.body[tid] = s.body[tid]
        term.stats[tid] = s.stats[tid]
        term.tanks[tid] = s.tanks[tid]
        term.time[tid] = s.time[tid]
        term.steps[tid] = s.steps[tid]
        term.wind[tid] = s.wind[tid]
        if auto_reset != 0:
            spawn(p, s, tid)


@wp.kernel
def gather_kernel(sources: wp.array(dtype=wp.int32), src: State, dst: State):
    """dst[i] = src[sources[i]] for everything but the random streams, so copies diverge."""
    tid = wp.tid()
    j = sources[tid]
    dst.body[tid] = src.body[j]
    dst.stats[tid] = src.stats[j]
    dst.engine[tid] = src.engine[j]
    dst.flaps[tid] = src.flaps[j]
    dst.tanks[tid] = src.tanks[j]
    dst.time[tid] = src.time[j]
    dst.steps[tid] = src.steps[j]
    dst.status[tid] = src.status[j]
    wind = src.wind[j]
    wind.rng = dst.wind[tid].rng
    dst.wind[tid] = wind


@wp.kernel
def reset_kernel(p: Params, mask: wp.array(dtype=wp.int32), s: State):
    tid = wp.tid()
    if mask[tid] != 0:
        spawn(p, s, tid)


@wp.kernel
def init_rng_kernel(seed: int, rng: wp.array(dtype=wp.uint32)):
    tid = wp.tid()
    rng[tid] = wp.rand_init(seed, tid)
