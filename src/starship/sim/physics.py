"""Batched 6-DOF step: actuators, mass, aerodynamics, engines and ground contact per substep.

Each env runs `substeps` physics steps per control step with its commands held. An episode ends
when the hull touches the ground (landed or crashed), leaves the bounds, or times out; with
auto-reset the final state goes to the terminal buffers and the env respawns in the same launch.
"""

import warp as wp

from starship.sim.aero import AeroParams, aero_wrench
from starship.sim.atmosphere import atmosphere
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
from starship.sim.rigid import Body, gravity, integrate
from starship.sim.types import add_wrench

FLYING = 0
LANDED = 1
CRASHED = 2
OUT_OF_BOUNDS = 3
TIMEOUT = 4


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
    pitch: float
    pitch_noise: float
    angular_rate_noise: float
    flap_angle: float
    tanks: wp.vec4


@wp.struct
class Params:
    dt: float
    substeps: int
    max_time: float
    max_altitude: float
    max_offset: float
    length: float
    radius: float
    nose_length: float
    ogive_radius: float  # radius of the tangent-ogive arc
    pad_radius: float
    max_vertical_speed: float
    max_horizontal_speed: float
    max_tilt: float
    max_angular_rate: float
    spawn: SpawnParams
    mass: MassParams
    propulsion: PropulsionParams
    aero: AeroParams


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
    engine: wp.array(dtype=EngineState)
    flaps: wp.array(dtype=FlapState)
    tanks: wp.array(dtype=wp.vec4)
    time: wp.array(dtype=float)
    steps: wp.array(dtype=wp.int32)
    status: wp.array(dtype=wp.int32)
    # Per-env random stream, advanced on every respawn.
    rng: wp.array(dtype=wp.uint32)


@wp.struct
class Terminal:
    """Final state of the episode that ended on the last step (valid where done is set)."""

    done: wp.array(dtype=wp.int32)
    status: wp.array(dtype=wp.int32)
    body: wp.array(dtype=Body)
    tanks: wp.array(dtype=wp.vec4)
    time: wp.array(dtype=float)
    steps: wp.array(dtype=wp.int32)


@wp.func
def attitude(pitch: float, heading: float) -> wp.quat:
    """Body axis b1 tilted `pitch` from vertical towards `heading`, belly (b3) facing down."""
    sp, cp = wp.sin(pitch), wp.cos(pitch)
    sh, ch = wp.sin(heading), wp.cos(heading)
    b1 = wp.vec3(sp * ch, sp * sh, cp)
    b3 = wp.vec3(cp * ch, cp * sh, -sp)
    b2 = wp.cross(b3, b1)
    m = wp.mat33(b1[0], b2[0], b3[0], b1[1], b2[1], b3[1], b1[2], b2[2], b3[2])
    return wp.quat_from_matrix(m)


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
    b.rot = attitude(pitch, heading)
    b.omega = wp.vec3(wp.randf(rng, -w, w), wp.randf(rng, -w, w), wp.randf(rng, -w, w))
    s.body[tid] = b

    e = EngineState()
    e.throttle = wp.vec3(p.propulsion.min_throttle)
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


@wp.func
def lowest_point(p: Params, b: Body, com: float) -> float:
    """Height of the lowest hull point: the engine-plane rim, the nose-base rim, the tip, or the
    point of the tangent-ogive nose whose surface slope matches the tilt."""
    axis = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))
    s = wp.sqrt(wp.max(1.0 - axis[2] * axis[2], 0.0))
    base = b.pos[2] - com * axis[2]
    nose_base = p.length - p.nose_length
    tail = base - p.radius * s
    shoulder = base + nose_base * axis[2] - p.radius * s
    tip = base + p.length * axis[2]
    lowest = wp.min(wp.min(tail, shoulder), tip)
    # Surface height along the nose: z(u) = z(nose_base) + u a_z - r(u) s, with the ogive radius
    # r(u) = sqrt(R^2 - u^2) + radius - R. Its minimum sits where u / sqrt(R^2 - u^2) = -a_z / s,
    # which exists only when the nose points down.
    if axis[2] < 0.0 and s > 0.0:
        k = -axis[2] / s
        u = wp.min(p.ogive_radius * k / wp.sqrt(1.0 + k * k), p.nose_length)
        r = (
            wp.sqrt(wp.max(p.ogive_radius * p.ogive_radius - u * u, 0.0))
            + p.radius
            - p.ogive_radius
        )
        lowest = wp.min(lowest, base + (nose_base + u) * axis[2] - wp.max(r, 0.0) * s)
    return lowest


@wp.func
def touchdown_status(p: Params, b: Body, com: float) -> int:
    axis = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))
    base = b.pos - axis * com
    horizontal_speed = wp.sqrt(b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1])
    miss = wp.sqrt(base[0] * base[0] + base[1] * base[1])
    tilt = wp.acos(wp.clamp(axis[2], -1.0, 1.0))
    rate = wp.length(b.omega)
    if (
        -b.vel[2] <= p.max_vertical_speed
        and horizontal_speed <= p.max_horizontal_speed
        and tilt <= p.max_tilt
        and rate <= p.max_angular_rate
        and miss <= p.pad_radius
    ):
        return LANDED
    return CRASHED


@wp.func
def advance(p: Params, s: State, c: Commands, tid: int):
    b = s.body[tid]
    e = s.engine[tid]
    f = s.flaps[tid]
    tanks = s.tanks[tid]
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
        v_air = wp.quat_rotate_inv(b.rot, b.vel)
        w = aero_wrench(p.aero, v_air, b.omega, mp.com, f.angle, air)
        mdot = engine_mass_flow(p.propulsion, e)
        supply = burn_fraction(p.propulsion, tanks, mdot, p.dt)
        thrust = engine_wrench(p.propulsion, e, air.pressure, mp.com)
        thrust.force = thrust.force * supply
        thrust.torque = thrust.torque * supply
        w = add_wrench(w, thrust)
        w = add_wrench(w, rcs_wrench(p.propulsion, c.rcs[tid]))
        force = wp.quat_rotate(b.rot, w.force) - wp.vec3(0.0, 0.0, gravity(b.pos[2]) * mp.mass)
        b = integrate(b, force, w.torque, mp.mass, mp.inertia, p.dt)
        tanks = draw_propellant(p.propulsion, tanks, mdot * supply, p.dt)
        t += p.dt
        if lowest_point(p, b, mp.com) <= 0.0:
            status = touchdown_status(p, b, mp.com)
            break
        horizontal = wp.sqrt(b.pos[0] * b.pos[0] + b.pos[1] * b.pos[1])
        if b.pos[2] > p.max_altitude or horizontal > p.max_offset:
            status = OUT_OF_BOUNDS
            break
    if status == FLYING and t >= p.max_time:
        status = TIMEOUT
    s.body[tid] = b
    s.engine[tid] = e
    s.flaps[tid] = f
    s.tanks[tid] = tanks
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
        term.tanks[tid] = s.tanks[tid]
        term.time[tid] = s.time[tid]
        term.steps[tid] = s.steps[tid]
        if auto_reset != 0:
            spawn(p, s, tid)


@wp.kernel
def reset_kernel(p: Params, mask: wp.array(dtype=wp.int32), s: State):
    tid = wp.tid()
    if mask[tid] != 0:
        spawn(p, s, tid)


@wp.kernel
def init_rng_kernel(seed: int, rng: wp.array(dtype=wp.uint32)):
    tid = wp.tid()
    rng[tid] = wp.rand_init(seed, tid)
