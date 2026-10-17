"""Batched 6-DOF step: actuators, mass, aerodynamics, engines and ground contact per substep.

Each env runs `substeps` physics steps per control step with its commands held. An episode ends
when the hull touches the ground (landed or crashed), leaves the bounds, or times out; with
auto-reset the final state goes to the terminal buffers and the env respawns in the same launch.
Aerodynamics act on the air-relative velocity (`starship.sim.wind`); ground speed is the state.
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
from starship.sim.rigid import G0, Body, gravity, integrate
from starship.sim.types import add_wrench
from starship.sim.wind import (
    WindParams,
    WindState,
    flap_gain,
    sample_wind,
    unsteady_step,
    wind_step,
)

FLYING = 0
LANDED = 1
CRASHED = 2
OUT_OF_BOUNDS = 3
TIMEOUT = 4
TUMBLED = 5

ATTITUDE_FIELDS = (
    "roll_travel",
    "horizontal_travel",
    "peak_roll_rate",
    "peak_horizontal_rate",
    "flips",
    "reflips",
    "peak_dynamic_pressure",
    "peak_mach",
    "peak_load",
    "flip_altitude",
    "ignition_altitude",
    "start_offset",
)


@wp.struct
class Attitude:
    roll_travel: float
    horizontal_travel: float
    peak_roll_rate: float
    peak_horizontal_rate: float
    flips: int
    reflips: int
    sideslip: float
    phase: int  # 0: neither threshold reached; 1: above 60 degrees; 2: below 30 degrees
    peak_dynamic_pressure: float  # Pa, air-relative
    peak_mach: float
    peak_load: float  # non-gravitational acceleration, g
    flip_altitude: float  # m at the first flip to upright, -1 if none
    ignition_altitude: float  # m when thrust first appears, -1 if never
    start_offset: float  # m, horizontal distance to the pad at spawn


@wp.func
def track_attitude(a: Attitude, b: Body, v_air: wp.vec3, dt: float) -> Attitude:
    roll = wp.abs(b.omega[0])
    world_rate = wp.quat_rotate(b.rot, b.omega)
    horizontal = wp.length(wp.vec2(world_rate[0], world_rate[1]))
    a.roll_travel += roll * dt
    a.horizontal_travel += horizontal * dt
    a.peak_roll_rate = wp.max(a.peak_roll_rate, roll)
    a.peak_horizontal_rate = wp.max(a.peak_horizontal_rate, horizontal)
    speed_sq = wp.dot(v_air, v_air)
    if speed_sq > 400.0:
        a.sideslip += v_air[1] * v_air[1] / speed_sq * dt
    axis_z = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))[2]
    if axis_z < wp.cos(wp.pi / 3.0):
        if a.phase == 2:
            a.reflips += 1
        a.phase = 1
    elif axis_z > wp.cos(wp.pi / 6.0):
        if a.phase == 1:
            a.flips += 1
        a.phase = 2
    return a


@wp.func
def track_flight(
    a: Attitude, b: Body, dynamic_pressure: float, mach: float, load: float, thrusting: bool
) -> Attitude:
    a.peak_dynamic_pressure = wp.max(a.peak_dynamic_pressure, dynamic_pressure)
    a.peak_mach = wp.max(a.peak_mach, mach)
    a.peak_load = wp.max(a.peak_load, load)
    if a.flips > 0 and a.flip_altitude < 0.0:
        a.flip_altitude = b.pos[2]
    if thrusting and a.ignition_altitude < 0.0:
        a.ignition_altitude = b.pos[2]
    return a


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
class Params:
    dt: float
    substeps: int
    max_steps: int
    max_altitude: float
    max_offset: float
    max_pitch: float  # from vertical; beyond it the vehicle counts as tumbling
    max_rate: float
    max_roll_rate: float
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
    attitude: wp.array(dtype=Attitude)
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
    attitude: wp.array(dtype=Attitude)
    tanks: wp.array(dtype=wp.vec4)
    time: wp.array(dtype=float)
    steps: wp.array(dtype=wp.int32)
    wind: wp.array(dtype=WindState)


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
    b.rot = attitude(pitch, heading)
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
        b.rot = attitude(angle + pitch - 0.5 * wp.pi, heading)
    s.body[tid] = b
    a = track_attitude(Attitude(), b, wp.vec3(0.0), 0.0)
    a.flip_altitude = -1.0
    a.ignition_altitude = -1.0
    if sp.lit_throttle > 0.0:
        a.ignition_altitude = b.pos[2]
    a.start_offset = wp.length(wp.vec2(b.pos[0], b.pos[1]))
    s.attitude[tid] = a

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
    a = s.attitude[tid]
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
        a = track_attitude(a, b, v_air, p.dt)
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
        if lowest_point(p, b, mp.com) <= 0.0:
            status = touchdown_status(p, b, mp.com)
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
    s.attitude[tid] = a
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
        term.attitude[tid] = s.attitude[tid]
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
    dst.attitude[tid] = src.attitude[j]
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
