"""Warp functions and kernels for the 2D rigid-body rocket (frame conventions in `config.py`)."""

import warp as wp

from starship.sim.config import (
    CRASHED,
    FLYING,
    LANDED,
    LEFT,
    MAIN,
    OUT_OF_BOUNDS,
    RIGHT,
    TIMEOUT,
    Params,
)

# International Standard Atmosphere, troposphere (0-11 km) and lower stratosphere (11-20 km).
ISA_T0 = wp.constant(288.15)
ISA_LAPSE = wp.constant(0.0065)
ISA_DENSITY_EXP = wp.constant(4.2559)
ISA_TROPOPAUSE = wp.constant(11000.0)
ISA_TROPOPAUSE_RATIO = wp.constant(0.29707)
ISA_STRAT_SCALE_HEIGHT = wp.constant(6341.6)

STATUS_FLYING = wp.constant(FLYING)
STATUS_LANDED = wp.constant(LANDED)
STATUS_CRASHED = wp.constant(CRASHED)
STATUS_OUT_OF_BOUNDS = wp.constant(OUT_OF_BOUNDS)
STATUS_TIMEOUT = wp.constant(TIMEOUT)
ACTION_MAIN = wp.constant(MAIN)
ACTION_LEFT = wp.constant(LEFT)
ACTION_RIGHT = wp.constant(RIGHT)


@wp.func
def density_ratio(altitude: float) -> float:
    """Air density relative to sea level."""
    h = wp.max(altitude, 0.0)
    if h <= ISA_TROPOPAUSE:
        return wp.pow(1.0 - ISA_LAPSE * h / ISA_T0, ISA_DENSITY_EXP)
    return ISA_TROPOPAUSE_RATIO * wp.exp(-(h - ISA_TROPOPAUSE) / ISA_STRAT_SCALE_HEIGHT)


@wp.func
def body_axis(theta: float) -> wp.vec2:
    return wp.vec2(-wp.sin(theta), wp.cos(theta))


@wp.func
def body_normal(theta: float) -> wp.vec2:
    return wp.vec2(wp.cos(theta), wp.sin(theta))


@wp.func
def wrap_angle(theta: float) -> float:
    return wp.atan2(wp.sin(theta), wp.cos(theta))


@wp.func
def lowest_point(y: float, theta: float, p: Params) -> float:
    """Height of the lowest corner of the rocket's rectangular outline."""
    return y - 0.5 * p.length * wp.abs(wp.cos(theta)) - 0.5 * p.diameter * wp.abs(wp.sin(theta))


@wp.func
def accelerations(
    pos: wp.vec2, vel: wp.vec2, theta: float, omega: float, action: int, fuel: float, p: Params
):
    a = body_axis(theta)
    n = body_normal(theta)
    mass = p.dry_mass + fuel

    force = wp.vec2(0.0, -mass * p.gravity)
    torque = 0.0

    if action & ACTION_MAIN:
        force += p.main_thrust * a
    # Side thrusters are named by mounting side: the left one exhausts to the left, pushing the
    # nose right (clockwise); the right one pushes the nose left (counter-clockwise).
    if action & ACTION_LEFT:
        force += p.side_thrust * n
        torque -= p.side_arm * p.side_thrust
    if action & ACTION_RIGHT:
        force -= p.side_thrust * n
        torque += p.side_arm * p.side_thrust

    # Crossflow drag: axial and normal velocity components are opposed independently, with the
    # center of pressure at the center of mass, so drag produces no torque from translation.
    rho = p.sea_level_density * density_ratio(pos[1])
    v_a = wp.dot(vel, a)
    v_n = wp.dot(vel, n)
    force -= 0.5 * rho * p.drag_coeff_axial * p.area_axial * wp.abs(v_a) * v_a * a
    force -= 0.5 * rho * p.drag_coeff_normal * p.area_normal * wp.abs(v_n) * v_n * n

    # Rotation through still air: integrating normal-flow drag over the length gives
    # tau = -rho Cd D L^4 omega |omega| / 64.
    l2 = p.length * p.length
    torque -= rho * p.drag_coeff_normal * p.diameter * l2 * l2 * omega * wp.abs(omega) / 64.0

    return force / mass, torque / (mass * p.inertia_per_kg)


@wp.func
def fuel_flow(action: int, p: Params) -> float:
    flow = 0.0
    if action & ACTION_MAIN:
        flow += p.main_mass_flow
    if action & ACTION_LEFT:
        flow += p.side_mass_flow
    if action & ACTION_RIGHT:
        flow += p.side_mass_flow
    return flow


@wp.func
def touchdown_status(pos: wp.vec2, vel: wp.vec2, theta: float, omega: float, p: Params) -> int:
    if (
        wp.abs(pos[0]) <= p.pad_half_width
        and wp.abs(vel[1]) <= p.max_vertical_speed
        and wp.abs(vel[0]) <= p.max_horizontal_speed
        and wp.abs(theta) <= p.max_tilt
        and wp.abs(omega) <= p.max_angular_rate
    ):
        return STATUS_LANDED
    return STATUS_CRASHED


@wp.struct
class State:
    pos: wp.array(dtype=wp.vec2)
    vel: wp.array(dtype=wp.vec2)
    theta: wp.array(dtype=float)
    omega: wp.array(dtype=float)
    status: wp.array(dtype=wp.int32)
    steps: wp.array(dtype=wp.int32)
    fuel: wp.array(dtype=float)
    # Per-env random stream, advanced on every respawn.
    rng: wp.array(dtype=wp.uint32)


@wp.struct
class Terminal:
    """State at the moment an episode ended, valid where `done` is set for the latest step."""

    done: wp.array(dtype=wp.int32)
    status: wp.array(dtype=wp.int32)
    pos: wp.array(dtype=wp.vec2)
    vel: wp.array(dtype=wp.vec2)
    theta: wp.array(dtype=float)
    omega: wp.array(dtype=float)
    steps: wp.array(dtype=wp.int32)
    fuel: wp.array(dtype=float)


@wp.func
def spawn(p: Params, s: State, tid: int):
    rng = s.rng[tid]
    sp = p.spawn
    if wp.randf(rng) < p.easy_spawn_fraction:
        sp = p.easy_spawn
    s.pos[tid] = wp.vec2(
        sp.pos[0] + sp.pos_noise[0] * wp.randf(rng, -1.0, 1.0),
        sp.pos[1] + sp.pos_noise[1] * wp.randf(rng, -1.0, 1.0),
    )
    s.vel[tid] = wp.vec2(
        sp.vel[0] + sp.vel_noise[0] * wp.randf(rng, -1.0, 1.0),
        sp.vel[1] + sp.vel_noise[1] * wp.randf(rng, -1.0, 1.0),
    )
    s.theta[tid] = wrap_angle(sp.theta + sp.theta_noise * wp.randf(rng, -1.0, 1.0))
    s.omega[tid] = sp.omega + sp.omega_noise * wp.randf(rng, -1.0, 1.0)
    s.status[tid] = STATUS_FLYING
    s.steps[tid] = 0
    s.fuel[tid] = p.fuel_mass
    s.rng[tid] = rng


@wp.kernel
def step_kernel(
    p: Params, auto_reset: int, actions: wp.array(dtype=wp.int32), s: State, term: Terminal
):
    tid = wp.tid()
    term.done[tid] = 0
    if s.status[tid] != STATUS_FLYING:
        return

    x = s.pos[tid]
    v = s.vel[tid]
    th = s.theta[tid]
    om = s.omega[tid]
    fuel = s.fuel[tid]
    action = actions[tid]
    st = STATUS_FLYING

    for _ in range(p.substeps):
        # An empty tank silences every thruster.
        if fuel <= 0.0:
            action = 0
        lin_acc, ang_acc = accelerations(x, v, th, om, action, fuel, p)
        # Semi-implicit Euler: velocities first, then positions with the updated velocities.
        v = v + lin_acc * p.dt
        om = om + ang_acc * p.dt
        x = x + v * p.dt
        th = wrap_angle(th + om * p.dt)
        fuel = wp.max(fuel - fuel_flow(action, p) * p.dt, 0.0)
        if lowest_point(x[1], th, p) <= 0.0:
            st = touchdown_status(x, v, th, om, p)
            break

    n_steps = s.steps[tid] + 1
    if st == STATUS_FLYING:
        if wp.abs(x[0]) > p.max_abs_x or x[1] > p.max_y:
            st = STATUS_OUT_OF_BOUNDS
        elif n_steps >= p.max_steps:
            st = STATUS_TIMEOUT

    s.pos[tid] = x
    s.vel[tid] = v
    s.theta[tid] = th
    s.omega[tid] = om
    s.status[tid] = st
    s.steps[tid] = n_steps
    s.fuel[tid] = fuel

    if st != STATUS_FLYING:
        term.done[tid] = 1
        term.status[tid] = st
        term.pos[tid] = x
        term.vel[tid] = v
        term.theta[tid] = th
        term.omega[tid] = om
        term.steps[tid] = n_steps
        term.fuel[tid] = fuel
        if auto_reset != 0:
            spawn(p, s, tid)


@wp.kernel
def reset_kernel(p: Params, mask: wp.array(dtype=wp.int32), s: State, term: Terminal):
    tid = wp.tid()
    if mask[tid] == 0:
        return
    term.done[tid] = 0
    spawn(p, s, tid)


@wp.kernel
def init_rng_kernel(seed: int, rng: wp.array(dtype=wp.uint32)):
    tid = wp.tid()
    rng[tid] = wp.rand_init(seed, tid)
