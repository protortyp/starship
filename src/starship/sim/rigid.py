"""Rigid-body integration and gravity.

Semi-implicit Euler for the translation. The attitude advances by the exponential map of the body
angular velocity over a step, which keeps the quaternion unit length without renormalisation
error building up.
"""

import warp as wp

EARTH_MEAN_RADIUS = 6371000.0
G0 = 9.80665


@wp.func
def gravity(altitude: float) -> float:
    r = EARTH_MEAN_RADIUS / (EARTH_MEAN_RADIUS + altitude)
    return G0 * r * r


@wp.func
def angular_acceleration(inertia: wp.vec3, omega: wp.vec3, torque: wp.vec3) -> wp.vec3:
    """Euler's equations for principal axes: I dw/dt = tau - w x (I w)."""
    momentum = wp.cw_mul(inertia, omega)
    return wp.cw_div(torque - wp.cross(omega, momentum), inertia)


@wp.func
def rotate_by_body_rate(q: wp.quat, omega: wp.vec3, dt: float) -> wp.quat:
    rate = wp.length(omega)
    if rate < 1.0e-9:
        return q
    dq = wp.quat_from_axis_angle(omega / rate, rate * dt)
    return wp.normalize(q * dq)


@wp.struct
class Body:
    pos: wp.vec3  # centre of mass, world
    vel: wp.vec3  # world
    rot: wp.quat  # body to world
    omega: wp.vec3  # body


@wp.func
def integrate(
    b: Body, force_world: wp.vec3, torque_body: wp.vec3, mass: float, inertia: wp.vec3, dt: float
) -> Body:
    out = Body()
    out.vel = b.vel + force_world * (dt / mass)
    out.pos = b.pos + out.vel * dt
    # RK4 on Euler's equations with the torque held over the step: explicit Euler on the
    # gyroscopic term pumps energy into a tumbling body.
    k1 = angular_acceleration(inertia, b.omega, torque_body)
    k2 = angular_acceleration(inertia, b.omega + k1 * (0.5 * dt), torque_body)
    k3 = angular_acceleration(inertia, b.omega + k2 * (0.5 * dt), torque_body)
    k4 = angular_acceleration(inertia, b.omega + k3 * dt, torque_body)
    out.omega = b.omega + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)
    out.rot = rotate_by_body_rate(b.rot, out.omega, dt)
    return out


@wp.kernel
def _integrate_kernel(
    bodies: wp.array(dtype=Body),
    force_world: wp.vec3,
    torque_body: wp.vec3,
    mass: float,
    inertia: wp.vec3,
    dt: float,
    steps: int,
):
    i = wp.tid()
    b = bodies[i]
    for _ in range(steps):
        f = force_world - wp.vec3(0.0, 0.0, gravity(b.pos[2]) * mass)
        b = integrate(b, f, torque_body, mass, inertia, dt)
    bodies[i] = b


def simulate(
    bodies: list[tuple],
    force=(0.0, 0.0, 0.0),
    torque=(0.0, 0.0, 0.0),
    mass: float = 1.0,
    inertia=(1.0, 1.0, 1.0),
    dt: float = 0.01,
    steps: int = 100,
    device: str | None = None,
) -> list[Body]:
    """Integrate free bodies (pos, vel, rot as (x, y, z, w), omega) under gravity plus a constant
    force (world) and torque (body); for tests."""
    host = []
    for pos, vel, rot, omega in bodies:
        b = Body()
        b.pos, b.vel, b.rot, b.omega = wp.vec3(*pos), wp.vec3(*vel), wp.quat(*rot), wp.vec3(*omega)
        host.append(b)
    arr = wp.array(host, dtype=Body, device=device)
    wp.launch(
        _integrate_kernel,
        dim=len(host),
        inputs=[arr, wp.vec3(*force), wp.vec3(*torque), mass, wp.vec3(*inertia), dt, steps],
        device=device,
    )
    return list(arr.numpy())
