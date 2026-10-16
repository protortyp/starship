"""Independent float64 NumPy implementation of the dynamics, used to cross-check the Warp kernel."""

import numpy as np

from starship.sim import CRASHED, FLYING, LANDED, LEFT, MAIN, RIGHT, SimConfig


def density(cfg: SimConfig, altitude: float) -> float:
    h = max(altitude, 0.0)
    if h <= 11000.0:
        ratio = (1.0 - 0.0065 * h / 288.15) ** 4.2559
    else:
        ratio = 0.29707 * np.exp(-(h - 11000.0) / 6341.6)
    return cfg.world.sea_level_density * ratio


def derivatives(cfg: SimConfig, pos, vel, theta, omega, action):
    r = cfg.rocket
    a = np.array([-np.sin(theta), np.cos(theta)])
    n = np.array([np.cos(theta), np.sin(theta)])
    force = np.array([0.0, -r.mass * cfg.world.gravity])
    torque = 0.0
    if action & MAIN:
        force += r.main_thrust * a
    if action & LEFT:
        force += r.side_thrust * n
        torque -= r.side_arm * r.side_thrust
    if action & RIGHT:
        force -= r.side_thrust * n
        torque += r.side_arm * r.side_thrust
    rho = density(cfg, pos[1])
    v_a, v_n = vel @ a, vel @ n
    force -= 0.5 * rho * r.drag_coeff_axial * r.area_axial * abs(v_a) * v_a * a
    force -= 0.5 * rho * r.drag_coeff_normal * r.area_normal * abs(v_n) * v_n * n
    torque -= rho * r.drag_coeff_normal * r.diameter * r.length**4 * omega * abs(omega) / 64.0
    return force / r.mass, torque / r.inertia


def lowest_point(cfg: SimConfig, y, theta):
    r = cfg.rocket
    return y - 0.5 * r.length * abs(np.cos(theta)) - 0.5 * r.diameter * abs(np.sin(theta))


def step(cfg: SimConfig, pos, vel, theta, omega, action):
    """One policy step; returns (pos, vel, theta, omega, status)."""
    pos, vel = pos.astype(np.float64), vel.astype(np.float64)
    theta, omega = float(theta), float(omega)
    for _ in range(cfg.substeps):
        lin, ang = derivatives(cfg, pos, vel, theta, omega, action)
        vel = vel + lin * cfg.dt
        omega = omega + ang * cfg.dt
        pos = pos + vel * cfg.dt
        theta = float(np.arctan2(np.sin(theta + omega * cfg.dt), np.cos(theta + omega * cfg.dt)))
        if lowest_point(cfg, pos[1], theta) <= 0.0:
            land = cfg.landing
            ok = (
                abs(pos[0]) <= land.pad_half_width
                and abs(vel[1]) <= land.max_vertical_speed
                and abs(vel[0]) <= land.max_horizontal_speed
                and abs(theta) <= land.max_tilt
                and abs(omega) <= land.max_angular_rate
            )
            return pos, vel, theta, omega, LANDED if ok else CRASHED
    return pos, vel, theta, omega, FLYING
