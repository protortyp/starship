"""Potential shaping and costs in physical units; terminal potential is always zero."""

from dataclasses import dataclass, fields

import warp as wp

from starship.sim import CRASHED, LANDED, OUT_OF_BOUNDS, TIMEOUT, TUMBLED
from starship.sim.mass import mass_properties
from starship.sim.physics import Params, State, Terminal, lowest_point
from starship.sim.rigid import Body
from starship.train.actions import CONTINUOUS_HEADS
from starship.train.encoding import fuel_fraction

REWARD_TERMS = ("distance", "speed", "tilt", "angular_rate", "jerk", "fuel", "time", "terminal")


@dataclass
class RewardConfig:
    gamma: float = 0.999  # Must equal PPO's discount.
    distance_weight: float = 0.01  # per m
    speed_weight: float = 1.0  # per m/s above the safe profile
    tilt_weight: float = 10.0  # per rad, faded out above tilt_height
    safe_deceleration: float = 20.0  # m/s^2 net; conservative three-engine landing burn
    tilt_height: float = 200.0  # m above ground contact
    angular_rate_cost: float = 0.1  # per (rad/s)^2 s
    jerk_cost: float = 0.1  # per squared change in squashed actuator targets
    fuel_cost: float = 0.001  # per kg
    time_cost: float = 0.01  # per s
    landed_bonus: float = 100.0
    fuel_bonus: float = 50.0
    crash_penalty: float = 100.0
    crash_vertical_weight: float = 1.0  # per m/s excess
    crash_horizontal_weight: float = 1.0  # per m/s excess
    crash_tilt_weight: float = 20.0  # per rad excess
    crash_rate_weight: float = 5.0  # per rad/s excess
    crash_miss_weight: float = 0.1  # per m outside the pad
    # Caps the weighted excess so a crash costs at most crash_penalty + cap: unbounded penalties
    # (a km-scale miss) swamp the value targets. Kept below tumble_penalty - crash_penalty.
    crash_excess_cap: float = 50.0
    tumble_penalty: float = 200.0
    out_of_bounds_penalty: float = 100.0
    timeout_penalty: float = 50.0


@wp.struct
class RewardParams:
    gamma: float
    distance_weight: float
    speed_weight: float
    tilt_weight: float
    safe_deceleration: float
    tilt_height: float
    angular_rate_cost: float
    jerk_cost: float
    fuel_cost: float
    time_cost: float
    landed_bonus: float
    fuel_bonus: float
    crash_penalty: float
    crash_vertical_weight: float
    crash_horizontal_weight: float
    crash_tilt_weight: float
    crash_rate_weight: float
    crash_miss_weight: float
    crash_excess_cap: float
    tumble_penalty: float
    out_of_bounds_penalty: float
    timeout_penalty: float


def make_reward_params(config: RewardConfig) -> RewardParams:
    import math

    for field in fields(config):
        value = getattr(config, field.name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"reward {field.name} must be finite and nonnegative")
    if config.gamma > 1 or config.tilt_height <= 0:
        raise ValueError("gamma must be <= 1 and tilt_height must be positive")
    params = RewardParams()
    for field in fields(config):
        setattr(params, field.name, float(getattr(config, field.name)))
    return params


@wp.func
def potential(p: Params, r: RewardParams, b: Body, tanks: wp.vec4) -> wp.vec3:
    com = mass_properties(p.mass, tanks).com
    height = wp.max(lowest_point(p, b, com), 0.0)
    safe_speed = wp.sqrt(2.0 * r.safe_deceleration * height) + p.max_vertical_speed
    excess_speed = wp.max(wp.length(b.vel) - safe_speed, 0.0)
    axis_z = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))[2]
    tilt = wp.acos(wp.clamp(axis_z, -1.0, 1.0))
    fade = wp.clamp(1.0 - height / r.tilt_height, 0.0, 1.0)
    return wp.vec3(
        -r.distance_weight * wp.length(b.pos),
        -r.speed_weight * excess_speed,
        -r.tilt_weight * tilt * fade,
    )


@wp.func
def terminal_value(p: Params, r: RewardParams, t: Terminal, i: int, load: float) -> float:
    status = t.status[i]
    if status == LANDED:
        return r.landed_bonus + r.fuel_bonus * fuel_fraction(t.tanks[i], load)
    if status == CRASHED:
        b = t.body[i]
        axis = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))
        base = b.pos - axis * mass_properties(p.mass, t.tanks[i]).com
        horizontal = wp.length(wp.vec2(b.vel[0], b.vel[1]))
        miss = wp.length(wp.vec2(base[0], base[1]))
        tilt = wp.acos(wp.clamp(axis[2], -1.0, 1.0))
        excess = (
            r.crash_vertical_weight * wp.max(-b.vel[2] - p.max_vertical_speed, 0.0)
            + r.crash_horizontal_weight * wp.max(horizontal - p.max_horizontal_speed, 0.0)
            + r.crash_tilt_weight * wp.max(tilt - p.max_tilt, 0.0)
            + r.crash_rate_weight * wp.max(wp.length(b.omega) - p.max_angular_rate, 0.0)
            + r.crash_miss_weight * wp.max(miss - p.pad_radius, 0.0)
        )
        return -(r.crash_penalty + wp.min(excess, r.crash_excess_cap))
    if status == TUMBLED:
        return -r.tumble_penalty
    if status == OUT_OF_BOUNDS:
        return -r.out_of_bounds_penalty
    if status == TIMEOUT:
        return -r.timeout_penalty
    return 0.0


@wp.kernel
def reward_kernel(
    p: Params,
    r: RewardParams,
    state: State,
    terminal: Terminal,
    before: wp.array(dtype=wp.vec3),
    fuel_before: wp.array(dtype=float),
    time_before: wp.array(dtype=float),
    landing_load: wp.array(dtype=float),
    squashed: wp.array2d(dtype=float),
    previous: wp.array2d(dtype=float),
    has_previous: wp.array(dtype=int),
    reward: wp.array(dtype=float),
    terminal_reward: wp.array(dtype=float),
    terms: wp.array2d(dtype=float),
    terminated: wp.array(dtype=int),
    truncated: wp.array(dtype=int),
):
    i = wp.tid()
    b = state.body[i]
    tanks = state.tanks[i]
    time = state.time[i]
    done = terminal.done[i] != 0
    phi = wp.vec3(0.0)
    terminal_reward[i] = 0.0
    terminated[i] = 0
    truncated[i] = 0
    if done:
        b = terminal.body[i]
        tanks = terminal.tanks[i]
        time = terminal.time[i]
        terminal_reward[i] = terminal_value(p, r, terminal, i, landing_load[i])
        if terminal.status[i] == TIMEOUT:
            truncated[i] = 1
        else:
            terminated[i] = 1
    else:
        phi = potential(p, r, b, tanks)
    shaping = r.gamma * phi - before[i]
    for j in range(3):
        terms[i, j] = shaping[j]
    dt = wp.max(time - time_before[i], 0.0)
    jerk = 0.0
    for j in range(CONTINUOUS_HEADS):
        if has_previous[i] != 0:
            delta = squashed[i, j] - previous[i, j]
            jerk += delta * delta
        previous[i, j] = squashed[i, j]
        if done:
            previous[i, j] = 0.0
    has_previous[i] = 1
    if done:
        has_previous[i] = 0
    terms[i, 3] = -r.angular_rate_cost * wp.dot(b.omega, b.omega) * dt
    terms[i, 4] = -r.jerk_cost * jerk
    burned = wp.max(fuel_before[i] - wp.dot(tanks, wp.vec4(1.0)), 0.0)
    terms[i, 5] = -r.fuel_cost * burned
    terms[i, 6] = -r.time_cost * dt
    terms[i, 7] = terminal_reward[i]
    total = 0.0
    for j in range(8):
        total += terms[i, j]
    reward[i] = total
