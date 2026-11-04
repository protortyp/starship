"""Potential shaping and costs in physical units; terminal potential is always zero."""

from dataclasses import dataclass, fields

import warp as wp

from starship.sim import CRASHED, LANDED, OUT_OF_BOUNDS, TIMEOUT, TUMBLED
from starship.sim.mass import mass_properties
from starship.sim.physics import Attitude, Params, State, Terminal, lowest_point
from starship.sim.rigid import Body
from starship.train.actions import CONTINUOUS_HEADS
from starship.train.encoding import fuel_fraction

REWARD_TERMS = (
    "distance",
    "speed",
    "tilt",
    "angular_rate",
    "jerk",
    "fuel",
    "time",
    "terminal",
    "roll_travel_cost",
    "horizontal_rotation_cost",
    "sideslip_cost",
    "reflip_cost",
    "loop_progress",
    "loop_overshoot_cost",
    "loop_rate_cost",
)


@dataclass
class RewardConfig:
    gamma: float = 0.999  # Must equal PPO's discount.
    distance_weight: float = 0.05  # per m
    distance_saturation: float = 0.0  # m; zero retains legacy linear shaping
    # Dedicated pad-centering potential. Total 3D distance is dominated by altitude during the
    # descent, so it gives little gradient for correcting a lateral miss. Fade this term in near
    # the ground, where horizontal position determines whether contact is on the pad.
    horizontal_distance_weight: float = 0.0  # per horizontal m
    horizontal_distance_height: float = 500.0  # m
    speed_weight: float = 1.0  # per m/s of total speed above the safe profile (legacy)
    # Vertical-only counterpart for landing curricula whose hard vertical and horizontal limits are
    # already shaped separately. Keeping this additive preserves every existing checkpoint/config.
    vertical_speed_weight: float = 0.0  # per m/s of descent above the safe profile
    # Horizontal speed is nearly free under the descent profile alone; without this term policies
    # tip over and fly away sideways (2D needed the same lateral shaping, docs/curriculum.md).
    horizontal_speed_weight: float = 0.5  # per m/s, part of the speed term
    # Optional extra braking pressure near touchdown. Keeping it additive preserves the global
    # anti-flyaway term while avoiding a larger high-altitude penalty on cross-range corrections.
    touchdown_horizontal_speed_weight: float = 0.0  # per m/s at ground level
    horizontal_speed_height: float = 500.0  # m, exponential fade height for the extra term
    tilt_weight: float = 10.0  # per rad, faded out above tilt_height
    safe_deceleration: float = 10.0  # m/s^2 net, well inside the three-engine landing burn
    tilt_height: float = 1000.0  # m above ground contact
    angular_rate_cost: float = 0.1  # per (rad/s)^2 s
    jerk_cost: float = 0.1  # per squared change in squashed actuator targets
    fuel_cost: float = 0.001  # per kg
    time_cost: float = 0.01  # per s
    landed_bonus: float = 100.0
    fuel_bonus: float = 50.0
    # Terminal ordering: landing > gentle crash > hard crash > timeout / out of bounds > tumble.
    # A timeout must cost more than a typical crash, or policies learn to avoid the ground.
    crash_penalty: float = 20.0
    crash_vertical_weight: float = 1.0  # per m/s excess
    crash_horizontal_weight: float = 1.0  # per m/s excess
    crash_tilt_weight: float = 20.0  # per rad excess
    crash_rate_weight: float = 5.0  # per rad/s excess
    crash_miss_weight: float = 0.1  # per m outside the pad
    # Crash penalty = crash_penalty + crash_excess_scale * log(1 + weighted excess). Linear
    # penalties let km-scale misses swamp the value targets, and a hard cap ranks all bad crashes
    # the same; the log keeps ranking them while a 2 km miss stays below the tumble penalty.
    crash_excess_scale: float = 10.0
    tumble_penalty: float = 200.0
    out_of_bounds_penalty: float = 150.0
    timeout_penalty: float = 150.0
    roll_travel_cost: float = 0.0  # per rad
    horizontal_rotation_cost: float = 0.0  # per rad about world-horizontal axes
    sideslip_cost: float = 0.0  # per squared air-relative sideslip fraction s above 20 m/s
    reflip_cost: float = 0.0  # per return above 60 degrees after reaching below 30 degrees
    # Optional stunt objective. Horizontal travel integrates rotation about either world-horizontal
    # axis, excluding cheap spins about the ship's own longitudinal axis. A zero target disables it.
    loop_target: float = 0.0  # rad of end-over-end rotation, normally 2*pi
    loop_progress_weight: float = 0.0  # reward per rad, capped at loop_target
    loop_overshoot_cost: float = 0.0  # cost per rad beyond loop_target
    loop_bonus: float = 0.0  # timeout bonus, scaled by how upright the completed loop finishes
    loop_finish_rate_cost: float = 0.0  # timeout cost per rad/s after completing the loop
    loop_rate_cost: float = 0.0  # progress-weighted squared end-over-end rate cost per second


@wp.struct
class RewardParams:
    gamma: float
    distance_weight: float
    distance_saturation: float
    horizontal_distance_weight: float
    horizontal_distance_height: float
    speed_weight: float
    vertical_speed_weight: float
    horizontal_speed_weight: float
    touchdown_horizontal_speed_weight: float
    horizontal_speed_height: float
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
    crash_excess_scale: float
    tumble_penalty: float
    out_of_bounds_penalty: float
    timeout_penalty: float
    roll_travel_cost: float
    horizontal_rotation_cost: float
    sideslip_cost: float
    reflip_cost: float
    loop_target: float
    loop_progress_weight: float
    loop_overshoot_cost: float
    loop_bonus: float
    loop_finish_rate_cost: float
    loop_rate_cost: float


def make_reward_params(config: RewardConfig) -> RewardParams:
    import math

    for field in fields(config):
        value = getattr(config, field.name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"reward {field.name} must be finite and nonnegative")
    if (
        config.gamma > 1
        or config.tilt_height <= 0
        or config.horizontal_distance_height <= 0
        or config.horizontal_speed_height <= 0
    ):
        raise ValueError("gamma must be <= 1 and shaping fade heights must be positive")
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
    excess_vertical_speed = wp.max(-b.vel[2] - safe_speed, 0.0)
    axis_z = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))[2]
    tilt = wp.acos(wp.clamp(axis_z, -1.0, 1.0))
    fade = wp.clamp(1.0 - height / r.tilt_height, 0.0, 1.0)
    distance = wp.length(b.pos)
    if r.distance_saturation > 0.0:
        distance = distance / (1.0 + distance / r.distance_saturation)
    horizontal_distance = wp.length(wp.vec2(b.pos[0], b.pos[1]))
    horizontal_distance_fade = wp.exp(-height / r.horizontal_distance_height)
    horizontal_speed = wp.length(wp.vec2(b.vel[0], b.vel[1]))
    horizontal_speed_fade = wp.exp(-height / r.horizontal_speed_height)
    return wp.vec3(
        -r.distance_weight * distance
        - r.horizontal_distance_weight * horizontal_distance * horizontal_distance_fade,
        -r.speed_weight * excess_speed
        - r.vertical_speed_weight * excess_vertical_speed
        - r.horizontal_speed_weight * horizontal_speed
        - r.touchdown_horizontal_speed_weight * horizontal_speed * horizontal_speed_fade,
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
        return -(r.crash_penalty + r.crash_excess_scale * wp.log(1.0 + excess))
    if status == TUMBLED:
        return -r.tumble_penalty
    if status == OUT_OF_BOUNDS:
        return -r.out_of_bounds_penalty
    if status == TIMEOUT:
        if r.loop_target > 0.0:
            a = t.attitude[i]
            if a.horizontal_travel >= r.loop_target:
                axis_z = wp.quat_rotate(t.body[i].rot, wp.vec3(1.0, 0.0, 0.0))[2]
                # No bonus while sideways or inverted; full bonus when upright.
                upright = wp.clamp(axis_z, 0.0, 1.0)
                return r.loop_bonus * upright - r.loop_finish_rate_cost * wp.length(
                    wp.vec2(t.body[i].omega[1], t.body[i].omega[2])
                )
            return 0.0
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
    attitude_before: wp.array(dtype=Attitude),
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
    a = state.attitude[i]
    done = terminal.done[i] != 0
    phi = wp.vec3(0.0)
    terminal_reward[i] = 0.0
    terminated[i] = 0
    truncated[i] = 0
    if done:
        b = terminal.body[i]
        tanks = terminal.tanks[i]
        time = terminal.time[i]
        a = terminal.attitude[i]
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
    prev = attitude_before[i]
    terms[i, 8] = -r.roll_travel_cost * (a.roll_travel - prev.roll_travel)
    terms[i, 9] = -r.horizontal_rotation_cost * (a.horizontal_travel - prev.horizontal_travel)
    terms[i, 10] = -r.sideslip_cost * (a.sideslip - prev.sideslip)
    terms[i, 11] = -r.reflip_cost * float(a.reflips - prev.reflips)
    loop_before = wp.min(prev.horizontal_travel, r.loop_target)
    loop_after = wp.min(a.horizontal_travel, r.loop_target)
    if done:
        loop_after = 0.0
    # Potential shaping: progress is useful as a dense learning signal but cannot be exploited by
    # earning it early and paying overshoot costs later under discounting.
    terms[i, 12] = r.loop_progress_weight * (r.gamma * loop_after - loop_before)
    overshoot_before = wp.max(prev.horizontal_travel - r.loop_target, 0.0)
    overshoot_after = wp.max(a.horizontal_travel - r.loop_target, 0.0)
    terms[i, 13] = -r.loop_overshoot_cost * (overshoot_after - overshoot_before)
    loop_fraction = 0.0
    if r.loop_target > 0.0:
        loop_fraction = wp.min(a.horizontal_travel / r.loop_target, 1.0)
    transverse_rate_sq = b.omega[1] * b.omega[1] + b.omega[2] * b.omega[2]
    terms[i, 14] = -r.loop_rate_cost * loop_fraction * transverse_rate_sq * dt
    total = 0.0
    for j in range(wp.static(len(REWARD_TERMS))):
        total += terms[i, j]
    reward[i] = total
