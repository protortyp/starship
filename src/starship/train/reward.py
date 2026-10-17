"""Stage 1 reward: learn to land.

reward = gamma * potential(s') - potential(s) - fuel_cost * fuel burned - time_cost * dt
         + terminal reward

The potential scores how close the rocket is to a good touchdown: near the pad, upright near the
ground, and not faster than a safe descent profile v_safe(h) = sqrt(2 a h) + v_touch. Only speed in
excess of the profile counts: a plain speed term rewards braking at any altitude, and fading it in
with altitude penalizes descending at all, and PPO's short advantage horizon follows either one
greedily into hovering until the fuel runs out.

With gamma equal to the learner's discount and a zero potential after the episode ends, the
discounted shaping sums to -potential(start) for every trajectory, so it guides learning without
changing which behavior is optimal; only the terminal reward decides that. A nonzero terminal
potential would instead pay (1 - gamma) * |potential| per step for staying alive.

The fuel cost makes burning propellant to stay airborne expensive. Without it, a policy that cannot
land yet prefers to float and crash late, because discounting shrinks a later crash penalty.
"""

from dataclasses import dataclass

import warp as wp

from starship.sim.config import CRASHED, FLYING, LANDED, OUT_OF_BOUNDS, TIMEOUT, Params
from starship.sim.physics import State, Terminal, lowest_point

STATUS_FLYING = wp.constant(FLYING)
STATUS_LANDED = wp.constant(LANDED)
STATUS_CRASHED = wp.constant(CRASHED)
STATUS_OUT_OF_BOUNDS = wp.constant(OUT_OF_BOUNDS)
STATUS_TIMEOUT = wp.constant(TIMEOUT)


# Columns of the per-step reward breakdown; the terminal reward is kept separately.
REWARD_TERMS = ("distance", "speed", "lateral", "tilt", "fuel", "time")


@dataclass
class RewardConfig:
    # Must match the learner's discount factor for the shaping to be policy-invariant.
    gamma: float = 0.999
    # Multiplier on rewards handed to the learner, keeping returns near the O(1-10) range that
    # PPO's value clipping and gradient clipping assume. Env metrics stay in unscaled units.
    learner_scale: float = 1.0
    distance_weight: float = 10.0  # per km to the pad center
    speed_weight: float = 0.2  # per m/s above the safe descent profile
    # Safe descent profile: braking at this deceleration from v_safe(h) stops at v_touch.
    profile_decel: float = 3.8  # m/s^2, about half the engine's net deceleration
    profile_touchdown_speed: float = 2.0  # m/s
    # Horizontal distance from the pad, faded in near the ground where touching down off the
    # pad (beyond its half-width) counts as a crash.
    lateral_weight: float = 0.05  # per m
    lateral_fade_altitude: float = 500.0  # m
    # Partial landing credit for touching down on the pad slightly too hard: the landed reward
    # scaled by (1 - impact speed / soft_touchdown_speed). Smooths the cliff between a 4 m/s
    # landing and a 5 m/s crash, which exploration noise falls off. 0 disables it.
    soft_touchdown_speed: float = 0.0  # m/s
    tilt_weight: float = 10.0  # per rad, faded in near the ground
    tilt_fade_altitude: float = 200.0  # m; tilt weight is exp(-altitude / this)
    landed_reward: float = 100.0
    fuel_bonus: float = 50.0  # on landing, times the fraction of fuel left
    crash_penalty: float = 10.0
    crash_speed_weight: float = 1.0  # extra penalty per m/s of impact speed
    out_of_bounds_penalty: float = 100.0
    # Without this, hovering until the time limit beats any crash.
    timeout_penalty: float = 50.0
    fuel_cost: float = 0.002  # per kg burned; a full 30 t tank is worth 60
    time_cost: float = 0.0  # per second of flight; rewards landing quickly


@wp.struct
class RewardParams:
    gamma: float
    distance_weight: float
    speed_weight: float
    profile_decel: float
    profile_touchdown_speed: float
    lateral_weight: float
    lateral_fade_altitude: float
    soft_touchdown_speed: float
    tilt_weight: float
    tilt_fade_altitude: float
    landed_reward: float
    fuel_bonus: float
    crash_penalty: float
    crash_speed_weight: float
    out_of_bounds_penalty: float
    timeout_penalty: float
    fuel_cost: float
    time_cost: float


def make_reward_params(cfg: RewardConfig) -> RewardParams:
    rp = RewardParams()
    for name in RewardParams.vars:
        setattr(rp, name, getattr(cfg, name))
    return rp


@wp.func
def potential(p: Params, rp: RewardParams, pos: wp.vec2, vel: wp.vec2, theta: float) -> wp.vec4:
    """Potential split into its (distance, speed, lateral, tilt) terms; the potential is the sum."""
    h = wp.max(lowest_point(pos[1], theta, p), 0.0)
    distance = wp.sqrt(pos[0] * pos[0] + h * h)
    safe_speed = wp.sqrt(2.0 * rp.profile_decel * h) + rp.profile_touchdown_speed
    return -wp.vec4(
        rp.distance_weight * distance / 1000.0,
        rp.speed_weight * wp.max(wp.length(vel) - safe_speed, 0.0),
        rp.lateral_weight * wp.abs(pos[0]) * wp.exp(-h / rp.lateral_fade_altitude),
        rp.tilt_weight * wp.abs(theta) * wp.exp(-h / rp.tilt_fade_altitude),
    )


@wp.func
def terminal_reward(
    p: Params, rp: RewardParams, status: int, pos: wp.vec2, vel: wp.vec2, fuel: float
) -> float:
    if status == STATUS_LANDED:
        bonus = 0.0
        if p.fuel_mass > 0.0:
            bonus = rp.fuel_bonus * fuel / p.fuel_mass
        return rp.landed_reward + bonus
    if status == STATUS_CRASHED:
        speed = wp.length(vel)
        credit = 0.0
        if rp.soft_touchdown_speed > 0.0 and wp.abs(pos[0]) <= p.pad_half_width:
            credit = rp.landed_reward * wp.max(1.0 - speed / rp.soft_touchdown_speed, 0.0)
        return credit - (rp.crash_penalty + rp.crash_speed_weight * speed)
    if status == STATUS_OUT_OF_BOUNDS:
        return -rp.out_of_bounds_penalty
    if status == STATUS_TIMEOUT:
        return -rp.timeout_penalty
    return 0.0


@wp.kernel
def reward_kernel(
    p: Params,
    rp: RewardParams,
    s: State,
    term: Terminal,
    last_potential: wp.array(dtype=wp.vec4),
    last_fuel: wp.array(dtype=float),
    reward: wp.array(dtype=float),
    terminal: wp.array(dtype=float),
    terms: wp.array2d(dtype=float),
):
    tid = wp.tid()
    current = potential(p, rp, s.pos[tid], s.vel[tid], s.theta[tid])
    shaping = wp.vec4(0.0)
    fuel = 0.0
    time = 0.0
    end = 0.0
    if term.done[tid] != 0:
        end = terminal_reward(p, rp, term.status[tid], term.pos[tid], term.vel[tid], term.fuel[tid])
        shaping = -last_potential[tid]
        fuel = -rp.fuel_cost * (last_fuel[tid] - term.fuel[tid])
        time = -rp.time_cost * p.dt * float(p.substeps)
    elif s.status[tid] == STATUS_FLYING:
        shaping = rp.gamma * current - last_potential[tid]
        fuel = -rp.fuel_cost * (last_fuel[tid] - s.fuel[tid])
        time = -rp.time_cost * p.dt * float(p.substeps)
    for i in range(4):
        terms[tid, i] = shaping[i]
    terms[tid, 4] = fuel
    terms[tid, 5] = time
    terminal[tid] = end
    reward[tid] = shaping[0] + shaping[1] + shaping[2] + shaping[3] + fuel + time + end
    # After an auto-reset `current` and the fuel already belong to the respawned state.
    last_potential[tid] = current
    last_fuel[tid] = s.fuel[tid]


@wp.kernel
def reset_potential_kernel(
    p: Params,
    rp: RewardParams,
    s: State,
    last_potential: wp.array(dtype=wp.vec4),
    last_fuel: wp.array(dtype=float),
):
    tid = wp.tid()
    last_potential[tid] = potential(p, rp, s.pos[tid], s.vel[tid], s.theta[tid])
    last_fuel[tid] = s.fuel[tid]
