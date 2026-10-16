"""Stage 1 reward: learn to land.

reward = gamma * potential(s') - potential(s) - fuel_cost * fuel burned + terminal reward

The potential scores how close the rocket is to a good touchdown: near the pad, slow, and (only
near the ground) upright. With gamma equal to the learner's discount and a zero potential after the
episode ends, the discounted shaping sums to -potential(start) for every trajectory, so it guides
learning without changing which behavior is optimal; only the terminal reward decides that. A
nonzero terminal potential would instead pay (1 - gamma) * |potential| per step for staying alive.

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


@dataclass
class RewardConfig:
    # Must match the learner's discount factor for the shaping to be policy-invariant.
    gamma: float = 0.999
    # Multiplier on rewards handed to the learner, keeping returns near the O(1-10) range that
    # PPO's value clipping and gradient clipping assume. Env metrics stay in unscaled units.
    learner_scale: float = 0.05
    distance_weight: float = 10.0  # per km to the pad center
    speed_weight: float = 0.2  # per m/s
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


@wp.struct
class RewardParams:
    gamma: float
    distance_weight: float
    speed_weight: float
    tilt_weight: float
    tilt_fade_altitude: float
    landed_reward: float
    fuel_bonus: float
    crash_penalty: float
    crash_speed_weight: float
    out_of_bounds_penalty: float
    timeout_penalty: float
    fuel_cost: float


def make_reward_params(cfg: RewardConfig) -> RewardParams:
    rp = RewardParams()
    for name in RewardParams.vars:
        setattr(rp, name, getattr(cfg, name))
    return rp


@wp.func
def potential(p: Params, rp: RewardParams, pos: wp.vec2, vel: wp.vec2, theta: float) -> float:
    h = wp.max(lowest_point(pos[1], theta, p), 0.0)
    distance = wp.sqrt(pos[0] * pos[0] + h * h)
    tilt_fade = wp.exp(-h / rp.tilt_fade_altitude)
    return -(
        rp.distance_weight * distance / 1000.0
        + rp.speed_weight * wp.length(vel)
        + rp.tilt_weight * wp.abs(theta) * tilt_fade
    )


@wp.func
def terminal_reward(p: Params, rp: RewardParams, status: int, vel: wp.vec2, fuel: float) -> float:
    if status == STATUS_LANDED:
        bonus = 0.0
        if p.fuel_mass > 0.0:
            bonus = rp.fuel_bonus * fuel / p.fuel_mass
        return rp.landed_reward + bonus
    if status == STATUS_CRASHED:
        return -(rp.crash_penalty + rp.crash_speed_weight * wp.length(vel))
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
    last_potential: wp.array(dtype=float),
    last_fuel: wp.array(dtype=float),
    reward: wp.array(dtype=float),
    terminal: wp.array(dtype=float),
):
    tid = wp.tid()
    current = potential(p, rp, s.pos[tid], s.vel[tid], s.theta[tid])
    terminal[tid] = 0.0
    if term.done[tid] != 0:
        end = terminal_reward(p, rp, term.status[tid], term.vel[tid], term.fuel[tid])
        terminal[tid] = end
        burned = last_fuel[tid] - term.fuel[tid]
        reward[tid] = -last_potential[tid] - rp.fuel_cost * burned + end
    elif s.status[tid] == STATUS_FLYING:
        burned = last_fuel[tid] - s.fuel[tid]
        reward[tid] = rp.gamma * current - last_potential[tid] - rp.fuel_cost * burned
    else:
        reward[tid] = 0.0
    # After an auto-reset `current` and the fuel already belong to the respawned state.
    last_potential[tid] = current
    last_fuel[tid] = s.fuel[tid]


@wp.kernel
def reset_potential_kernel(
    p: Params,
    rp: RewardParams,
    s: State,
    last_potential: wp.array(dtype=float),
    last_fuel: wp.array(dtype=float),
):
    tid = wp.tid()
    last_potential[tid] = potential(p, rp, s.pos[tid], s.vel[tid], s.theta[tid])
    last_fuel[tid] = s.fuel[tid]
