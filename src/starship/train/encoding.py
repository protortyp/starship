"""Observation encoding: simulator state to normalized policy inputs, computed on the Warp device.

Features, in order:
    0  horizontal offset from the pad center   x / position_scale
    1  altitude of the rocket's lowest point   h / position_scale
    2  horizontal velocity                     vx / velocity_scale
    3  vertical velocity                       vy / velocity_scale
    4  sin(theta)
    5  cos(theta)                              sin/cos avoid the wrap-around jump at +/-pi
    6  angular velocity                        omega / angular_rate_scale
"""

from dataclasses import dataclass

import warp as wp

from starship.sim.config import TIMEOUT, Params
from starship.sim.physics import Terminal, lowest_point

OBS_DIM = 7

STATUS_TIMEOUT = wp.constant(TIMEOUT)


@dataclass(frozen=True)
class ObsConfig:
    """Scales chosen so the base scenario's features stay roughly within [-3, 3]."""

    position_scale: float = 1000.0
    velocity_scale: float = 50.0
    angular_rate_scale: float = 1.0


@wp.struct
class ObsScales:
    position: float
    velocity: float
    angular_rate: float


def make_obs_scales(cfg: ObsConfig) -> ObsScales:
    scales = ObsScales()
    scales.position = cfg.position_scale
    scales.velocity = cfg.velocity_scale
    scales.angular_rate = cfg.angular_rate_scale
    return scales


@wp.func
def encode(
    p: Params,
    scales: ObsScales,
    pos: wp.vec2,
    vel: wp.vec2,
    theta: float,
    omega: float,
    obs: wp.array2d(dtype=float),
    tid: int,
):
    obs[tid, 0] = pos[0] / scales.position
    obs[tid, 1] = lowest_point(pos[1], theta, p) / scales.position
    obs[tid, 2] = vel[0] / scales.velocity
    obs[tid, 3] = vel[1] / scales.velocity
    obs[tid, 4] = wp.sin(theta)
    obs[tid, 5] = wp.cos(theta)
    obs[tid, 6] = omega / scales.angular_rate


@wp.kernel
def observe_kernel(
    p: Params,
    scales: ObsScales,
    pos: wp.array(dtype=wp.vec2),
    vel: wp.array(dtype=wp.vec2),
    theta: wp.array(dtype=float),
    omega: wp.array(dtype=float),
    obs: wp.array2d(dtype=float),
):
    tid = wp.tid()
    encode(p, scales, pos[tid], vel[tid], theta[tid], omega[tid], obs, tid)


@wp.kernel
def episode_end_kernel(
    p: Params,
    scales: ObsScales,
    term: Terminal,
    final_obs: wp.array2d(dtype=float),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
):
    """Encode the final observation of envs that just finished and split termination from
    truncation: a timeout cuts off an episode that could have continued, so value bootstrapping
    should use `final_obs` there, while touchdown and out-of-bounds are true terminal states."""
    tid = wp.tid()
    terminated[tid] = 0
    truncated[tid] = 0
    if term.done[tid] == 0:
        return
    encode(
        p, scales, term.pos[tid], term.vel[tid], term.theta[tid], term.omega[tid], final_obs, tid
    )
    if term.status[tid] == STATUS_TIMEOUT:
        truncated[tid] = 1
    else:
        terminated[tid] = 1
