"""Heading-invariant observations in the vehicle frame; scales are SI denominators."""

import math
from dataclasses import dataclass

import warp as wp

from starship.sim.atmosphere import atmosphere
from starship.sim.physics import State
from starship.sim.propulsion import EngineState, FlapState
from starship.sim.rigid import Body

LEGACY_OBS_DIM = 24
OBS_DIM = 28


@dataclass
class ObsConfig:
    position: float = 3000.0
    velocity: float = 100.0
    angular_rate: float = math.radians(40.0)
    altitude: float = 3000.0
    flap_angle: float = math.radians(85.0)
    gimbal_angle: float = math.radians(15.0)
    altitude_clip: float = 5000.0
    position_clip: float = 5100.0  # covers the 5 km / 500 m stage-4 spawn envelope
    log_height: float = 1000.0
    log_distance: float = 1000.0
    log_scale: float = 3.0
    dynamic_pressure: float = 5000.0
    mirror: bool = True


@wp.struct
class ObsScales:
    position: float
    velocity: float
    angular_rate: float
    altitude: float
    flap_angle: float
    gimbal_angle: float
    altitude_clip: float
    position_clip: float
    log_height: float
    log_distance: float
    log_scale: float
    dynamic_pressure: float


def make_obs_scales(config: ObsConfig | None = None) -> ObsScales:
    config = config or ObsConfig()
    scales = ObsScales()
    for name in ObsScales.vars:
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"observation scale {name} must be finite and positive")
        setattr(scales, name, value if name.endswith("_clip") else 1.0 / value)
    return scales


@wp.func
def fuel_fraction(tanks: wp.vec4, landing_load: float) -> float:
    # All four tanks feed the landing burn, including the residual main-tank propellant.
    return wp.clamp(wp.dot(tanks, wp.vec4(1.0)) / wp.max(landing_load, 1.0e-6), 0.0, 1.0)


@wp.func
def encode(
    b: Body,
    e: EngineState,
    f: FlapState,
    tanks: wp.vec4,
    landing_load: float,
    mirrored: int,
    scales: ObsScales,
    obs: wp.array2d(dtype=float),
    i: int,
):
    up = wp.quat_rotate_inv(b.rot, wp.vec3(0.0, 0.0, 1.0))
    vel = wp.quat_rotate_inv(b.rot, b.vel)
    pad = wp.quat_rotate_inv(b.rot, -b.pos)
    distance = wp.length(b.pos)
    if distance > scales.position_clip:
        pad = pad * (scales.position_clip / distance)
    omega = b.omega
    flaps = f.angle
    spool = e.spool
    gimbal2 = wp.dot(e.gimbal2, wp.vec3(1.0 / 3.0))
    if mirrored != 0:
        up[1] = -up[1]
        vel[1] = -vel[1]
        pad[1] = -pad[1]
        omega[0] = -omega[0]
        omega[2] = -omega[2]
        flaps = wp.vec4(flaps[1], flaps[0], flaps[3], flaps[2])
        # Engines at +/-120 degrees exchange places under the same reflection.
        spool = wp.vec3(spool[0], spool[2], spool[1])
        gimbal2 = -gimbal2
    for j in range(3):
        obs[i, j] = up[j]
        obs[i, 3 + j] = vel[j] * scales.velocity
        obs[i, 6 + j] = pad[j] * scales.position
        obs[i, 9 + j] = omega[j] * scales.angular_rate
        obs[i, 17 + j] = spool[j]
    obs[i, 12] = wp.min(b.pos[2], scales.altitude_clip) * scales.altitude
    for j in range(4):
        obs[i, 13 + j] = flaps[j] * scales.flap_angle
    obs[i, 20] = wp.dot(e.throttle, wp.vec3(1.0 / 3.0))
    obs[i, 21] = gimbal2 * scales.gimbal_angle
    obs[i, 22] = wp.dot(e.gimbal3, wp.vec3(1.0 / 3.0)) * scales.gimbal_angle
    obs[i, 23] = fuel_fraction(tanks, landing_load)
    air = atmosphere(wp.clamp(b.pos[2], 0.0, 86000.0))
    speed = wp.length(b.vel)
    obs[i, 24] = wp.log(1.0 + wp.max(b.pos[2], 0.0) * scales.log_height) * scales.log_scale
    obs[i, 25] = speed / air.speed_of_sound
    obs[i, 26] = 0.5 * air.density * speed * speed * scales.dynamic_pressure
    miss = wp.length(wp.vec2(b.pos[0], b.pos[1]))
    obs[i, 27] = wp.log(1.0 + miss * scales.log_distance) * scales.log_scale


@wp.kernel
def observe_kernel(
    state: State,
    landing_load: wp.array(dtype=float),
    mirrored: wp.array(dtype=int),
    scales: ObsScales,
    obs: wp.array2d(dtype=float),
):
    i = wp.tid()
    encode(
        state.body[i],
        state.engine[i],
        state.flaps[i],
        state.tanks[i],
        landing_load[i],
        mirrored[i],
        scales,
        obs,
        i,
    )
