"""Heading-invariant observations in the vehicle frame; scales are SI denominators."""

import math
from dataclasses import dataclass

import warp as wp

from starship.sim.atmosphere import atmosphere
from starship.sim.catch import (
    catch_center,
    flap_axes,
    flap_center,
    flap_half_size,
    pin,
    pin_error,
    projected_radius,
)
from starship.sim.mass import mass_properties
from starship.sim.physics import Params, State
from starship.sim.propulsion import EngineState, FlapState
from starship.sim.rigid import Body

LEGACY_OBS_DIM = 24
OBS_DIM = 32
CATCH_OBS_DIM = 38

# Policy-frame order, including port/starboard reflection when enabled.
OBSERVATION_NAMES = (
    "up_body_1",
    "up_body_2",
    "up_body_3",
    "velocity_body_1",
    "velocity_body_2",
    "velocity_body_3",
    "target_body_1",
    "target_body_2",
    "target_body_3",
    "angular_rate_body_1",
    "angular_rate_body_2",
    "angular_rate_body_3",
    "altitude",
    "flap_aft_starboard",
    "flap_aft_port",
    "flap_fwd_starboard",
    "flap_fwd_port",
    "spool_0",
    "spool_1",
    "spool_2",
    "throttle",
    "gimbal_2",
    "gimbal_3",
    "fuel_fraction",
    "log_altitude",
    "mach",
    "dynamic_pressure",
    "log_horizontal_miss",
    "tower_heading_body_1",
    "tower_heading_body_2",
    "tower_heading_body_3",
    "arm_open_fraction",
    "pin_height_port",
    "pin_height_starboard",
    "lower_fin_clearance",
    "pin_horizontal_error",
    "arm_close_time",
    "pin_arrival_time",
)


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
    catch_geometry: bool = False


@wp.struct
class ObsScales:
    catch_geometry: int
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
        if name == "catch_geometry":
            scales.catch_geometry = int(config.catch_geometry)
            continue
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
    p: Params,
    b: Body,
    arm_angle: float,
    e: EngineState,
    f: FlapState,
    tanks: wp.vec4,
    landing_load: float,
    mirrored: int,
    scales: ObsScales,
    obs: wp.array2d(dtype=float),
    i: int,
):
    if scales.catch_geometry != 0:
        for j in range(OBS_DIM, CATCH_OBS_DIM):
            obs[i, j] = 0.0
        if p.catcher.enabled != 0:
            com = mass_properties(p.mass, tanks).com
            c = p.catcher
            side = -1.0
            if mirrored != 0:
                side = 1.0
            obs[i, 32] = wp.clamp((pin(c, b, com, side)[2] - c.height) / 50.0, -10.0, 10.0)
            obs[i, 33] = wp.clamp((pin(c, b, com, -side)[2] - c.height) / 50.0, -10.0, 10.0)
            top = -1.0e10
            for j in range(2):
                center = flap_center(p.aero, b, com, f.angle, j)
                extent = projected_radius(
                    flap_axes(p.aero, b, f.angle, j),
                    flap_half_size(c, p.aero, j),
                    wp.vec3(0.0, 0.0, 1.0),
                )
                top = wp.max(top, center[2] + extent)
            obs[i, 34] = wp.clamp((c.height - 1.0 - top) / 50.0, -10.0, 10.0)
            obs[i, 35] = wp.min(pin_error(c, b, com) / 10.0, 10.0)
            if c.moving != 0:
                obs[i, 36] = arm_angle / (2.0 * c.angular_speed)
            eta = -1.0
            for j in range(2):
                sign = float(2 * j - 1)
                lever = wp.vec3(c.station - com, sign * c.pin_radius, 0.0)
                vz = (b.vel + wp.quat_rotate(b.rot, wp.cross(b.omega, lever)))[2]
                height = pin(c, b, com, sign)[2] - c.height
                if vz < -1.0e-6 and height >= 0.0:
                    arrival = height / -vz
                    if eta < 0.0 or arrival < eta:
                        eta = arrival
            obs[i, 37] = wp.min(eta, 30.0) / 10.0
    obs[i, 31] = 0.0
    if p.catcher.enabled != 0 and p.catcher.moving != 0:
        obs[i, 31] = arm_angle / p.catcher.open_angle
    up = wp.quat_rotate_inv(b.rot, wp.vec3(0.0, 0.0, 1.0))
    vel = wp.quat_rotate_inv(b.rot, b.vel)
    relative = b.pos
    heading = wp.vec3(0.0)
    if p.catcher.enabled != 0:
        relative = catch_center(p.catcher, b, mass_properties(p.mass, tanks).com) - wp.vec3(
            0.0, 0.0, p.catcher.height
        )
        # World +x points away from the tower and is invariant under port/starboard reflection.
        heading = wp.quat_rotate_inv(b.rot, wp.vec3(1.0, 0.0, 0.0))
    pad = wp.quat_rotate_inv(b.rot, -relative)
    distance = wp.length(relative)
    if distance > scales.position_clip:
        pad = pad * (scales.position_clip / distance)
    omega = b.omega
    flaps = f.angle
    spool = e.spool
    gimbal2 = wp.dot(e.gimbal2, wp.vec3(1.0 / 3.0))
    if mirrored != 0:
        heading[1] = -heading[1]
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
        obs[i, 28 + j] = heading[j]
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
    p: Params,
    state: State,
    landing_load: wp.array(dtype=float),
    mirrored: wp.array(dtype=int),
    scales: ObsScales,
    obs: wp.array2d(dtype=float),
):
    i = wp.tid()
    encode(
        p,
        state.body[i],
        state.arm_angle[i],
        state.engine[i],
        state.flaps[i],
        state.tanks[i],
        landing_load[i],
        mirrored[i],
        scales,
        obs,
        i,
    )
