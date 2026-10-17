"""Vehicle and simulation parameters. Values and sources: docs/starship.md (V2 ship, Raptor 2).

Frames and conventions (shared by every module):
    world   z up, x/y horizontal, pad centred at the origin, SI units.
    body    b1 along the axis from the engines to the nose, b2 to starboard (towards the +b2 aft
            flap), b3 = b1 x b2 towards the belly (windward tile side in the belly-flop).
    station x   distance along b1 from the engine plane (base of the skirt, nozzle exits).
    azimuth phi angle around b1 measured from +b3 (belly) towards +b2, so a point on the hull at
            radius r and azimuth phi sits at r*(cos(phi) b3 + sin(phi) b2).
    attitude    unit quaternion rotating body vectors into the world frame.
    angular velocity  body frame.
"""

import math
from dataclasses import dataclass, field

G0 = 9.80665


@dataclass
class HullConfig:
    length: float = 52.1
    diameter: float = 9.0
    nose_length: float = 14.0  # tangent ogive from station length - nose_length to the tip
    skirt_length: float = 7.3


@dataclass
class MassConfig:
    """Dry structure as lumped components; positions are stations (m)."""

    dry_mass: float = 160_000.0
    engines_mass: float = 6 * 1630.0  # 3 sea-level + 3 vacuum Raptors
    engines_station: float = 2.5
    flaps_mass: float = 4 * 2_500.0
    aft_mass: float = 30_000.0  # skirt, thrust structure and aft dome, spread over the skirt
    # Remaining dry mass is spread over the hull: this fraction as a uniform cylinder shell over
    # the tank section, the rest in the nose (payload bay, header tanks, flap hinges).
    shell_fraction: float = 0.75


@dataclass
class TankConfig:
    """Cylindrical tank spanning [bottom, top] stations; liquid settles towards the bottom."""

    bottom: float
    top: float
    capacity: float  # kg of propellant when full
    density: float  # kg/m^3


def default_tanks() -> dict[str, TankConfig]:
    # Header tanks feed the landing burn; main tanks hold the residuals after the ascent.
    return {
        "main_lox": TankConfig(bottom=7.3, top=23.5, capacity=1_170_000.0, density=1141.0),
        "main_ch4": TankConfig(bottom=23.5, top=36.0, capacity=330_000.0, density=423.0),
        "header_lox": TankConfig(bottom=48.5, top=50.5, capacity=18_000.0, density=1141.0),
        "header_ch4": TankConfig(bottom=45.0, top=47.0, capacity=5_000.0, density=423.0),
    }


@dataclass
class EngineConfig:
    """Sea-level Raptor 2; three of them gimbal for the landing burn."""

    count: int = 3
    vacuum_thrust: float = 2.39e6  # N at 100 % throttle; ambient pressure subtracts p_a * exit_area
    exit_area: float = 1.33
    vacuum_isp: float = 347.0
    mixture_ratio: float = 3.6  # oxidizer / fuel mass
    min_throttle: float = 0.4
    throttle_rate: float = 0.4  # throttle fraction per second
    gimbal_limit: float = math.radians(15.0)
    gimbal_rate: float = math.radians(20.0)
    gimbal_lag: float = 0.1  # s, first-order lag in front of the rate limit
    ignition_delay: float = 2.0  # s from the on command to full thrust (linear ramp)
    shutdown_time: float = 0.3  # s ramp down after the off command
    pivot_station: float = 3.1
    pivot_radius: float = 1.15
    azimuths: tuple[float, ...] = (0.0, 2 * math.pi / 3, 4 * math.pi / 3)


@dataclass
class FlapConfig:
    """One flat-plate flap. The hinge runs along b1 (aft) or along the nose surface (forward); at
    angle 0 the plate lies flat, extending outwards in the plane of the hinge and the axis."""

    azimuth: float  # rad, where the hinge sits around the hull
    hinge_bottom: float  # station of the hinge's lower end
    hinge_top: float
    area: float
    span: float  # outboard extent of the plate from the hull surface
    cant: float = 0.0  # rad, hinge tilt towards the axis (forward flaps follow the nose taper)


def default_flaps() -> list[FlapConfig]:
    aft = {"hinge_bottom": 1.0, "hinge_top": 12.0, "area": 40.0, "span": 4.5}
    fwd = {
        "hinge_bottom": 39.0,
        "hinge_top": 47.0,
        "area": 18.0,
        "span": 3.0,
        "cant": math.radians(12.0),
    }
    # Order: aft starboard, aft port, forward starboard, forward port. Forward flaps sit 20 deg
    # towards the leeward side (140 deg apart).
    return [
        FlapConfig(azimuth=math.radians(90.0), **aft),
        FlapConfig(azimuth=math.radians(-90.0), **aft),
        FlapConfig(azimuth=math.radians(110.0), **fwd),
        FlapConfig(azimuth=math.radians(-110.0), **fwd),
    ]


@dataclass
class FlapActuatorConfig:
    min_angle: float = 0.0  # flat, fully exposed
    max_angle: float = math.radians(85.0)  # folded against the hull
    rate: float = math.radians(25.0)
    lag: float = 0.15


@dataclass
class RcsConfig:
    """Vent thrusters used for roll: a torque about b1 of +-thrust * arm per pair."""

    thrust: float = 2_000.0
    arm: float = 4.5
    pairs: int = 2


@dataclass
class AeroConfig:
    strips: int = 32  # hull slices for the crossflow model
    crossflow_scale: float = 0.63  # finite-length factor eta (subsonic)
    axial_nose: float = 0.4  # C_A at 0 deg AoA (nose first), reference area = cross-section
    axial_tail: float = 0.9  # C_A at 180 deg AoA (tail first)


@dataclass
class VehicleConfig:
    hull: HullConfig = field(default_factory=HullConfig)
    mass: MassConfig = field(default_factory=MassConfig)
    tanks: dict[str, TankConfig] = field(default_factory=default_tanks)
    engine: EngineConfig = field(default_factory=EngineConfig)
    flaps: list[FlapConfig] = field(default_factory=default_flaps)
    flap_actuator: FlapActuatorConfig = field(default_factory=FlapActuatorConfig)
    rcs: RcsConfig = field(default_factory=RcsConfig)
    aero: AeroConfig = field(default_factory=AeroConfig)


@dataclass
class SpawnConfig:
    """Initial state, each component uniform in value +- noise. Attitude is given as the angle
    of b1 from vertical (pitch, 90 deg = belly-flop) about a heading; belly faces down."""

    altitude: float = 3000.0
    altitude_noise: float = 0.0
    offset: float = 0.0  # horizontal distance from the pad
    offset_noise: float = 0.0
    vertical_speed: float = -85.0
    vertical_speed_noise: float = 0.0
    horizontal_speed: float = 0.0
    horizontal_speed_noise: float = 0.0
    pitch: float = math.radians(90.0)
    pitch_noise: float = 0.0
    angular_rate_noise: float = 0.0
    flap_angle: float = math.radians(45.0)
    header_fill: float = 1.0  # fraction of header tank capacity
    main_fill: float = 0.01  # residuals in the main tanks


@dataclass
class LandingConfig:
    pad_radius: float = 15.0
    max_vertical_speed: float = 2.0
    max_horizontal_speed: float = 1.0
    max_tilt: float = math.radians(5.0)
    max_angular_rate: float = math.radians(5.0)


@dataclass
class SimConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    landing: LandingConfig = field(default_factory=LandingConfig)
    dt: float = 0.01
    substeps: int = 10  # physics steps per control step
    max_time: float = 180.0
    max_altitude: float = 25_000.0
    max_offset: float = 10_000.0
