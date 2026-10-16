"""Simulation configuration.

Conventions (SI units throughout):
    - World frame: x to the right, y up, landing pad centered at the origin on the ground (y = 0).
    - theta: rocket attitude, counter-clockwise from vertical. theta = 0 is upright (nose up),
      theta = +pi/2 is lying flat with the nose pointing to the left (-x).
    - Body axis a = (-sin theta, cos theta) points from tail to nose.
      Body normal n = (cos theta, sin theta) points to the rocket's right side when upright.

Default vehicle: Starship upper stage during landing (public figures, approximate):
    - 50 m long, 9 m diameter, ~100 t dry, ~30 t of landing propellant in the header tanks.
    - Main engine: one Raptor 2 at sea level, 2.26 MN, Isp 327 s (~705 kg/s); T/W ~1.8 at spawn.
    - Belly-flop terminal velocity at sea level: sqrt(2 m g / (rho Cd_n L D)) ~ 82 m/s, matching
      public estimates of 80-85 m/s; nose-first it is ~160 m/s.
    - Cancelling ~82 m/s takes an ~11 s burn, ~7.6 t of propellant, so 30 t is ~4x the minimum.
    - Side thrusters stand in for the real flip (engine gimbal and flaps): sized to rotate the
      vehicle by 90 deg in ~5 s, drawing from the same tank.
"""

import math
from dataclasses import dataclass, field, replace

import warp as wp

G0 = 9.80665

# Action bitmask: each policy action is an int in [0, 8) combining these thruster flags.
MAIN = 1
LEFT = 2
RIGHT = 4
NUM_ACTIONS = 8

# Episode status codes.
FLYING = 0
LANDED = 1
CRASHED = 2
OUT_OF_BOUNDS = 3
TIMEOUT = 4

STATUS_NAMES = {
    FLYING: "flying",
    LANDED: "landed",
    CRASHED: "crashed",
    OUT_OF_BOUNDS: "out_of_bounds",
    TIMEOUT: "timeout",
}


@dataclass(frozen=True)
class RocketConfig:
    dry_mass: float = 100_000.0
    fuel_mass: float = 30_000.0
    length: float = 50.0
    diameter: float = 9.0
    main_thrust: float = 2.26e6
    main_isp: float = 327.0
    side_thrust: float = 2.8e5
    side_isp: float = 300.0
    # Side thrusters sit at the nose; distance measured from the center of mass along the axis.
    side_arm: float = 25.0
    # Crossflow drag model: separate coefficients for flow along and across the body axis.
    drag_coeff_axial: float = 0.75
    drag_coeff_normal: float = 0.7

    @classmethod
    def small(cls) -> "RocketConfig":
        """A 30 m vehicle with the same thrust-to-weight and fuel margin as the default."""
        return cls(
            dry_mass=52_000.0,
            fuel_mass=8_000.0,
            length=30.0,
            main_thrust=1.18e6,
            side_thrust=1.2e5,
            side_arm=15.0,
        )

    @property
    def mass(self) -> float:
        """Fully fueled mass."""
        return self.dry_mass + self.fuel_mass

    @property
    def inertia_per_kg(self) -> float:
        """Solid cylinder about a transverse axis; fuel is treated as uniformly distributed."""
        r = 0.5 * self.diameter
        return (3.0 * r * r + self.length * self.length) / 12.0

    @property
    def inertia(self) -> float:
        return self.mass * self.inertia_per_kg

    @property
    def main_mass_flow(self) -> float:
        return self.main_thrust / (self.main_isp * G0)

    @property
    def side_mass_flow(self) -> float:
        return self.side_thrust / (self.side_isp * G0)

    @property
    def area_axial(self) -> float:
        return math.pi * (0.5 * self.diameter) ** 2

    @property
    def area_normal(self) -> float:
        return self.length * self.diameter


@dataclass(frozen=True)
class WorldConfig:
    gravity: float = G0
    # Sea-level air density; set to 0 to simulate vacuum.
    sea_level_density: float = 1.225


@dataclass(frozen=True)
class SpawnConfig:
    """Initial state: each component is drawn uniformly from value +/- noise."""

    x: float = 0.0
    y: float = 3000.0
    vx: float = 0.0
    vy: float = -90.0
    theta: float = 0.5 * math.pi
    omega: float = 0.0
    x_noise: float = 0.0
    y_noise: float = 0.0
    vx_noise: float = 0.0
    vy_noise: float = 0.0
    theta_noise: float = 0.0
    omega_noise: float = 0.0

    def with_noise(self, scale: float) -> "SpawnConfig":
        """Preset randomization: at scale 1, +/-300 m offset, +/-10 m/s, +/-0.2 rad tilt."""
        return replace(
            self,
            x_noise=300.0 * scale,
            vx_noise=10.0 * scale,
            vy_noise=10.0 * scale,
            theta_noise=0.2 * scale,
        )


@dataclass(frozen=True)
class LandingConfig:
    """Touchdown is a successful landing only if every limit is satisfied."""

    pad_half_width: float = 25.0
    max_vertical_speed: float = 4.0
    max_horizontal_speed: float = 2.0
    max_tilt: float = math.radians(10.0)
    max_angular_rate: float = 0.2


@dataclass(frozen=True)
class SimConfig:
    rocket: RocketConfig = field(default_factory=RocketConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    landing: LandingConfig = field(default_factory=LandingConfig)
    # Physics timestep; each policy step advances `substeps` physics steps (default 20 Hz policy).
    dt: float = 0.01
    substeps: int = 5
    max_time: float = 120.0
    max_abs_x: float = 5000.0
    max_y: float = 6000.0

    @property
    def policy_dt(self) -> float:
        return self.dt * self.substeps

    @property
    def max_steps(self) -> int:
        return math.ceil(self.max_time / self.policy_dt)


@wp.struct
class Params:
    dry_mass: float
    fuel_mass: float
    inertia_per_kg: float
    main_mass_flow: float
    side_mass_flow: float
    length: float
    diameter: float
    main_thrust: float
    side_thrust: float
    side_arm: float
    drag_coeff_axial: float
    drag_coeff_normal: float
    area_axial: float
    area_normal: float

    gravity: float
    sea_level_density: float

    spawn_pos: wp.vec2
    spawn_vel: wp.vec2
    spawn_theta: float
    spawn_omega: float
    spawn_pos_noise: wp.vec2
    spawn_vel_noise: wp.vec2
    spawn_theta_noise: float
    spawn_omega_noise: float

    pad_half_width: float
    max_vertical_speed: float
    max_horizontal_speed: float
    max_tilt: float
    max_angular_rate: float

    dt: float
    substeps: int
    max_steps: int
    max_abs_x: float
    max_y: float


def make_params(cfg: SimConfig) -> Params:
    r, w, s, land = cfg.rocket, cfg.world, cfg.spawn, cfg.landing
    p = Params()
    p.dry_mass = r.dry_mass
    p.fuel_mass = r.fuel_mass
    p.inertia_per_kg = r.inertia_per_kg
    p.main_mass_flow = r.main_mass_flow
    p.side_mass_flow = r.side_mass_flow
    p.length = r.length
    p.diameter = r.diameter
    p.main_thrust = r.main_thrust
    p.side_thrust = r.side_thrust
    p.side_arm = r.side_arm
    p.drag_coeff_axial = r.drag_coeff_axial
    p.drag_coeff_normal = r.drag_coeff_normal
    p.area_axial = r.area_axial
    p.area_normal = r.area_normal

    p.gravity = w.gravity
    p.sea_level_density = w.sea_level_density

    p.spawn_pos = wp.vec2(s.x, s.y)
    p.spawn_vel = wp.vec2(s.vx, s.vy)
    p.spawn_theta = s.theta
    p.spawn_omega = s.omega
    p.spawn_pos_noise = wp.vec2(s.x_noise, s.y_noise)
    p.spawn_vel_noise = wp.vec2(s.vx_noise, s.vy_noise)
    p.spawn_theta_noise = s.theta_noise
    p.spawn_omega_noise = s.omega_noise

    p.pad_half_width = land.pad_half_width
    p.max_vertical_speed = land.max_vertical_speed
    p.max_horizontal_speed = land.max_horizontal_speed
    p.max_tilt = land.max_tilt
    p.max_angular_rate = land.max_angular_rate

    p.dt = cfg.dt
    p.substeps = cfg.substeps
    p.max_steps = cfg.max_steps
    p.max_abs_x = cfg.max_abs_x
    p.max_y = cfg.max_y
    return p
