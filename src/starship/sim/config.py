"""Simulation configuration.

Conventions (SI units throughout):
    - World frame: x to the right, y up, landing pad centered at the origin on the ground (y = 0).
    - theta: rocket attitude, counter-clockwise from vertical. theta = 0 is upright (nose up),
      theta = +pi/2 is lying flat with the nose pointing to the left (-x).
    - Body axis a = (-sin theta, cos theta) points from tail to nose.
      Body normal n = (cos theta, sin theta) points to the rocket's right side when upright.

Default vehicle sizing (back-of-envelope, roughly a scaled-down Starship upper stage):
    - 60 t landing mass, 30 m long, 9 m diameter.
    - Belly-flop terminal velocity at sea level: sqrt(2 m g / (rho Cd_n L D)) ~ 71 m/s.
    - Nose-first terminal velocity: sqrt(2 m g / (rho Cd_a pi r^2)) ~ 142 m/s.
    - Main engine thrust-to-weight ~ 2, so a full burn decelerates at ~1 g.
    - Side thrusters can flip the vehicle by 90 deg in ~4 s with bang-bang control.
"""

import math
from dataclasses import dataclass, field

import warp as wp

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
    mass: float = 60_000.0
    length: float = 30.0
    diameter: float = 9.0
    main_thrust: float = 1.18e6
    side_thrust: float = 1.2e5
    # Side thrusters sit at the nose; distance measured from the center of mass along the axis.
    side_arm: float = 15.0
    # Crossflow drag model: separate coefficients for flow along and across the body axis.
    drag_coeff_axial: float = 0.75
    drag_coeff_normal: float = 0.7

    @property
    def inertia(self) -> float:
        """Moment of inertia of a solid cylinder about a transverse axis through its center."""
        r = 0.5 * self.diameter
        return self.mass * (3.0 * r * r + self.length * self.length) / 12.0

    @property
    def area_axial(self) -> float:
        return math.pi * (0.5 * self.diameter) ** 2

    @property
    def area_normal(self) -> float:
        return self.length * self.diameter


@dataclass(frozen=True)
class WorldConfig:
    gravity: float = 9.80665
    # Sea-level air density; set to 0 to simulate vacuum.
    sea_level_density: float = 1.225


@dataclass(frozen=True)
class SpawnConfig:
    """Initial state: each component is drawn uniformly from value +/- noise."""

    x: float = 0.0
    y: float = 3000.0
    vx: float = 0.0
    vy: float = -70.0
    theta: float = 0.5 * math.pi
    omega: float = 0.0
    x_noise: float = 0.0
    y_noise: float = 0.0
    vx_noise: float = 0.0
    vy_noise: float = 0.0
    theta_noise: float = 0.0
    omega_noise: float = 0.0


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
    mass: float
    inertia: float
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
    p.mass = r.mass
    p.inertia = r.inertia
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
