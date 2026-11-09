"""Per-episode flight statistics (travel, peak rates and loads, flips) accumulated every physics
step. They do not affect the dynamics; training reads them for diagnostics and rewards."""

import warp as wp

from starship.sim.rigid import Body

STATS_FIELDS = (
    "roll_travel",
    "horizontal_travel",
    "peak_roll_rate",
    "peak_horizontal_rate",
    "flips",
    "reflips",
    "peak_dynamic_pressure",
    "peak_mach",
    "peak_load",
    "flip_altitude",
    "ignition_altitude",
    "start_offset",
)


@wp.struct
class FlightStats:
    roll_travel: float
    horizontal_travel: float
    peak_roll_rate: float
    peak_horizontal_rate: float
    flips: int
    reflips: int
    sideslip: float
    phase: int  # 0: neither threshold reached; 1: above 60 degrees; 2: below 30 degrees
    peak_dynamic_pressure: float  # Pa, air-relative
    peak_mach: float
    peak_load: float  # non-gravitational acceleration, g
    flip_altitude: float  # m at the first flip to upright, -1 if none
    ignition_altitude: float  # m when thrust first appears, -1 if never
    start_offset: float  # m, horizontal distance to the pad at spawn


@wp.func
def track_motion(a: FlightStats, b: Body, v_air: wp.vec3, dt: float) -> FlightStats:
    roll = wp.abs(b.omega[0])
    world_rate = wp.quat_rotate(b.rot, b.omega)
    horizontal = wp.length(wp.vec2(world_rate[0], world_rate[1]))
    a.roll_travel += roll * dt
    a.horizontal_travel += horizontal * dt
    a.peak_roll_rate = wp.max(a.peak_roll_rate, roll)
    a.peak_horizontal_rate = wp.max(a.peak_horizontal_rate, horizontal)
    speed_sq = wp.dot(v_air, v_air)
    if speed_sq > 400.0:
        a.sideslip += v_air[1] * v_air[1] / speed_sq * dt
    axis_z = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))[2]
    if axis_z < wp.cos(wp.pi / 3.0):
        if a.phase == 2:
            a.reflips += 1
        a.phase = 1
    elif axis_z > wp.cos(wp.pi / 6.0):
        if a.phase == 1:
            a.flips += 1
        a.phase = 2
    return a


@wp.func
def track_flight(
    a: FlightStats, b: Body, dynamic_pressure: float, mach: float, load: float, thrusting: bool
) -> FlightStats:
    a.peak_dynamic_pressure = wp.max(a.peak_dynamic_pressure, dynamic_pressure)
    a.peak_mach = wp.max(a.peak_mach, mach)
    a.peak_load = wp.max(a.peak_load, load)
    if a.flips > 0 and a.flip_altitude < 0.0:
        a.flip_altitude = b.pos[2]
    if thrusting and a.ignition_altitude < 0.0:
        a.ignition_altitude = b.pos[2]
    return a


vec_stats = wp.types.vector(length=len(STATS_FIELDS), dtype=wp.float32)


@wp.func
def stats_vector(a: FlightStats) -> vec_stats:
    """The exported statistics as one row, in `STATS_FIELDS` order."""
    return vec_stats(
        a.roll_travel,
        a.horizontal_travel,
        a.peak_roll_rate,
        a.peak_horizontal_rate,
        float(a.flips),
        float(a.reflips),
        a.peak_dynamic_pressure,
        a.peak_mach,
        a.peak_load,
        a.flip_altitude,
        a.ignition_altitude,
        a.start_offset,
    )
