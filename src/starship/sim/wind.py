"""Per-env wind field and unsteady-aero noise: a mean profile, Dryden turbulence, discrete gusts,
hull vortex shedding and flap force noise, all drawn from the env's own random stream so that
enabling them leaves the spawn sequence untouched.

Wind is the world-frame velocity of the air; aerodynamics see v - wind, everything else (contact,
reward, observations) keeps the ground velocity. Turbulence and gusts are frozen fields that the
vehicle traverses at its speed relative to the mean wind (Taylor's hypothesis), so the filters
advance by distance, not time. Dryden axes: both horizontal components use the transverse
(second-order) MIL-F-8785C form with (sigma_v, L_v) and the vertical one (sigma_w, L_w); the
longitudinal form is not used because the descent is near vertical and the horizontal axes must
stay interchangeable under heading changes. Rotational gusts and the wind difference along the
hull are omitted. Sources: docs/starship.md [40]-[47].
"""

import math

import numpy as np
import warp as wp

from starship.sim.config import WindConfig

FT = 0.3048
W20_LIGHT = 15.0 * 0.514444  # m/s, MIL-F-8785C "light" wind at 20 ft; moderate 30 kt, severe 45 kt
MIN_ADVECTION = 0.5  # m/s, keeps the frozen field moving past a vehicle hovering in calm air
SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)

# MIL-F-8785C figure 7 (p. 49), as tabulated by JSBSim [41]: turbulence rms (ft/s) above 2000 ft
# against altitude (ft) for the exceedance curves used as light (1e-2), moderate (1e-3) and
# severe (1e-5); stored in metres and m/s.
_POE_ALTITUDE_FT = (
    500.0,
    1750.0,
    3750.0,
    7500.0,
    15000.0,
    25000.0,
    35000.0,
    45000.0,
    55000.0,
    65000.0,
)
_POE_LIGHT = (6.6, 6.9, 7.4, 6.7, 4.6, 2.7, 0.4, 0.0, 0.0, 0.0)
_POE_MODERATE = (8.6, 9.6, 10.6, 10.1, 8.0, 6.6, 5.0, 4.2, 2.7, 0.0)
_POE_SEVERE = (15.6, 17.6, 23.0, 23.6, 22.1, 20.0, 16.0, 15.1, 12.1, 7.9)
vec10 = wp.types.vector(length=10, dtype=wp.float32)
_TABLE_ALTITUDE = wp.constant(vec10(*(a * FT for a in _POE_ALTITUDE_FT)))
_TABLE_LIGHT = wp.constant(vec10(*(v * FT for v in _POE_LIGHT)))
_TABLE_MODERATE = wp.constant(vec10(*(v * FT for v in _POE_MODERATE)))
_TABLE_SEVERE = wp.constant(vec10(*(v * FT for v in _POE_SEVERE)))


@wp.struct
class WindParams:
    active: int  # any mean wind, turbulence or gust configured
    unsteady: int  # shedding or flap noise configured
    surface_speed: wp.vec2  # min, max at 10 m
    direction: float
    direction_noise: float
    shear_exponent: float
    shear_exponent_noise: float
    boundary_layer: float
    jet_speed: wp.vec2
    jet_height: float
    jet_decay: float
    jet_direction_noise: float
    turbulence: wp.vec2
    gust_speed: float
    gust_width: float
    gust_interval: float
    shedding_amplitude: float
    shedding_rate: float  # 2 pi St / D: phase advance per metre of crossflow travel
    flap_noise: float
    flap_noise_time: float


@wp.struct
class WindState:
    """Episode sample (constant until respawn) and the filter states advanced every substep."""

    surface: wp.vec2  # mean wind at 10 m, world x/y
    jet: wp.vec2  # mean wind at the jet height
    shear: float
    turbulence: float  # severity level, 0 off .. 3 severe
    gust: wp.vec3  # amplitude x, y and pulse length
    gust_progress: float  # distance travelled through the pulse; negative when none is active
    dryden1: wp.vec3  # first-stage filter states per world axis, unit variance
    dryden2: wp.vec3
    shed_phase: float
    flap_noise: wp.vec4  # fractional force noise per flap
    wind: wp.vec3  # total wind at the vehicle after the last step, world
    rng: wp.uint32


def make_wind_params(cfg: WindConfig, diameter: float) -> WindParams:
    for lo, hi in (
        (cfg.surface_speed_min, cfg.surface_speed),
        (cfg.jet_speed_min, cfg.jet_speed),
        (cfg.turbulence_min, cfg.turbulence),
    ):
        if not 0.0 <= lo <= hi:
            raise ValueError("wind ranges need 0 <= min <= max")
    if cfg.boundary_layer <= 0 or cfg.jet_height <= cfg.boundary_layer or cfg.jet_decay <= 0:
        raise ValueError("wind needs 0 < boundary_layer < jet_height and jet_decay > 0")
    if cfg.gust_width <= 0 or cfg.gust_interval <= 0 or cfg.flap_noise_time <= 0:
        raise ValueError("gust_width, gust_interval and flap_noise_time must be positive")
    p = WindParams()
    p.active = int(
        cfg.surface_speed > 0 or cfg.jet_speed > 0 or cfg.turbulence > 0 or cfg.gust_speed > 0
    )
    p.unsteady = int(cfg.shedding_amplitude > 0 or cfg.flap_noise > 0)
    p.surface_speed = wp.vec2(cfg.surface_speed_min, cfg.surface_speed)
    p.direction = cfg.direction
    p.direction_noise = cfg.direction_noise
    p.shear_exponent = cfg.shear_exponent
    p.shear_exponent_noise = cfg.shear_exponent_noise
    p.boundary_layer = cfg.boundary_layer
    p.jet_speed = wp.vec2(cfg.jet_speed_min, cfg.jet_speed)
    p.jet_height = cfg.jet_height
    p.jet_decay = cfg.jet_decay
    p.jet_direction_noise = cfg.jet_direction_noise
    p.turbulence = wp.vec2(cfg.turbulence_min, cfg.turbulence)
    p.gust_speed = cfg.gust_speed
    p.gust_width = cfg.gust_width
    p.gust_interval = cfg.gust_interval
    p.shedding_amplitude = cfg.shedding_amplitude
    p.shedding_rate = 2.0 * math.pi * cfg.shedding_strouhal / diameter
    p.flap_noise = cfg.flap_noise
    p.flap_noise_time = cfg.flap_noise_time
    return p


@wp.func
def _interpolate10(xs: vec10, ys: vec10, x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    for i in range(1, 10):
        if x <= xs[i]:
            return wp.lerp(ys[i - 1], ys[i], (x - xs[i - 1]) / (xs[i] - xs[i - 1]))
    return ys[9]


@wp.func
def _exceedance_sigma(level: float, altitude: float) -> float:
    light = _interpolate10(_TABLE_ALTITUDE, _TABLE_LIGHT, altitude)
    if level <= 1.0:
        return level * light
    moderate = _interpolate10(_TABLE_ALTITUDE, _TABLE_MODERATE, altitude)
    if level <= 2.0:
        return wp.lerp(light, moderate, level - 1.0)
    severe = _interpolate10(_TABLE_ALTITUDE, _TABLE_SEVERE, altitude)
    return wp.lerp(moderate, severe, wp.clamp(level - 2.0, 0.0, 1.0))


@wp.func
def dryden_scales(level: float, altitude: float) -> wp.vec4:
    """(sigma_h, L_h, sigma_w, L_w) in m/s and m. MIL-F-8785C low-altitude forms below 1000 ft
    with the height clamped to 10 ft, the figure-7 curves with L = 1750 ft above 2000 ft, and a
    linear blend between; sigma_u = sigma_v = sigma_w = 0.1 W20 and L = 1000 ft meet at 1000 ft."""
    h = wp.max(altitude, 10.0 * FT)
    h_ft = h / FT
    sigma_low = 0.1 * W20_LIGHT * level
    if h_ft < 1000.0:
        k = wp.pow(0.177 + 0.000823 * h_ft, 0.4)
        return wp.vec4(sigma_low / k, h / (k * k * k), sigma_low, h)
    f = wp.clamp((h_ft - 1000.0) / 1000.0, 0.0, 1.0)
    sigma = wp.lerp(sigma_low, _exceedance_sigma(level, h), f)
    scale = wp.lerp(1000.0 * FT, 1750.0 * FT, f)
    return wp.vec4(sigma, scale, sigma, scale)


@wp.func
def _rotated(v: wp.vec2, angle: float) -> wp.vec2:
    c, s = wp.cos(angle), wp.sin(angle)
    return wp.vec2(c * v[0] - s * v[1], s * v[0] + c * v[1])


@wp.func
def mean_wind(p: WindParams, w: WindState, altitude: float) -> wp.vec3:
    z = wp.max(altitude, 0.0)
    v = wp.vec2(0.0)
    if z <= p.boundary_layer:
        v = w.surface * wp.pow(z / 10.0, w.shear)
    else:
        top = w.surface * wp.pow(p.boundary_layer / 10.0, w.shear)
        if z <= p.jet_height:
            v = wp.lerp(top, w.jet, (z - p.boundary_layer) / (p.jet_height - p.boundary_layer))
        else:
            v = w.jet * wp.max(1.0 - (z - p.jet_height) / p.jet_decay, 0.0)
    return wp.vec3(v[0], v[1], 0.0)


@wp.func
def sample_wind(p: WindParams, rng: wp.uint32, altitude: float) -> WindState:
    """Episode sample; the Dryden states start in their stationary distribution."""
    w = WindState()
    direction = p.direction + p.direction_noise * wp.randf(rng, -1.0, 1.0)
    speed = wp.randf(rng, p.surface_speed[0], p.surface_speed[1])
    w.surface = wp.vec2(speed * wp.cos(direction), speed * wp.sin(direction))
    jet = wp.randf(rng, p.jet_speed[0], p.jet_speed[1])
    jet_direction = direction + p.jet_direction_noise * wp.randf(rng, -1.0, 1.0)
    w.jet = wp.vec2(jet * wp.cos(jet_direction), jet * wp.sin(jet_direction))
    w.shear = p.shear_exponent + p.shear_exponent_noise * wp.randf(rng, -1.0, 1.0)
    w.turbulence = wp.randf(rng, p.turbulence[0], p.turbulence[1])
    w.gust_progress = -1.0
    for k in range(3):
        y1 = wp.randn(rng)
        w.dryden1[k] = y1
        # Second stage: variance 1/2 and covariance 1/2 with the first (see turbulence_step).
        w.dryden2[k] = 0.5 * y1 + 0.5 * wp.randn(rng)
    w.shed_phase = wp.randf(rng, 0.0, 2.0 * wp.pi)
    for k in range(4):
        w.flap_noise[k] = p.flap_noise * wp.randn(rng)
    w.wind = mean_wind(p, w, altitude)
    w.rng = rng
    return w


@wp.func
def _dryden_step(w: WindState, scale_h: float, scale_w: float, distance: float) -> WindState:
    """Advance the Dryden transverse filters by `distance` of frozen field.

    H(s) = sigma sqrt(L/piV) (1 + sqrt3 L/V s) / (1 + L/V s)^2 realised as two cascaded first-order
    stages with unit-variance drive: y1 is Ornstein-Uhlenbeck (exact step), y2 follows y1 with the
    same time constant. Structs pass by value, so the advanced state is returned."""
    rng = w.rng
    for k in range(3):
        scale = scale_h
        if k == 2:
            scale = scale_w
        a = wp.exp(-distance / scale)
        y1 = a * w.dryden1[k] + wp.sqrt(1.0 - a * a) * wp.randn(rng)
        w.dryden2[k] = a * w.dryden2[k] + (1.0 - a) * y1
        w.dryden1[k] = y1
    w.rng = rng
    return w


@wp.func
def _dryden_output(w: WindState, sigma: float, k: int) -> float:
    # sqrt3 y1 + (1 - sqrt3) y2 has variance 2 (Var y2 = Cov(y1, y2) = 1/2), hence the 1/sqrt2.
    return sigma * (SQRT3 * w.dryden1[k] + (1.0 - SQRT3) * w.dryden2[k]) / SQRT2


@wp.func
def wind_step(p: WindParams, w: WindState, pos: wp.vec3, vel: wp.vec3, dt: float) -> WindState:
    """Advance turbulence and gusts over one substep and store the total wind in w.wind."""
    mean = mean_wind(p, w, pos[2])
    distance = wp.max(wp.length(vel - mean), MIN_ADVECTION) * dt
    total = mean
    if w.turbulence > 0.0:
        sc = dryden_scales(w.turbulence, pos[2])
        w = _dryden_step(w, sc[1], sc[3], distance)
        total = total + wp.vec3(
            _dryden_output(w, sc[0], 0), _dryden_output(w, sc[0], 1), _dryden_output(w, sc[2], 2)
        )
    if p.gust_speed > 0.0:
        rng = w.rng
        if w.gust_progress < 0.0 and wp.randf(rng) < dt / p.gust_interval:
            amplitude = wp.randf(rng, 0.0, p.gust_speed)
            heading = wp.randf(rng, 0.0, 2.0 * wp.pi)
            width = p.gust_width * wp.randf(rng, 0.5, 1.5)
            w.gust = wp.vec3(amplitude * wp.cos(heading), amplitude * wp.sin(heading), width)
            w.gust_progress = 0.0
        w.rng = rng
        if w.gust_progress >= 0.0:
            w.gust_progress += distance
            if w.gust_progress >= w.gust[2]:
                w.gust_progress = -1.0
            else:
                shape = 0.5 * (1.0 - wp.cos(2.0 * wp.pi * w.gust_progress / w.gust[2]))
                total = total + wp.vec3(w.gust[0] * shape, w.gust[1] * shape, 0.0)
    w.wind = total
    return w


@wp.func
def unsteady_step(p: WindParams, w: WindState, v_air: wp.vec3, dt: float) -> WindState:
    """Advance the shedding phase with the crossflow speed and the flap noise filters."""
    if p.shedding_amplitude > 0.0:
        crossflow = wp.sqrt(v_air[1] * v_air[1] + v_air[2] * v_air[2])
        phase = w.shed_phase + p.shedding_rate * crossflow * dt
        if phase > 2.0 * wp.pi:
            phase -= 2.0 * wp.pi
        w.shed_phase = phase
    if p.flap_noise > 0.0:
        a = wp.exp(-dt / p.flap_noise_time)
        drive = p.flap_noise * wp.sqrt(1.0 - a * a)
        rng = w.rng
        for k in range(4):
            w.flap_noise[k] = a * w.flap_noise[k] + drive * wp.randn(rng)
        w.rng = rng
    return w


@wp.func
def flap_gain(w: WindState) -> wp.vec4:
    """Multiplicative flap force factors; the noise never reverses a force."""
    return wp.vec4(
        wp.max(1.0 + w.flap_noise[0], 0.0),
        wp.max(1.0 + w.flap_noise[1], 0.0),
        wp.max(1.0 + w.flap_noise[2], 0.0),
        wp.max(1.0 + w.flap_noise[3], 0.0),
    )


@wp.kernel
def init_wind_kernel(seed: int, wind: wp.array(dtype=WindState)):
    tid = wp.tid()
    w = WindState()
    w.rng = wp.rand_init(seed, tid)
    wind[tid] = w


# Host helpers for tests.


@wp.kernel
def _series_kernel(
    p: WindParams,
    seed: int,
    pos: wp.vec3,
    vel: wp.vec3,
    dt: float,
    out: wp.array2d(dtype=wp.vec3),
):
    i = wp.tid()
    w = sample_wind(p, wp.rand_init(seed, i), pos[2])
    for k in range(out.shape[1]):
        w = wind_step(p, w, pos, vel, dt)
        out[i, k] = w.wind


@wp.kernel
def _profile_kernel(
    p: WindParams, seed: int, altitudes: wp.array(dtype=float), out: wp.array(dtype=wp.vec3)
):
    i = wp.tid()
    w = sample_wind(p, wp.rand_init(seed, 0), altitudes[i])
    out[i] = mean_wind(p, w, altitudes[i])


def wind_series(
    cfg: WindConfig,
    pos=(0.0, 0.0, 100.0),
    vel=(0.0, 0.0, -50.0),
    steps: int = 1000,
    envs: int = 1,
    dt: float = 0.01,
    seed: int = 0,
    diameter: float = 9.0,
    device: str | None = None,
) -> np.ndarray:
    """(envs, steps, 3) total wind seen by a vehicle held at `pos` moving at `vel`."""
    p = make_wind_params(cfg, diameter)
    out = wp.zeros((envs, steps), dtype=wp.vec3, device=device)
    wp.launch(
        _series_kernel,
        dim=envs,
        inputs=[p, seed, wp.vec3(*pos), wp.vec3(*vel), dt, out],
        device=device,
    )
    return out.numpy()


def wind_profile(
    cfg: WindConfig, altitudes, seed: int = 0, device: str | None = None
) -> np.ndarray:
    """(N, 3) mean wind of one episode sample at the given altitudes."""
    p = make_wind_params(cfg, 9.0)
    z = wp.array(np.asarray(altitudes, dtype=np.float32), dtype=float, device=device)
    out = wp.zeros(len(z), dtype=wp.vec3, device=device)
    wp.launch(_profile_kernel, dim=len(z), inputs=[p, seed, z, out], device=device)
    return out.numpy()
