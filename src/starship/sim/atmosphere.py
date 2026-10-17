"""U.S. Standard Atmosphere 1976 up to 86 km geometric altitude (84.852 km geopotential).

Seven layers with linear temperature in geopotential altitude; pressure follows the hydrostatic
equation in each layer. Above the top layer the last layer is extended, which keeps the functions
smooth and finite for out-of-range inputs.
"""

import math

import warp as wp

G0 = 9.80665
R_AIR = 287.05287  # specific gas constant of dry air, J/(kg K)
GAMMA = 1.4
EARTH_RADIUS = 6356766.0  # effective radius used by the standard for geopotential altitude
SUTHERLAND_BETA = 1.458e-6
SUTHERLAND_S = 110.4

# Layer bases: geopotential altitude (m) and temperature lapse rate (K/m).
LAYER_BASE = (0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0)
LAYER_LAPSE = (-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002)


def _layer_tables() -> tuple[list[float], list[float]]:
    """Base temperature and pressure of every layer, integrated up from sea level."""
    temps, pressures = [288.15], [101325.0]
    for k in range(len(LAYER_BASE) - 1):
        t, p, lapse = temps[-1], pressures[-1], LAYER_LAPSE[k]
        dh = LAYER_BASE[k + 1] - LAYER_BASE[k]
        t_top = t + lapse * dh
        if lapse == 0.0:
            p_top = p * math.exp(-G0 * dh / (R_AIR * t))
        else:
            p_top = p * (t_top / t) ** (-G0 / (R_AIR * lapse))
        temps.append(t_top)
        pressures.append(p_top)
    return temps, pressures


_TEMPS, _PRESSURES = _layer_tables()
vec7 = wp.types.vector(length=7, dtype=wp.float32)
_BASE = wp.constant(vec7(*LAYER_BASE))
_LAPSE = wp.constant(vec7(*LAYER_LAPSE))
_T_BASE = wp.constant(vec7(*_TEMPS))
_P_BASE = wp.constant(vec7(*_PRESSURES))


@wp.struct
class Air:
    temperature: float  # K
    pressure: float  # Pa
    density: float  # kg/m^3
    speed_of_sound: float  # m/s
    viscosity: float  # dynamic viscosity, Pa s


@wp.func
def geopotential(z: float) -> float:
    return EARTH_RADIUS * z / (EARTH_RADIUS + z)


@wp.func
def atmosphere(z: float) -> Air:
    """Air properties at geometric altitude z (m); below sea level the first layer continues."""
    h = geopotential(z)
    k = 0
    for i in range(1, 7):
        if h >= _BASE[i]:
            k = i
    lapse = _LAPSE[k]
    dh = h - _BASE[k]
    t = _T_BASE[k] + lapse * dh
    p = 0.0
    if lapse == 0.0:
        p = _P_BASE[k] * wp.exp(-G0 * dh / (R_AIR * t))
    else:
        p = _P_BASE[k] * wp.pow(t / _T_BASE[k], -G0 / (R_AIR * lapse))
    air = Air()
    air.temperature = t
    air.pressure = p
    air.density = p / (R_AIR * t)
    air.speed_of_sound = wp.sqrt(GAMMA * R_AIR * t)
    air.viscosity = SUTHERLAND_BETA * t * wp.sqrt(t) / (t + SUTHERLAND_S)
    return air


@wp.kernel
def _atmosphere_kernel(z: wp.array(dtype=float), out: wp.array2d(dtype=float)):
    i = wp.tid()
    air = atmosphere(z[i])
    out[i, 0] = air.temperature
    out[i, 1] = air.pressure
    out[i, 2] = air.density
    out[i, 3] = air.speed_of_sound
    out[i, 4] = air.viscosity


def sample(altitudes, device: str | None = None):
    """Evaluate the atmosphere on the host for a sequence of altitudes; returns a (N, 5) array of
    temperature, pressure, density, speed of sound and viscosity."""
    z = wp.array(list(altitudes), dtype=float, device=device)
    out = wp.zeros((len(z), 5), dtype=float, device=device)
    wp.launch(_atmosphere_kernel, dim=len(z), inputs=[z, out], device=device)
    return out.numpy()
