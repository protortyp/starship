"""Mass, centre of mass and principal inertia from the dry structure and the tank contents.

The dry structure is a fixed set of lumped components, so its mass, first moment and inertia about
station 0 are summed once on the host. Per env only the liquid columns change; they are added and
the total is shifted to the centre of mass with the parallel axis theorem. Products of inertia
(from the flaps and the off-axis engines) are small and dropped.
"""

import math

import numpy as np
import warp as wp

from starship.sim.config import HullConfig, VehicleConfig
from starship.sim.types import MassProps

_TANK_ORDER = ("main_lox", "main_ch4", "header_lox", "header_ch4")
_NOSE_SLICES = 400


@wp.struct
class MassParams:
    dry_mass: float
    dry_moment: float  # sum of m * station, kg m
    dry_inertia: wp.vec3  # about station 0 on the axis, along (b1, b2, b3)
    tank_bottom: wp.vec4
    tank_height: wp.vec4
    tank_radius: wp.vec4
    tank_line_density: wp.vec4  # liquid mass per metre of column, kg/m


def ogive_radius(hull: HullConfig, station: float) -> float:
    """Hull radius at a station: cylinder below the nose, tangent ogive above."""
    r, length = hull.diameter / 2, hull.nose_length
    from_tip = hull.length - station
    if from_tip >= length:
        return r
    rho = (r * r + length * length) / (2 * r)
    return math.sqrt(max(rho * rho - (length - from_tip) ** 2, 0.0)) + r - rho


def _point(m: float, x: float, y: float, z: float) -> np.ndarray:
    """Mass, first moment and inertia about station 0 of a point mass at (x, y, z)."""
    return np.array([m, m * x, m * (y * y + z * z), m * (x * x + z * z), m * (x * x + y * y)])


def _dry_components(cfg: VehicleConfig) -> np.ndarray:
    hull, mc = cfg.hull, cfg.mass
    radius = hull.diameter / 2
    total = np.zeros(5)

    # Engine ring: axial m r^2, transverse m r^2 / 2 about its own centre.
    m, x, r = mc.engines_mass, mc.engines_station, cfg.engine.pivot_radius
    total += [m, m * x, m * r * r, m * (r * r / 2 + x * x), m * (r * r / 2 + x * x)]

    m_flap = mc.flaps_mass / len(cfg.flaps)
    for flap in cfg.flaps:
        x = 0.5 * (flap.hinge_bottom + flap.hinge_top)
        r = ogive_radius(hull, x) + 0.5 * flap.span
        total += _point(m_flap, x, r * math.sin(flap.azimuth), r * math.cos(flap.azimuth))

    # Aft structure (skirt, thrust puck, aft dome) as a thin shell over the skirt.
    m, h = mc.aft_mass, hull.skirt_length
    x = 0.5 * h
    transverse = m * (radius * radius / 2 + h * h / 12 + x * x)
    total += [m, m * x, m * radius * radius, transverse, transverse]

    rest = mc.dry_mass - mc.engines_mass - mc.flaps_mass - mc.aft_mass
    # Thin shell over the tank section: axial m r^2, transverse m r^2 / 2 + m h^2 / 12.
    m = rest * mc.shell_fraction
    bottom, top = hull.skirt_length, hull.length - hull.nose_length
    x, h = 0.5 * (bottom + top), top - bottom
    transverse = m * (radius * radius / 2 + h * h / 12 + x * x)
    total += [m, m * x, m * radius * radius, transverse, transverse]

    # Uniform solid nose, integrated as thin discs (axial dm r^2 / 2, transverse dm r^2 / 4).
    m = rest - m
    dx = hull.nose_length / _NOSE_SLICES
    xs = top + dx * (np.arange(_NOSE_SLICES) + 0.5)
    rs = np.array([ogive_radius(hull, x) for x in xs])
    dm = m * rs**2 / np.sum(rs**2)
    transverse = np.sum(dm * (rs**2 / 4 + xs**2))
    total += [m, np.sum(dm * xs), np.sum(dm * rs**2 / 2), transverse, transverse]
    return total


def make_mass_params(cfg: VehicleConfig) -> MassParams:
    dry = _dry_components(cfg)
    p = MassParams()
    p.dry_mass = float(dry[0])
    p.dry_moment = float(dry[1])
    p.dry_inertia = wp.vec3(*dry[2:].tolist())
    tanks = [cfg.tanks[name] for name in _TANK_ORDER]
    heights = [t.top - t.bottom for t in tanks]
    # Radius that holds the full capacity over the tank height; the hull caps the main LOX tank.
    radii = [
        min(math.sqrt(t.capacity / (t.density * math.pi * h)), cfg.hull.diameter / 2)
        for t, h in zip(tanks, heights, strict=True)
    ]
    p.tank_bottom = wp.vec4(*[t.bottom for t in tanks])
    p.tank_height = wp.vec4(*heights)
    p.tank_radius = wp.vec4(*radii)
    p.tank_line_density = wp.vec4(
        *[t.density * math.pi * r * r for t, r in zip(tanks, radii, strict=True)]
    )
    return p


@wp.func
def mass_properties(p: MassParams, tanks: wp.vec4) -> MassProps:
    """Mass properties for tank contents (main_lox, main_ch4, header_lox, header_ch4) in kg."""
    mass = p.dry_mass
    moment = p.dry_moment
    inertia = p.dry_inertia
    for i in range(4):
        m = wp.max(tanks[i], 0.0)
        r = p.tank_radius[i]
        h = wp.min(m / p.tank_line_density[i], p.tank_height[i])
        x = p.tank_bottom[i] + 0.5 * h
        transverse = m * ((3.0 * r * r + h * h) / 12.0 + x * x)
        mass += m
        moment += m * x
        inertia += wp.vec3(0.5 * m * r * r, transverse, transverse)
    com = moment / mass
    shift = mass * com * com
    out = MassProps()
    out.mass = mass
    out.com = com
    out.inertia = wp.vec3(inertia[0], inertia[1] - shift, inertia[2] - shift)
    return out


@wp.kernel
def _mass_kernel(p: MassParams, tanks: wp.array(dtype=wp.vec4), out: wp.array2d(dtype=float)):
    i = wp.tid()
    props = mass_properties(p, tanks[i])
    out[i, 0] = props.mass
    out[i, 1] = props.com
    out[i, 2] = props.inertia[0]
    out[i, 3] = props.inertia[1]
    out[i, 4] = props.inertia[2]


def sample(p: MassParams, tanks, device: str | None = None):
    """Evaluate mass_properties for a sequence of tank vectors; returns a (N, 5) array of mass,
    centre-of-mass station and the inertia about b1, b2, b3."""
    t = wp.array(np.asarray(tanks, dtype=np.float32).reshape(-1, 4), dtype=wp.vec4, device=device)
    out = wp.zeros((len(t), 5), dtype=float, device=device)
    wp.launch(_mass_kernel, dim=len(t), inputs=[p, t, out], device=device)
    return out.numpy()
