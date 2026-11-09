"""Hull outline: a cylinder topped by a tangent ogive, as host functions for the precomputed
tables (mass, aerodynamic strips) and as a Warp struct for the kernels (aerodynamic shadowing,
ground contact). Every module takes its hull geometry from here. The ogive is a proxy
(docs/starship.md [4]), not a surveyed Ship profile."""

import math

import warp as wp

from starship.sim.config import HullConfig
from starship.sim.rigid import Body


def ogive_arc_radius(hull: HullConfig) -> float:
    """Radius of the circular arc that forms the nose profile."""
    r = hull.diameter / 2
    return (hull.nose_length**2 + r * r) / (2 * r)


def hull_radius(hull: HullConfig, station: float) -> float:
    """Hull radius at a station: cylinder below the nose, tangent ogive above."""
    r, nose_base = hull.diameter / 2, hull.length - hull.nose_length
    if station <= nose_base:
        return r
    rho = ogive_arc_radius(hull)
    return max(math.sqrt(max(rho * rho - (station - nose_base) ** 2, 0.0)) + r - rho, 0.0)


def hull_slope(hull: HullConfig, station: float) -> float:
    """tan of the surface angle to the axis; zero on the cylinder."""
    nose_base = hull.length - hull.nose_length
    if station <= nose_base:
        return 0.0
    arc = station - nose_base
    return arc / math.sqrt(max(ogive_arc_radius(hull) ** 2 - arc**2, 1e-12))


@wp.struct
class HullParams:
    length: float
    radius: float
    nose_length: float
    ogive_radius: float  # radius of the tangent-ogive arc


def make_hull_params(hull: HullConfig) -> HullParams:
    p = HullParams()
    p.length = hull.length
    p.radius = hull.diameter / 2
    p.nose_length = hull.nose_length
    p.ogive_radius = ogive_arc_radius(hull)
    return p


@wp.func
def station_radius(h: HullParams, station: float) -> float:
    """Hull radius at a station; zero beyond either end."""
    nose_base = h.length - h.nose_length
    if station < 0.0 or station > h.length:
        return 0.0
    if station <= nose_base:
        return h.radius
    arc = station - nose_base
    return (
        wp.sqrt(wp.max(h.ogive_radius * h.ogive_radius - arc * arc, 0.0))
        + h.radius
        - h.ogive_radius
    )


@wp.func
def lowest_point(h: HullParams, b: Body, com: float) -> float:
    """Height of the lowest hull point: the engine-plane rim, the nose-base rim, the tip, or the
    point of the tangent-ogive nose whose surface slope matches the tilt."""
    axis = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))
    s = wp.sqrt(wp.max(1.0 - axis[2] * axis[2], 0.0))
    base = b.pos[2] - com * axis[2]
    nose_base = h.length - h.nose_length
    tail = base - h.radius * s
    shoulder = base + nose_base * axis[2] - h.radius * s
    tip = base + h.length * axis[2]
    lowest = wp.min(wp.min(tail, shoulder), tip)
    # Surface height along the nose: z(u) = z(nose_base) + u a_z - r(u) s, with the ogive radius
    # r(u) = sqrt(R^2 - u^2) + radius - R. Its minimum sits where u / sqrt(R^2 - u^2) = -a_z / s,
    # which exists only when the nose points down.
    if axis[2] < 0.0 and s > 0.0:
        k = -axis[2] / s
        u = wp.min(h.ogive_radius * k / wp.sqrt(1.0 + k * k), h.nose_length)
        r = (
            wp.sqrt(wp.max(h.ogive_radius * h.ogive_radius - u * u, 0.0))
            + h.radius
            - h.ogive_radius
        )
        lowest = wp.min(lowest, base + (nose_base + u) * axis[2] - wp.max(r, 0.0) * s)
    return lowest
