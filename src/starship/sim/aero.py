"""Quasi-steady aerodynamics in body axes from rest to Mach 25: strip-wise hull crossflow, a
slender-body nose term, axial drag tables, modified-Newtonian ogive strips above Mach ~1.1 and
four hinged flat-plate flaps with hull shadowing.

Sources (docs/starship.md): Jorgensen TN D-6996 / TR R-474 [25] [26] for the hull crossflow,
Allen-Perkins [32] and Ortiz et al. low aspect ratio plate data for the flaps, Li et al. 2022 [12]
for flap geometry and hypersonic validation, Lees' modified Newtonian law [35] with Penland's
high-alpha cylinder data [36] for the entry regime, Fleeman [37] and Love [38] for axial drag.
High-Re cylinder branch throughout, no drag crisis. Real-gas effects, lee-side suction, wake
shielding and plume interference are omitted (docs/gaps.md); an engineering model, not a
flight-identified aerodynamic database.
"""

import math

import numpy as np
import warp as wp

from starship.sim.atmosphere import Air, atmosphere
from starship.sim.config import VehicleConfig
from starship.sim.types import Wrench

# (crossflow Mach, C_dn, eta blend): TN D-6996 figs. 1-3, high-Re column, digitization ~+/-0.05.
# The 0.7 low-M plateau is the tile-roughness prior in docs/starship.md, not smooth-cylinder
# data. The uncertain transonic peak uses 1.75; M=8 uses the reported 1.25-1.30 midpoint and is
# held to M 25, where the Newtonian cylinder limit (2/3) Cp_max gives 1.22-1.27 [35].
# The blend column moves the finite-length factor eta from its subsonic value to the supersonic
# one between M_n 0.8 and 1.2: with supersonic crossflow the tips stop relieving the pressure
# and eta -> 1 (TR R-474 [26]). Outside the table the nearest endpoint is held.
CROSS_FLOW_TABLE = (
    (0.0, 0.70, 0.0),
    (0.3, 0.70, 0.0),
    (0.4, 0.70, 0.0),
    (0.5, 1.00, 0.0),
    (0.6, 1.50, 0.0),
    (0.7, 1.60, 0.0),
    (0.8, 1.60, 0.0),
    (1.0, 1.75, 0.5),
    (1.2, 1.60, 1.0),
    (1.5, 1.52, 1.0),
    (2.0, 1.43, 1.0),
    (3.0, 1.35, 1.0),
    (4.0, 1.32, 1.0),
    (5.0, 1.30, 1.0),
    (8.0, 1.275, 1.0),
    (12.0, 1.27, 1.0),
    (25.0, 1.27, 1.0),
)

# (local Mach, K_p, C_N90), C_N = K_p sin(a) cos(a) + C_N90 sin^2(a).
# Ortiz, Rival & Wood (2015), doi:10.3390/en8042438: effective AR~1 gives K_p=1.6,
# ESDU plate drag gives C_N90=1.17. docs/starship.md [12] supplies the estimated transonic
# bridge and the 1.7 limit at M 2; fading K_p from M=1 to 2 is our interpolation. From M 3 on
# C_N90 is the modified-Newtonian plate value Cp_max(M) [35] (no edge relief, which the
# lee-side suction of order 1/M^2 roughly cancels).
PLATE_TABLE = (
    (0.0, 1.6, 1.17),
    (0.5, 1.6, 1.17),
    (1.0, 1.6, 1.60),
    (2.0, 0.0, 1.70),
    (3.0, 0.0, 1.76),
    (5.0, 0.0, 1.81),
    (8.0, 0.0, 1.83),
    (25.0, 0.0, 1.84),
)

# (freestream Mach, Cp_max, Newtonian weight, C_A0 nose-first, C_A0 tail-first).
# Cp_max is the pitot (normal shock) stagnation pressure coefficient for gamma 1.4 [35]; it is
# only used where the weight is non-zero, so the subsonic rows hold the M 1 value. The weight
# hands the ogive strips and the flap shadowing from the subsonic model to the Newtonian one
# between M 1.1 and 3, where Penland's data show Newtonian pressures take over [36].
# Nose-first C_A0 (reference area = cross-section) follows Fleeman's wave + base + friction
# build-up [37] with Love's base pressure [38]: 0.4 subsonic, a transonic peak near 0.65, then
# falling as the Newtonian ogive strips take over the wave drag and only base drag (~0.25/M)
# plus friction (~0.05) remain in this column. Tail-first C_A0 is Hoerner's flat-faced cylinder
# (0.8-0.9) subsonic, rising to ~0.9 Cp_max for the flat base once a detached bow shock sits
# in front of it (the engine bay is a cavity, not a plate, hence the 10 % relief).
MACH_TABLE = (
    (0.0, 1.276, 0.0, 0.40, 0.90),
    (0.8, 1.276, 0.0, 0.40, 0.90),
    (0.9, 1.276, 0.0, 0.45, 1.10),
    (1.1, 1.337, 0.0, 0.65, 1.30),
    (1.5, 1.532, 0.21, 0.51, 1.45),
    (2.0, 1.657, 0.47, 0.38, 1.55),
    (3.0, 1.756, 1.0, 0.13, 1.62),
    (5.0, 1.809, 1.0, 0.10, 1.65),
    (10.0, 1.832, 1.0, 0.075, 1.66),
    (25.0, 1.838, 1.0, 0.06, 1.67),
)


@wp.struct
class AeroParams:
    strip_station: wp.array(dtype=float)
    strip_area: wp.array(dtype=float)  # local planform width times dx, m^2
    strip_slope: wp.array(dtype=float)  # tan of the surface angle to the axis, 0 on the cylinder
    crossflow_table: wp.array2d(dtype=float)
    plate_table: wp.array2d(dtype=float)
    mach_table: wp.array2d(dtype=float)
    crossflow_drag_scale: float
    flap_force_scale: float
    crossflow_scale: float
    crossflow_scale_supersonic: float
    reference_area: float
    length: float
    radius: float
    nose_base: float  # station where the ogive starts
    ogive_radius: float
    potential_station: float  # where the slender-body normal force acts (nose centroid)
    hinge: wp.array(dtype=wp.vec3)  # absolute body positions measured from station zero
    span_open: wp.array(dtype=wp.vec3)  # unit span perpendicular to the canted hinge
    span_folded: wp.array(dtype=wp.vec3)
    hinge_axis: wp.array(dtype=wp.vec3)
    flap_area: wp.vec4
    flap_half_span: wp.vec4


def make_aero_params(cfg: VehicleConfig, device: str | None = None) -> AeroParams:
    """Precompute geometry and tables on the current Warp device (or the supplied device)."""
    h = cfg.hull
    if cfg.aero.strips < 1:
        raise ValueError("aero.strips must be positive")
    if h.diameter <= 0 or not 0 < h.nose_length <= h.length:
        raise ValueError("hull needs a positive diameter and 0 < nose_length <= length")
    if len(cfg.flaps) != 4:
        raise ValueError("aero requires four flaps in default_flaps order")

    radius = h.diameter / 2
    nose_base = h.length - h.nose_length
    ogive_radius = (h.nose_length**2 + radius**2) / (2 * radius)

    def hull_radius(station):
        # Tangent-ogive proxy (docs/starship.md [4]), not a surveyed Ship profile.
        if station <= nose_base:
            return radius
        return max(
            0.0,
            math.sqrt(max(0.0, ogive_radius**2 - (station - nose_base) ** 2))
            + radius
            - ogive_radius,
        )

    def hull_slope(station):
        if station <= nose_base:
            return 0.0
        arc = station - nose_base
        return arc / math.sqrt(max(ogive_radius**2 - arc**2, 1e-12))

    dx = h.length / cfg.aero.strips
    stations = [(i + 0.5) * dx for i in range(cfg.aero.strips)]
    hinges, axes, spans_open, spans_folded = [], [], [], []
    for f in cfg.flaps:
        station = (f.hinge_bottom + f.hinge_top) / 2
        radial = np.array([0.0, math.sin(f.azimuth), math.cos(f.azimuth)])
        axial = np.array([1.0, 0.0, 0.0])
        axis = math.cos(f.cant) * axial - math.sin(f.cant) * radial
        span = math.sin(f.cant) * axial + math.cos(f.cant) * radial
        # Li et al. hinge proxy (docs/starship.md [12]): a straight canted hinge tangent near
        # its midpoint on the tapered hull. A uniform plate puts its CP half a span out.
        hinge = np.array([station, 0.0, 0.0]) + hull_radius(station) * radial
        # phi is measured from +b3: increasing |phi| folds both sides leeward. Thus the
        # starboard fold is towards +phi, the port fold towards -phi (mirror symmetry).
        # 0 deg is the radial plane; 90 deg is circumferential/tangent, pointing towards
        # -b3. At the relocated forward hinges it also points inboard, not exactly -b3.
        side = 1.0 if math.sin(f.azimuth) >= 0 else -1.0
        folded = -side * np.cross(axis, span)
        hinges.append(hinge)
        axes.append(axis)
        spans_open.append(span)
        spans_folded.append(folded)

    p = AeroParams()
    p.strip_station = wp.array(stations, dtype=float, device=device)
    p.strip_area = wp.array([2 * hull_radius(x) * dx for x in stations], dtype=float, device=device)
    p.strip_slope = wp.array([hull_slope(x) for x in stations], dtype=float, device=device)
    p.crossflow_table = wp.array(CROSS_FLOW_TABLE, dtype=float, ndim=2, device=device)
    p.plate_table = wp.array(PLATE_TABLE, dtype=float, ndim=2, device=device)
    p.mach_table = wp.array(MACH_TABLE, dtype=float, ndim=2, device=device)
    p.crossflow_drag_scale = cfg.aero.crossflow_drag_scale
    p.flap_force_scale = cfg.aero.flap_force_scale
    p.crossflow_scale = cfg.aero.crossflow_scale
    p.crossflow_scale_supersonic = cfg.aero.crossflow_scale_supersonic
    p.reference_area = math.pi * radius**2
    p.length = h.length
    p.radius = radius
    p.nose_base = nose_base
    p.ogive_radius = ogive_radius
    p.potential_station = h.length - 0.5 * h.nose_length
    p.hinge = wp.array(hinges, dtype=wp.vec3, device=device)
    p.hinge_axis = wp.array(axes, dtype=wp.vec3, device=device)
    p.span_open = wp.array(spans_open, dtype=wp.vec3, device=device)
    p.span_folded = wp.array(spans_folded, dtype=wp.vec3, device=device)
    p.flap_area = wp.vec4(*(f.area for f in cfg.flaps))
    p.flap_half_span = wp.vec4(*(0.5 * f.span for f in cfg.flaps))
    return p


@wp.func
def _interpolate(table: wp.array2d(dtype=float), x: float, column: int) -> float:
    """Piecewise linear coefficient lookup with constant endpoint extrapolation."""
    if x <= table[0, 0]:
        return table[0, column]
    for i in range(1, table.shape[0]):
        if x <= table[i, 0]:
            t = (x - table[i - 1, 0]) / (table[i, 0] - table[i - 1, 0])
            return wp.lerp(table[i - 1, column], table[i, column], t)
    return table[table.shape[0] - 1, column]


@wp.func
def _interpolate2(table: wp.array2d(dtype=float), x: float, column: int) -> wp.vec2:
    """Two adjacent columns from one scan; the strip loop runs it per strip and substep."""
    if x <= table[0, 0]:
        return wp.vec2(table[0, column], table[0, column + 1])
    for i in range(1, table.shape[0]):
        if x <= table[i, 0]:
            t = (x - table[i - 1, 0]) / (table[i, 0] - table[i - 1, 0])
            return wp.vec2(
                wp.lerp(table[i - 1, column], table[i, column], t),
                wp.lerp(table[i - 1, column + 1], table[i, column + 1], t),
            )
    last = table.shape[0] - 1
    return wp.vec2(table[last, column], table[last, column + 1])


@wp.func
def _hull_radius(p: AeroParams, station: float) -> float:
    """Tangent-ogive hull radius; zero beyond either end."""
    if station < 0.0 or station > p.length:
        return 0.0
    if station <= p.nose_base:
        return p.radius
    arc = station - p.nose_base
    return (
        wp.sqrt(wp.max(p.ogive_radius * p.ogive_radius - arc * arc, 0.0))
        + p.radius
        - p.ogive_radius
    )


@wp.func
def _newtonian_strip(slope: float, cos_a: float, sin_a: float) -> wp.vec2:
    """Modified-Newtonian pressure integrals over the wetted sector of a body-of-revolution strip
    with surface slope tan(delta) = slope at local angle of attack alpha, per unit Cp_max r dx:
    axial force = slope * I0, normal force = I1 (Lees [35], integrated in closed form)."""
    cos_d = 1.0 / wp.sqrt(1.0 + slope * slope)
    a = slope * cos_d * cos_a
    b = cos_d * sin_a
    phi0 = 0.0
    if a >= b:
        phi0 = math.pi
    elif a > -b:
        phi0 = wp.acos(-a / b)
    else:
        return wp.vec2(0.0, 0.0)
    s = wp.sin(phi0)
    s2 = wp.sin(2.0 * phi0)
    i0 = 2.0 * a * a * phi0 + 4.0 * a * b * s + b * b * (phi0 + 0.5 * s2)
    i1 = 2.0 * a * a * s + 2.0 * a * b * (phi0 + 0.5 * s2) + 2.0 * b * b * (s - s * s * s / 3.0)
    return wp.vec2(slope * i0, i1)


@wp.func
def _shadow_fraction(p: AeroParams, root: wp.vec3, tip: wp.vec3, velocity: wp.vec3) -> float:
    """Fraction of the plate span whose upstream ray passes through the hull, i.e. that sits in
    the hull's Newtonian shadow. The hull is taken as a disc in the crossflow plane with the
    radius at the station where the plate centre's ray crosses the axis plane, so a shallow ray
    that leaves past the nose tip or the base is not shadowed."""
    lateral = wp.vec2(velocity[1], velocity[2])
    lateral_speed = wp.length(lateral)
    if lateral_speed == 0.0:
        return 0.0
    c = lateral / lateral_speed
    p0 = wp.vec2(root[1], root[2])
    d = wp.vec2(tip[1] - root[1], tip[2] - root[2])
    centre = p0 + 0.5 * d
    station = 0.5 * (root[0] + tip[0]) - wp.dot(centre, c) * velocity[0] / lateral_speed
    r = _hull_radius(p, station)
    if r <= 0.0:
        return 0.0
    # Along the span s in [0, 1] both the offset from the flow line through the axis, h(s), and
    # the position along the flow, g(s), are linear; shadowed where |h| < r and g < 0.
    h0 = p0[0] * c[1] - p0[1] * c[0]
    h1 = d[0] * c[1] - d[1] * c[0]
    g0 = wp.dot(p0, c)
    g1 = wp.dot(d, c)
    lo = 0.0
    hi = 1.0
    if wp.abs(h1) > 1.0e-6:
        s0 = (-r - h0) / h1
        s1 = (r - h0) / h1
        lo = wp.max(lo, wp.min(s0, s1))
        hi = wp.min(hi, wp.max(s0, s1))
    elif wp.abs(h0) >= r:
        return 0.0
    if wp.abs(g1) > 1.0e-6:
        s = -g0 / g1
        if g1 > 0.0:
            hi = wp.min(hi, s)
        else:
            lo = wp.max(lo, s)
    elif g0 >= 0.0:
        return 0.0
    return wp.max(hi - lo, 0.0)


@wp.func
def aero_wrench(
    p: AeroParams,
    v_air: wp.vec3,
    omega: wp.vec3,
    com: float,
    flap_angle: wp.vec4,
    air: Air,
) -> Wrench:
    """Body force and torque about CoM; angles are actual flap positions in radians."""
    force = wp.vec3(0.0)
    torque = wp.vec3(0.0)
    speed = wp.length(v_air)
    mach = speed / air.speed_of_sound
    cp_max = _interpolate(p.mach_table, mach, 1)
    newton = _interpolate(p.mach_table, mach, 2)

    for i in range(p.strip_station.shape[0]):
        r = wp.vec3(p.strip_station[i] - com, 0.0, 0.0)
        local = v_air + wp.cross(omega, r)
        crossflow = wp.vec3(0.0, local[1], local[2])
        crossflow_speed = wp.length(crossflow)
        crossflow_row = _interpolate2(p.crossflow_table, crossflow_speed / air.speed_of_sound, 1)
        cdn = crossflow_row[0] * p.crossflow_drag_scale
        eta = wp.lerp(p.crossflow_scale, p.crossflow_scale_supersonic, crossflow_row[1])
        slope = p.strip_slope[i]
        # Viscous crossflow drag exceeds the Newtonian value up to M_n ~4 (lee-side suction),
        # so the cylinder keeps the measured C_dn at every Mach number; only the tapered ogive
        # strips, whose Newtonian loads depend on the surface slope, hand over to Newtonian.
        weight = 1.0
        if slope > 0.0:
            weight = 1.0 - newton
        # TR R-474 strip crossflow: speed * velocity avoids dividing by zero at rest.
        f = -0.5 * air.density * weight * eta * cdn * p.strip_area[i] * crossflow_speed * crossflow
        if slope > 0.0 and newton > 0.0:
            local_speed = wp.length(local)
            if local_speed > 0.0:
                loads = _newtonian_strip(
                    slope, local[0] / local_speed, crossflow_speed / local_speed
                )
                # r dx = strip_area / 2; pressure pushes the forward-facing ogive aft and the
                # windward side towards the lee at any flow direction, tail-first included.
                scale = 0.25 * newton * air.density * local_speed * local_speed * cp_max
                scale = scale * p.strip_area[i]
                f = f - wp.vec3(scale * loads[0], 0.0, 0.0)
                if crossflow_speed > 0.0:
                    f = f - (scale * loads[1] / crossflow_speed) * crossflow
        force = force + f
        torque = torque + wp.cross(r, f)

    # Slender-body (potential) normal force, TN D-6996 eq. 1: C_N = (A_b/A) sin(2a) cos(a/2) with
    # reference area A_b, carried by the nose. Only for nose-first flow: the flat base of a
    # tail-first body generates no potential lift. The Newtonian ogive strips replace it.
    lateral = wp.vec3(0.0, v_air[1], v_air[2])
    lateral_speed = wp.length(lateral)
    if v_air[0] > 0.0 and lateral_speed > 0.0 and newton < 1.0:
        cos_a = v_air[0] / speed
        sin_a = lateral_speed / speed
        half = wp.sqrt(0.5 * (1.0 + cos_a))  # cos(a/2)
        cn = 2.0 * sin_a * cos_a * half * (1.0 - newton)
        r = wp.vec3(p.potential_station - com, 0.0, 0.0)
        f = -(0.5 * air.density * speed * speed * p.reference_area * cn / lateral_speed) * lateral
        force = force + f
        torque = torque + wp.cross(r, f)

    # TN D-6996 eqs. 2-3: q*cos^2(alpha) = rho*v_axial^2/2, the Newtonian law for the flat base
    # as well. Apply on the axis at CoM.
    ca = _interpolate(p.mach_table, mach, 3)
    if v_air[0] < 0.0:
        ca = _interpolate(p.mach_table, mach, 4)
    force = force + wp.vec3(
        -0.5 * air.density * p.reference_area * ca * v_air[0] * wp.abs(v_air[0]), 0.0, 0.0
    )

    for i in range(4):
        span = wp.cos(flap_angle[i]) * p.span_open[i] + wp.sin(flap_angle[i]) * p.span_folded[i]
        normal = wp.cross(p.hinge_axis[i], span)
        r = p.hinge[i] - wp.vec3(com, 0.0, 0.0) + p.flap_half_span[i] * span
        local = v_air + wp.cross(omega, r)
        local_speed = wp.length(local)
        if local_speed > 0.0:
            vn = wp.dot(local, normal)
            sin_a = wp.clamp(wp.abs(vn) / local_speed, 0.0, 1.0)
            cos_a = wp.sqrt(wp.max(0.0, 1.0 - sin_a * sin_a))
            local_mach = local_speed / air.speed_of_sound
            kp = _interpolate(p.plate_table, local_mach, 1)
            cn90 = _interpolate(p.plate_table, local_mach, 2)
            cn = kp * sin_a * cos_a + cn90 * sin_a * sin_a
            # A plate behind the hull sees no Newtonian flow: the subsonic model keeps its
            # wake loads, so the shadow only fades in with the Newtonian weight.
            exposed = 1.0
            if newton > 0.0:
                tip = p.hinge[i] + 2.0 * p.flap_half_span[i] * span
                exposed = 1.0 - newton * _shadow_fraction(p, p.hinge[i], tip, local)
            pressure = -wp.sign(vn) * 0.5 * air.density * local_speed * local_speed * exposed
            f = pressure * p.flap_area[i] * cn * normal
            f = f * p.flap_force_scale
            force = force + f
            torque = torque + wp.cross(r, f)

    out = Wrench()
    out.force = force
    out.torque = torque
    return out


@wp.kernel
def _sample_kernel(
    p: AeroParams,
    velocity: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    com: wp.array(dtype=float),
    angles: wp.array(dtype=wp.vec4),
    altitude: wp.array(dtype=float),
    out: wp.array2d(dtype=float),
):
    i = wp.tid()
    wrench = aero_wrench(p, velocity[i], omega[i], com[i], angles[i], atmosphere(altitude[i]))
    for j in range(3):
        out[i, j] = wrench.force[j]
        out[i, j + 3] = wrench.torque[j]


def sample(
    velocities,
    *,
    omega=(0.0, 0.0, 0.0),
    com=26.0,
    flap_angle=(math.pi / 4,) * 4,
    altitude=0.0,
    cfg: VehicleConfig | None = None,
    device: str | None = None,
):
    """Return (N, 6) force/torque samples; other inputs broadcast over N velocities.

    This test helper allocates parameters each call. The simulator should construct them once
    and call aero_wrench from its physics kernel instead.
    """
    velocities = np.asarray(velocities, dtype=np.float32).reshape((-1, 3))
    n = len(velocities)
    p = make_aero_params(cfg if cfg is not None else VehicleConfig(), device=device)

    def array(value, shape, dtype):
        return wp.array(np.broadcast_to(value, shape).copy(), dtype=dtype, device=device)

    inputs = [
        p,
        wp.array(velocities, dtype=wp.vec3, device=device),
        array(omega, (n, 3), wp.vec3),
        array(com, (n,), float),
        array(flap_angle, (n, 4), wp.vec4),
        array(altitude, (n,), float),
    ]
    out = wp.zeros((n, 6), dtype=float, device=device)
    wp.launch(_sample_kernel, dim=n, inputs=[*inputs, out], device=device)
    return out.numpy()
