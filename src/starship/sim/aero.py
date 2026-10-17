"""Quasi-steady hull crossflow and four hinged flat plates, in body axes.

Sources: docs/starship.md [25,26] (Jorgensen TN D-6996/TR R-474) and claude_aero.md
sections 4.2-4.5 (digitized cylinder data and the Ortiz et al. low-AR plate fit).
This uses the high-Re cylinder branch throughout, without a drag-crisis model. Potential
flow, skin friction, wake shielding and plume interference are omitted; it is a high-AoA
engineering model, not a flight-identified aerodynamic database.
"""

import math

import numpy as np
import warp as wp

from starship.sim.atmosphere import Air, atmosphere
from starship.sim.config import VehicleConfig
from starship.sim.types import Wrench

# (crossflow Mach, C_dn): TN D-6996 figs. 1-3, high-Re column, digitization ~+/-0.05.
# The 0.7 low-M plateau is the tile-roughness prior in docs/starship.md, not smooth-cylinder
# data. The uncertain transonic peak uses 1.75; M=8 uses the reported 1.25-1.30 midpoint.
# Append rows to extend the model; outside the table the nearest endpoint is held.
CROSS_FLOW_TABLE = (
    (0.0, 0.70),
    (0.3, 0.70),
    (0.4, 0.70),
    (0.5, 1.00),
    (0.6, 1.50),
    (0.7, 1.60),
    (0.8, 1.60),
    (1.0, 1.75),
    (1.2, 1.60),
    (1.5, 1.52),
    (2.0, 1.43),
    (3.0, 1.35),
    (4.0, 1.32),
    (5.0, 1.30),
    (8.0, 1.275),
)

# (local Mach, K_p, C_N90), C_N = K_p sin(a) cos(a) + C_N90 sin^2(a).
# Ortiz, Rival & Wood (2015), doi:10.3390/en8042438: effective AR~1 gives K_p=1.6,
# ESDU plate drag gives C_N90=1.17. claude_aero.md 4.5 supplies the estimated transonic
# bridge and Newtonian-like 1.7/1.8 limits; fading K_p from M=1 to 2 is our interpolation.
PLATE_TABLE = (
    (0.0, 1.6, 1.17),
    (0.5, 1.6, 1.17),
    (1.0, 1.6, 1.60),
    (2.0, 0.0, 1.70),
    (5.0, 0.0, 1.80),
)


@wp.struct
class AeroParams:
    strip_station: wp.array(dtype=float)
    strip_area: wp.array(dtype=float)  # local planform width times dx, m^2
    crossflow_table: wp.array2d(dtype=float)
    plate_table: wp.array2d(dtype=float)
    crossflow_scale: float
    reference_area: float
    axial_nose: float
    axial_tail: float
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
        # Tangent-ogive proxy from codex_report.md section 1, not a surveyed Ship profile.
        if station <= nose_base:
            return radius
        return max(
            0.0,
            math.sqrt(max(0.0, ogive_radius**2 - (station - nose_base) ** 2))
            + radius
            - ogive_radius,
        )

    dx = h.length / cfg.aero.strips
    stations = [(i + 0.5) * dx for i in range(cfg.aero.strips)]
    hinges, axes, spans_open, spans_folded = [], [], [], []
    for f in cfg.flaps:
        station = (f.hinge_bottom + f.hinge_top) / 2
        radial = np.array([0.0, math.sin(f.azimuth), math.cos(f.azimuth)])
        axial = np.array([1.0, 0.0, 0.0])
        axis = math.cos(f.cant) * axial - math.sin(f.cant) * radial
        span = math.sin(f.cant) * axial + math.cos(f.cant) * radial
        # Li et al. hinge proxy (claude_aero.md 1.1): a straight canted hinge tangent near
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
    p.crossflow_table = wp.array(CROSS_FLOW_TABLE, dtype=float, ndim=2, device=device)
    p.plate_table = wp.array(PLATE_TABLE, dtype=float, ndim=2, device=device)
    # The contract uses configured eta at all Mach numbers (TN D-6996 low-M finite length).
    p.crossflow_scale = cfg.aero.crossflow_scale
    p.reference_area = math.pi * radius**2
    p.axial_nose = cfg.aero.axial_nose
    p.axial_tail = cfg.aero.axial_tail
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
    for i in range(p.strip_station.shape[0]):
        r = wp.vec3(p.strip_station[i] - com, 0.0, 0.0)
        local = v_air + wp.cross(omega, r)
        crossflow = wp.vec3(0.0, local[1], local[2])
        speed = wp.length(crossflow)
        cdn = _interpolate(p.crossflow_table, speed / air.speed_of_sound, 1)
        # TR R-474 strip crossflow: speed * velocity avoids dividing by zero at rest.
        f = -0.5 * air.density * p.crossflow_scale * cdn * p.strip_area[i] * speed * crossflow
        force = force + f
        torque = torque + wp.cross(r, f)

    # TN D-6996 eqs. 2-3: q*cos^2(alpha) = rho*v_axial^2/2. Apply on the axis at CoM.
    ca = float(p.axial_nose)
    if v_air[0] < 0.0:
        ca = p.axial_tail
    force = force + wp.vec3(
        -0.5 * air.density * p.reference_area * ca * v_air[0] * wp.abs(v_air[0]), 0.0, 0.0
    )

    for i in range(4):
        span = wp.cos(flap_angle[i]) * p.span_open[i] + wp.sin(flap_angle[i]) * p.span_folded[i]
        normal = wp.cross(p.hinge_axis[i], span)
        r = p.hinge[i] - wp.vec3(com, 0.0, 0.0) + p.flap_half_span[i] * span
        local = v_air + wp.cross(omega, r)
        speed = wp.length(local)
        if speed > 0.0:
            vn = wp.dot(local, normal)
            sin_a = wp.clamp(wp.abs(vn) / speed, 0.0, 1.0)
            cos_a = wp.sqrt(wp.max(0.0, 1.0 - sin_a * sin_a))
            mach = speed / air.speed_of_sound
            kp = _interpolate(p.plate_table, mach, 1)
            cn90 = _interpolate(p.plate_table, mach, 2)
            cn = kp * sin_a * cos_a + cn90 * sin_a * sin_a
            f = -wp.sign(vn) * 0.5 * air.density * speed * speed * p.flap_area[i] * cn * normal
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
