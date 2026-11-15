"""Idealized pivoting chopsticks, catch pins, and swept hull/fin collision.

Dimensions are training assumptions, not measured tower or ship hardware. A missed descending
catch plane ends the attempt. Successful capture is terminal; support loads are not integrated.
"""

import warp as wp

from starship.sim.aero import AeroParams
from starship.sim.rigid import Body

# First detected swept contact, in tower/arm then hull/fin iteration order.
CATCH_FAILURES = {
    0: "none",
    1: "hull_tower",
    2: "hull_arm",
    3: "aft_fin_tower",
    4: "aft_fin_arm",
    5: "forward_fin_tower",
    6: "forward_fin_arm",
    7: "capture_constraints",
    8: "ground",
}


@wp.struct
class CatchParams:
    enabled: int
    moving: int
    policy_controlled: int
    flap_collisions: int
    open_angle: float
    angular_speed: float
    close_max_tilt: float
    close_max_offset: float
    close_clearance: float
    flap_half_thickness: float
    height: float
    station: float
    pin_radius: float
    tolerance: float
    height_tolerance: float
    arm_half_width: float
    arm_half_length: float
    arm_center_x: float
    tower_x: float
    tower_half_width: float
    tower_height: float


@wp.func
def catch_center(c: CatchParams, b: Body, com: float) -> wp.vec3:
    return b.pos + wp.quat_rotate(b.rot, wp.vec3(c.station - com, 0.0, 0.0))


@wp.func
def pin(c: CatchParams, b: Body, com: float, side: float) -> wp.vec3:
    return b.pos + wp.quat_rotate(b.rot, wp.vec3(c.station - com, side * c.pin_radius, 0.0))


@wp.func
def pin_height(c: CatchParams, b: Body, com: float) -> float:
    return wp.min(pin(c, b, com, -1.0)[2], pin(c, b, com, 1.0)[2])


@wp.func
def pin_error(c: CatchParams, b: Body, com: float) -> float:
    error = 0.0
    for side in range(2):
        v = pin(c, b, com, float(2 * side - 1))
        error = wp.max(error, wp.abs(v[0]))
        error = wp.max(error, wp.abs(wp.abs(v[1]) - c.pin_radius))
    return error


@wp.func
def segment_box(a: wp.vec3, b: wp.vec3, center: wp.vec3, half: wp.vec3) -> bool:
    """Slab intersection; hull radius expands the box conservatively at corners."""
    low = 0.0
    high = 1.0
    hit = True
    for j in range(3):
        d = b[j] - a[j]
        if wp.abs(d) < 1.0e-6:
            if wp.abs(a[j] - center[j]) > half[j]:
                hit = False
        else:
            t0 = (center[j] - half[j] - a[j]) / d
            t1 = (center[j] + half[j] - a[j]) / d
            low = wp.max(low, wp.min(t0, t1))
            high = wp.min(high, wp.max(t0, t1))
    return hit and low <= high


@wp.func
def arm_rotation(angle: float, side: float) -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), side * angle)


@wp.func
def arm_center(c: CatchParams, angle: float, side: float) -> wp.vec3:
    z = c.height
    if c.moving != 0:
        z -= 0.5  # height is the support surface; the box extends one metre below it
    pivot = wp.vec3(c.arm_center_x - c.arm_half_length, side * c.pin_radius, z)
    return pivot + wp.quat_rotate(arm_rotation(angle, side), wp.vec3(c.arm_half_length, 0.0, 0.0))


@wp.func
def flap_axes(a: AeroParams, b: Body, flaps: wp.vec4, i: int) -> wp.mat33:
    span = wp.cos(flaps[i]) * a.span_open[i] + wp.sin(flaps[i]) * a.span_folded[i]
    x = wp.quat_rotate(b.rot, a.hinge_axis[i])
    y = wp.quat_rotate(b.rot, span)
    z = wp.cross(x, y)
    # Rows are the plate's three orthonormal directions.
    return wp.mat33(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])


@wp.func
def flap_center(a: AeroParams, b: Body, com: float, flaps: wp.vec4, i: int) -> wp.vec3:
    span = wp.cos(flaps[i]) * a.span_open[i] + wp.sin(flaps[i]) * a.span_folded[i]
    local = a.hinge[i] - wp.vec3(com, 0.0, 0.0) + a.flap_half_span[i] * span
    return b.pos + wp.quat_rotate(b.rot, local)


@wp.func
def flap_half_size(c: CatchParams, a: AeroParams, i: int) -> wp.vec3:
    return wp.vec3(
        a.flap_area[i] / (4.0 * a.flap_half_span[i]), a.flap_half_span[i], c.flap_half_thickness
    )


@wp.func
def projected_radius(axes: wp.mat33, half: wp.vec3, direction: wp.vec3) -> float:
    result = 0.0
    for i in range(3):
        result += half[i] * wp.abs(wp.dot(axes[i], direction))
    return result


@wp.func
def boxes_overlap(
    ca: wp.vec3, aa: wp.mat33, ha: wp.vec3, cb: wp.vec3, ab: wp.mat33, hb: wp.vec3
) -> bool:
    """Full separating-axis test, including all nine edge-cross-edge axes."""
    delta = cb - ca
    for i in range(3):
        if wp.abs(wp.dot(delta, aa[i])) > ha[i] + projected_radius(ab, hb, aa[i]):
            return False
        if wp.abs(wp.dot(delta, ab[i])) > hb[i] + projected_radius(aa, ha, ab[i]):
            return False
        for j in range(3):
            axis = wp.cross(aa[i], ab[j])
            if wp.abs(wp.dot(delta, axis)) > (
                projected_radius(aa, ha, axis) + projected_radius(ab, hb, axis) + 1.0e-6
            ):
                return False
    return True


@wp.func
def commanded_arm_step(c: CatchParams, angle: float, close: float, dt: float) -> float:
    """Policy command only: no automatic clearance gate or override. Collisions remain active."""
    target = c.open_angle
    if close > 0.5:
        target = 0.0
    change = wp.clamp(target - angle, -c.angular_speed * dt, c.angular_speed * dt)
    return wp.clamp(angle + change, 0.0, c.open_angle)


@wp.func
def tower_step(
    c: CatchParams, a: AeroParams, b: Body, com: float, flaps: wp.vec4, angle: float, dt: float
) -> float:
    if c.moving == 0:
        return 0.0
    # Close only once BOTH aft fins have entirely cleared the arm underside.
    aft_top = -1.0e10
    for i in range(2):
        center = flap_center(a, b, com, flaps, i)
        top = center[2] + projected_radius(
            flap_axes(a, b, flaps, i), flap_half_size(c, a, i), wp.vec3(0.0, 0.0, 1.0)
        )
        aft_top = wp.max(aft_top, top)
    axis = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))
    center = catch_center(c, b, com)
    target = c.open_angle
    if (
        aft_top < c.height - 1.0 - c.close_clearance
        and axis[2] > wp.cos(c.close_max_tilt)
        and wp.length(wp.vec2(center[0], center[1])) < c.close_max_offset
        and pin_height(c, b, com) > c.height
    ):
        target = 0.0
    change = wp.clamp(target - angle, -c.angular_speed * dt, c.angular_speed * dt)
    return wp.clamp(angle + change, 0.0, c.open_angle)


@wp.func
def structure_contact(
    c: CatchParams,
    a: AeroParams,
    previous: Body,
    b: Body,
    com: float,
    old_flaps: wp.vec4,
    flaps: wp.vec4,
    length: float,
    radius: float,
    old_angle: float,
    angle: float,
) -> int:
    """Conservative swept hull capsule and fin boxes against tower and rotating arms.

    Translation and rotation bounds inflate midpoint shapes over a physics substep, preventing
    thin fins or moving arms from tunnelling between endpoints. This is not impact dynamics.
    """
    if wp.min(previous.pos[2], b.pos[2]) > c.tower_height + length:
        return 0
    mid = b
    mid.pos = 0.5 * (previous.pos + b.pos)
    mid.rot = wp.quat_slerp(previous.rot, b.rot, 0.5)
    q = wp.quat_inverse(previous.rot) * b.rot
    turn = 2.0 * wp.acos(wp.clamp(wp.abs(q[3]), 0.0, 1.0))
    travel = 0.5 * wp.length(b.pos - previous.pos)
    hull_sweep = travel + 0.5 * turn * length
    if c.moving == 0 and c.flap_collisions == 0:
        # Historical fixed-arm recordings retain their original endpoint-only hull test.
        mid = b
        hull_sweep = 0.0
    axis = wp.quat_rotate(mid.rot, wp.vec3(1.0, 0.0, 0.0))
    base = mid.pos - axis * com
    nose = base + axis * length
    midpoint_angle = 0.5 * (old_angle + angle)
    arm_sweep = (c.arm_half_length + 0.5 * c.arm_half_width) * wp.abs(angle - old_angle)
    for structure in range(3):
        center = wp.vec3(c.tower_x, 0.0, c.tower_height * 0.5)
        half = wp.vec3(c.tower_half_width, c.tower_half_width, c.tower_height * 0.5)
        rot = wp.quat_identity()
        if structure > 0:
            side = float(2 * structure - 3)
            center = arm_center(c, midpoint_angle, side)
            rot = arm_rotation(midpoint_angle, side)
            half = wp.vec3(c.arm_half_length, c.arm_half_width, 0.5) + wp.vec3(arm_sweep)
        local_base = wp.quat_rotate_inv(rot, base - center)
        local_nose = wp.quat_rotate_inv(rot, nose - center)
        if segment_box(local_base, local_nose, wp.vec3(0.0), half + wp.vec3(radius + hull_sweep)):
            return 1 if structure == 0 else 2
        if c.flap_collisions != 0:
            x = wp.quat_rotate(rot, wp.vec3(1.0, 0.0, 0.0))
            y = wp.quat_rotate(rot, wp.vec3(0.0, 1.0, 0.0))
            z = wp.vec3(0.0, 0.0, 1.0)
            axes = wp.mat33(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])
            angles = 0.5 * (old_flaps + flaps)
            for i in range(4):
                sweep = travel + 0.5 * turn * (length + 2.0 * a.flap_half_span[i])
                sweep += a.flap_half_span[i] * wp.abs(flaps[i] - old_flaps[i])
                if boxes_overlap(
                    flap_center(a, mid, com, angles, i),
                    flap_axes(a, mid, angles, i),
                    flap_half_size(c, a, i) + wp.vec3(sweep),
                    center,
                    axes,
                    half,
                ):
                    kind = 3 if i < 2 else 5
                    return kind if structure == 0 else kind + 1
    return 0


@wp.func
def structure_collision(
    c: CatchParams,
    a: AeroParams,
    previous: Body,
    b: Body,
    com: float,
    old_flaps: wp.vec4,
    flaps: wp.vec4,
    length: float,
    radius: float,
    old_angle: float,
    angle: float,
) -> bool:
    return (
        structure_contact(
            c, a, previous, b, com, old_flaps, flaps, length, radius, old_angle, angle
        )
        != 0
    )
