"""Idealized fixed chopsticks: two catch pins, rigid arms, conservative hull collision.

Dimensions are training assumptions, not measured tower or ship hardware. A missed descending
catch plane ends the attempt. Successful capture is terminal; support loads are not integrated.
"""

import warp as wp

from starship.sim.rigid import Body


@wp.struct
class CatchParams:
    enabled: int
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
def structure_collision(c: CatchParams, b: Body, com: float, length: float, radius: float) -> bool:
    axis = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))
    base = b.pos - axis * com
    nose = base + axis * length
    hit = segment_box(
        base,
        nose,
        wp.vec3(c.tower_x, 0.0, c.tower_height * 0.5),
        wp.vec3(
            c.tower_half_width + radius, c.tower_half_width + radius, c.tower_height * 0.5 + radius
        ),
    )
    for side in range(2):
        hit = hit or segment_box(
            base,
            nose,
            wp.vec3(c.arm_center_x, float(2 * side - 1) * c.pin_radius, c.height),
            wp.vec3(c.arm_half_length + radius, c.arm_half_width + radius, 0.5 + radius),
        )
    return hit
