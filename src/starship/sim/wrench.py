"""Force and torque on the body, and the helpers to accumulate them."""

import warp as wp


@wp.struct
class Wrench:
    force: wp.vec3  # body frame
    torque: wp.vec3  # body frame, about the centre of mass


@wp.func
def body_point(station: float, radius: float, azimuth: float, com: float) -> wp.vec3:
    """Position relative to the centre of mass, in body components (b1, b2, b3)."""
    return wp.vec3(station - com, radius * wp.sin(azimuth), radius * wp.cos(azimuth))


@wp.func
def add_wrench(a: Wrench, b: Wrench) -> Wrench:
    out = Wrench()
    out.force = a.force + b.force
    out.torque = a.torque + b.torque
    return out
