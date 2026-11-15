"""Terminal capture measurements, including rotational velocity at both catch pins.

These explain kinematic gate failures; satisfying them does not override swept contacts,
other terminal causes, or establish that a descending catch-plane crossing occurred.
"""

import numpy as np
import warp as wp

from starship.sim.catch import pin, pin_error
from starship.sim.mass import mass_properties
from starship.sim.physics import Params, Terminal

CAPTURE_METRICS = (
    "pin_horizontal_error",
    "pin_height_error",
    "pin_downward_speed",
    "pin_upward_speed",
    "pin_horizontal_speed",
    "tilt",
    "angular_rate",
    "arm_angle",
)


@wp.kernel
def _measure(p: Params, t: Terminal, out: wp.array2d(dtype=float)):
    i = wp.tid()
    b = t.body[i]
    com = mass_properties(p.mass, t.tanks[i]).com
    height, down, up, horizontal = 0.0, 0.0, 0.0, 0.0
    for side in range(2):
        sign = float(2 * side - 1)
        point = pin(p.catcher, b, com, sign)
        lever = wp.vec3(p.catcher.station - com, sign * p.catcher.pin_radius, 0.0)
        velocity = b.vel + wp.quat_rotate(b.rot, wp.cross(b.omega, lever))
        height = wp.max(height, wp.abs(point[2] - p.catcher.height))
        down = wp.max(down, -velocity[2])
        up = wp.max(up, velocity[2])
        horizontal = wp.max(horizontal, wp.length(wp.vec2(velocity[0], velocity[1])))
    axis = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))
    out[i, 0] = pin_error(p.catcher, b, com)
    out[i, 1] = height
    out[i, 2] = down
    out[i, 3] = up
    out[i, 4] = horizontal
    out[i, 5] = wp.acos(wp.clamp(axis[2], -1.0, 1.0))
    out[i, 6] = wp.length(b.omega)
    out[i, 7] = t.arm_angle[i]


class CaptureMetrics:
    """Read terminal buffers before their next overwrite; select newly ended lanes only."""

    def __init__(self, sim):
        if not sim.params.catcher.enabled:
            raise ValueError("Capture measurements require the catcher")
        self.sim = sim
        p = sim.params
        self.limits = np.array(
            [
                p.catcher.tolerance,
                p.catcher.height_tolerance,
                p.landing.max_vertical_speed,
                0,
                p.landing.max_horizontal_speed,
                p.landing.max_tilt,
                p.landing.max_angular_rate,
                1.0e-4,
            ]
        )
        self.scales = self.limits.copy()
        self.scales[3] = p.landing.max_vertical_speed
        self.out = wp.zeros((sim.num_envs, len(CAPTURE_METRICS)), device=sim.device)

    def sample(self):
        wp.launch(
            _measure,
            dim=self.sim.num_envs,
            inputs=[self.sim.params, self.sim.terminal, self.out],
            device=self.sim.device,
        )
        return self.out.numpy().copy()

    def residuals(self, measurements):
        """Signed normalized excess: positive violates that kinematic gate."""
        return (measurements - self.limits) / self.scales
