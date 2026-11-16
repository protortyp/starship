"""Ground touchdown limits measured from terminal physics buffers.

These limits describe the touchdown predicate, not whether ground contact occurred.
Non-contact terminations must be considered separately.
"""

import numpy as np
import warp as wp

from starship.sim.mass import mass_properties
from starship.sim.physics import Params, Terminal

LANDING_METRICS = ("downward_speed", "horizontal_speed", "tilt", "angular_rate", "pad_miss")


@wp.kernel
def _measure(p: Params, terminal: Terminal, out: wp.array2d(dtype=float)):
    i = wp.tid()
    b = terminal.body[i]
    axis = wp.quat_rotate(b.rot, wp.vec3(1.0, 0.0, 0.0))
    base = b.pos - axis * mass_properties(p.mass, terminal.tanks[i]).com
    out[i, 0] = -b.vel[2]
    out[i, 1] = wp.length(wp.vec2(b.vel[0], b.vel[1]))
    out[i, 2] = wp.acos(wp.clamp(axis[2], -1.0, 1.0))
    out[i, 3] = wp.length(b.omega)
    out[i, 4] = wp.length(wp.vec2(base[0], base[1]))


class LandingMetrics:
    """Sample newly ended lanes before their terminal buffers are overwritten."""

    def __init__(self, sim):
        if sim.params.catcher.enabled:
            raise ValueError("Ground landing measurements require the catcher disabled")
        self.sim = sim
        p = sim.params.landing
        self.limits = np.array(
            [
                p.max_vertical_speed,
                p.max_horizontal_speed,
                p.max_tilt,
                p.max_angular_rate,
                p.pad_radius,
            ]
        )
        self.out = wp.zeros((sim.num_envs, len(LANDING_METRICS)), device=sim.device)

    def sample(self):
        wp.launch(
            _measure,
            dim=self.sim.num_envs,
            inputs=[self.sim.params, self.sim.terminal, self.out],
            device=self.sim.device,
        )
        return self.out.numpy().copy()

    def excess(self, measurements):
        """Signed excess in physical units; positive means that limit was exceeded."""
        return measurements - self.limits
