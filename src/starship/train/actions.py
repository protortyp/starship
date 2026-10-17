"""Policy action layout, shared by the distribution, the env and the metrics.

The policy emits ACTION_DIM numbers per env: ENGINE_HEADS Bernoulli samples (0/1, engine on)
followed by CONTINUOUS_HEADS unbounded Gaussian samples. The env squashes each continuous
sample with tanh into its actuator range and sends the result to the simulator as a target.
"""

import math

ENGINE_HEADS = 3
ENGINE_NAMES = ("engine0", "engine1", "engine2")
# (name, low, high) after tanh squashing; the shared throttle and gimbal apply to all engines.
CONTINUOUS = (
    ("throttle", 0.4, 1.0),
    ("gimbal2", -math.radians(15.0), math.radians(15.0)),
    ("gimbal3", -math.radians(15.0), math.radians(15.0)),
    ("flap_aft_starboard", 0.0, math.radians(85.0)),
    ("flap_aft_port", 0.0, math.radians(85.0)),
    ("flap_fwd_starboard", 0.0, math.radians(85.0)),
    ("flap_fwd_port", 0.0, math.radians(85.0)),
    ("rcs_roll", -1.0, 1.0),
)
CONTINUOUS_HEADS = len(CONTINUOUS)
ACTION_DIM = ENGINE_HEADS + CONTINUOUS_HEADS
HEAD_NAMES = ENGINE_NAMES + tuple(name for name, _, _ in CONTINUOUS)
