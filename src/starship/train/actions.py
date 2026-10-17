"""Policy action layout, shared by the distribution, the env and the metrics.

The policy emits ACTION_DIM numbers per env: ENGINE_HEADS Bernoulli samples (0/1, engine on)
followed by CONTINUOUS_HEADS unbounded Gaussian samples. The env squashes each continuous
sample with tanh into its actuator range and sends the result to the simulator as a target.
"""

import math

import warp as wp

from starship.sim.physics import Commands

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


def make_action_ranges(device=None) -> wp.array:
    return wp.array([(lo, hi) for _, lo, hi in CONTINUOUS], dtype=wp.vec2, device=device)


@wp.kernel
def map_actions_kernel(
    actions: wp.array2d(dtype=float),
    mirrored: wp.array(dtype=int),
    ranges: wp.array(dtype=wp.vec2),
    commands: Commands,
    squashed: wp.array2d(dtype=float),
):
    i = wp.tid()
    for j in range(CONTINUOUS_HEADS):
        limits = ranges[j]
        squashed[i, j] = limits[0] + (limits[1] - limits[0]) * (
            0.5 + 0.5 * wp.tanh(actions[i, ENGINE_HEADS + j])
        )
    on = wp.vec3(0.0)
    for j in range(ENGINE_HEADS):
        if actions[i, j] > 0.5:
            on[j] = 1.0
    flaps = wp.vec4(squashed[i, 3], squashed[i, 4], squashed[i, 5], squashed[i, 6])
    gimbal2 = squashed[i, 1]
    rcs = squashed[i, 7]
    if mirrored[i] != 0:
        on = wp.vec3(on[0], on[2], on[1])
        flaps = wp.vec4(flaps[1], flaps[0], flaps[3], flaps[2])
        gimbal2 = -gimbal2
        rcs = -rcs
    commands.engine_on[i] = on
    commands.throttle[i] = wp.vec3(squashed[i, 0])
    commands.gimbal2[i] = wp.vec3(gimbal2)
    commands.gimbal3[i] = wp.vec3(squashed[i, 2])
    commands.flaps[i] = flaps
    commands.rcs[i] = rcs
