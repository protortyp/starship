"""Stage-0 reference controller for demonstrations, never a policy inference override.

Commands use the same 12-action interface, actuator limits, lag and collision checks as PPO.
The controller is only intended for the short, upright, near-aligned catch approach.
"""

import warp as wp

from starship.sim.atmosphere import atmosphere
from starship.sim.catch import catch_center, tower_step
from starship.sim.mass import mass_properties
from starship.sim.physics import Params, State
from starship.sim.rigid import gravity, orientation
from starship.train.actions import ACTION_DIM, ARM_HEAD, CONTINUOUS, ENGINE_HEADS


@wp.func
def raw_action(value: float, low: float, high: float) -> float:
    normalized = wp.clamp(2.0 * (value - low) / (high - low) - 1.0, -0.999, 0.999)
    return 0.5 * wp.log((1.0 + normalized) / (1.0 - normalized))


@wp.kernel
def teach(
    p: Params,
    s: State,
    mirrored: wp.array(dtype=int),
    ranges: wp.array(dtype=wp.vec2),
    actions: wp.array2d(dtype=float),
):
    i = wp.tid()
    b = s.body[i]
    mp = mass_properties(p.mass, s.tanks[i])
    center = catch_center(p.catcher, b, mp.com)
    lever = wp.vec3(p.catcher.station - mp.com, 0.0, 0.0)
    pin_velocity = b.vel + wp.quat_rotate(b.rot, wp.cross(b.omega, lever))
    acceleration = wp.vec3(
        -0.04 * center[0] - 0.4 * pin_velocity[0],
        -0.04 * center[1] - 0.4 * pin_velocity[1],
        -1.3 - b.vel[2],
    )
    force = mp.mass * (acceleration + wp.vec3(0.0, 0.0, gravity(b.pos[2])))
    length = mp.com - p.propulsion.pivot_station
    radius = p.propulsion.pivot_radius
    base = orientation(0.0, 0.0)
    # Engine 0 sits on +b3. Cant thrust through the COM, then tilt for vertical net thrust.
    direction = wp.normalize(wp.vec3(length, 0.0, -radius))
    a = wp.quat_rotate(base, direction)
    target = wp.normalize(force)
    axis = wp.cross(a, target)
    turn = wp.normalize(wp.quat(axis[0], axis[1], axis[2], 1.0 + wp.dot(a, target)))
    desired = turn * base
    error = wp.quat_inverse(b.rot) * desired
    sign = 1.0
    if error[3] < 0.0:
        sign = -1.0
    rotation_error = 2.0 * sign * wp.vec3(error[0], error[1], error[2])
    angular_acceleration = 2.0 * rotation_error - 2.8 * b.omega
    momentum = wp.cw_mul(mp.inertia, b.omega)
    torque = wp.cw_mul(mp.inertia, angular_acceleration) + wp.cross(b.omega, momentum)
    pressure_loss = atmosphere(b.pos[2]).pressure * p.propulsion.exit_area
    maximum = p.propulsion.vacuum_thrust - pressure_loss
    minimum = p.propulsion.min_throttle * p.propulsion.vacuum_thrust - pressure_loss
    thrust = wp.clamp(wp.length(force), minimum, maximum)
    fy = -torque[2] / length
    fz = (torque[1] - radius * thrust) / length
    g2 = wp.clamp(wp.atan2(fy, thrust), -p.propulsion.gimbal_limit, p.propulsion.gimbal_limit)
    g3 = wp.clamp(wp.atan2(fz, thrust), -p.propulsion.gimbal_limit, p.propulsion.gimbal_limit)
    rcs = wp.clamp((torque[0] + radius * fy) / p.propulsion.rcs_torque, -1.0, 1.0)
    if mirrored[i] != 0:
        g2 = -g2
        rcs = -rcs
    for j in range(ACTION_DIM):
        actions[i, j] = 0.0
    actions[i, 0] = 1.0
    for j in range(wp.static(len(CONTINUOUS))):
        value = wp.pi / 4.0
        if j == 0:
            value = (thrust + pressure_loss) / p.propulsion.vacuum_thrust
        elif j == 1:
            value = g2
        elif j == 2:
            value = g3
        elif j == 7:
            value = rcs
        actions[i, ENGINE_HEADS + j] = raw_action(value, ranges[j][0], ranges[j][1])
    actions[i, ARM_HEAD] = float(
        tower_step(
            p.catcher,
            p.aero,
            b,
            mp.com,
            s.flaps[i].angle,
            p.catcher.open_angle,
            p.dt,
        )
        < p.catcher.open_angle
    )


def teacher_actions(env):
    """Return policy-frame commands for a RocketEnv; does not advance or replace any state."""
    if not (
        env.config.catch.enabled and env.config.catch.moving and env.config.catch.policy_controlled
    ):
        raise ValueError("catch teacher requires policy-controlled moving arms")
    actions = wp.zeros((env.num_envs, ACTION_DIM), dtype=float, device=env.device)
    wp.launch(
        teach,
        dim=env.num_envs,
        inputs=[
            env.sim.params,
            env.sim.state,
            env.mirrored,
            env._ranges,
            actions,
        ],
        device=env.device,
    )
    return actions
