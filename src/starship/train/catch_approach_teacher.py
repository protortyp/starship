"""Demonstrations for upright approaches and ignition, through normal policy actions.

Uses one or three engines, aligns above the open arms, then descends through the capture
window. This supplies training labels only; it is not an inference override or an entry controller.
"""

import warp as wp

from starship.sim.atmosphere import atmosphere
from starship.sim.catch import catch_center, pin, pin_error, tower_step
from starship.sim.mass import mass_properties
from starship.sim.physics import Params, State
from starship.sim.rigid import gravity, orientation
from starship.train.actions import ACTION_DIM, ARM_HEAD, CONTINUOUS, ENGINE_HEADS
from starship.train.catch_teacher import raw_action

wp.set_module_options({"enable_backward": False})


@wp.kernel
def teach_approach(
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
    height = center[2] - p.catcher.height
    error_xy = pin_error(p.catcher, b, mp.com)
    horizontal_speed = wp.length(wp.vec2(pin_velocity[0], pin_velocity[1]))
    height_difference = wp.abs(
        pin(p.catcher, b, mp.com, -1.0)[2] - pin(p.catcher, b, mp.com, 1.0)[2]
    )
    aligned = error_xy < 0.3 and horizontal_speed < 0.15 and height_difference < 0.25
    hold_height = 10.0
    if not aligned and height > 25.0:
        hold_height = 35.0
    desired_vz = wp.clamp(0.8 * (hold_height - height), -5.0, 5.0)
    if height > hold_height:
        desired_vz = -wp.min(wp.sqrt(1.6 * (height - hold_height)), 20.0)
        if height < 45.0:
            desired_vz = wp.max(desired_vz, -5.0)
    if height < 12.0 and (aligned or height < 7.0):
        desired_vz = -1.3
    acceleration = wp.vec3(
        -0.12 * center[0] - 0.8 * pin_velocity[0],
        -0.12 * center[1] - 0.8 * pin_velocity[1],
        1.3 * (desired_vz - b.vel[2]),
    )
    force = mp.mass * (acceleration + wp.vec3(0.0, 0.0, gravity(b.pos[2])))
    length = mp.com - p.propulsion.pivot_station
    engine = s.engine[i]
    active = wp.max(engine.spool[0] + engine.spool[1] + engine.spool[2], 0.1)
    radius = (
        p.propulsion.pivot_radius
        * (engine.spool[0] - 0.5 * engine.spool[1] - 0.5 * engine.spool[2])
        / active
    )
    base = orientation(0.0, 0.0)
    # Follow the actual spool-weighted engine centroid while engines ignite or shut down.
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
    three = wp.length(force) > maximum * 1.05
    if engine.spool[1] > 0.2 and wp.length(force) > maximum * 0.9:
        three = True
    thrust = wp.clamp(wp.length(force), minimum * active, maximum * active)
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
    actions[i, 1] = float(three)
    actions[i, 2] = float(three)
    for j in range(wp.static(len(CONTINUOUS))):
        value = wp.pi / 4.0
        if j == 0:
            value = (thrust / active + pressure_loss) / p.propulsion.vacuum_thrust
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
        and error_xy < 0.5
        and wp.abs(b.vel[2]) < 2.5
        and height < 12.0
    )


def teacher_actions(env):
    """Return policy-frame commands for a RocketEnv; does not advance or replace any state."""
    if not (
        env.config.catch.enabled and env.config.catch.moving and env.config.catch.policy_controlled
    ):
        raise ValueError("catch teacher requires policy-controlled moving arms")
    if abs(env.config.spawn.pitch) > 0.3 or env.config.spawn.entry:
        raise ValueError("approach teacher requires upright, non-entry spawns")
    actions = wp.zeros((env.num_envs, ACTION_DIM), dtype=float, device=env.device)
    wp.launch(
        teach_approach,
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
