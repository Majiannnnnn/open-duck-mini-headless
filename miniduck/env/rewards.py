"""Task feedback for the V4 single-axis locomotion environment."""

import jax.numpy as jnp


def _wrapped(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def swing_profile(phase):
    """Return the lift envelope for the left and right foot at this gait tick."""
    offsets = jnp.asarray((3, 23))
    swing_tick = phase % 40 - offsets
    inside = (0 <= swing_tick) & (swing_tick < 13)
    height_fraction = jnp.sin(jnp.pi * (swing_tick + 0.5) / 13)
    return jnp.where(inside, height_fraction, 0.0)


def _roll_cost(gravity, home_gravity, gyro):
    up = -gravity / jnp.maximum(jnp.linalg.norm(gravity), 1e-6)
    roll = jnp.arctan2(up[1], up[2])
    home_roll = jnp.arctan2(-home_gravity[1], -home_gravity[2])
    excess = jnp.maximum(jnp.abs(_wrapped(roll - home_roll)) - jnp.deg2rad(8.0), 0.0)
    up_rate = jnp.cross(up, gyro)
    roll_rate = (up[2] * up_rate[1] - up[1] * up_rate[2]) / jnp.maximum(
        up[1] ** 2 + up[2] ** 2, 0.1
    )
    angle_penalty = 2.0 * (jnp.hypot(1.0, excess / jnp.deg2rad(4.0)) - 1.0)
    rate_penalty = 2.0 * (jnp.hypot(1.0, roll_rate / 0.6) - 1.0)
    return angle_penalty + 0.1 * rate_penalty


def aggregate_reward(terms, reward_weights, dt):
    """Return the clipped reward and signed weighted contributions per tick."""
    weighted = {name: terms[name] * scale for name, scale in reward_weights.items()}
    # Keep the original V4 summation order and apply dt once to the total.
    reward = jnp.clip(sum(weighted.values()) * dt, -10_000.0, 10_000.0)
    contributions = {name: value * dt for name, value in weighted.items()}
    return reward, contributions


def early_guidance(reference_frame, joint_positions, contact):
    """Score reference leg pose and left/right contacts with equal weight."""
    target = jnp.concatenate((reference_frame[:5], reference_frame[11:16]))
    squared_error = jnp.sum(jnp.square(joint_positions - target))
    pose_match = jnp.exp(-15.0 * squared_error)
    contact_match = jnp.mean(contact == (reference_frame[32:34] > 0.5))
    return 0.5 * (pose_match + contact_match)


def evaluate_reward(
    command,
    heading,
    origin_xy,
    root_xy,
    root_velocity_xy,
    yaw,
    yaw_rate,
    foot_speed,
    loads,
    contact,
    last_contact,
    sole_height,
    phase,
    gravity_body,
    home_gravity,
    gyro,
    action,
    last_action,
    second_last_action,
    joint_positions,
    joint_velocities,
    home_joints,
    actuator_force,
    failed,
    weight,
    velocity_window,
    velocity_count,
    dt,
    swing_height_m,
    reward_weights,
):
    """Return signed reward, raw terms, updated progress history and progress.

    The window holds the most recent along-track, cross-track and signed yaw
    velocity samples. Callers clear it when the command changes.
    """
    translation_speed = jnp.linalg.norm(command[:2])
    turning = jnp.abs(command[2]) > 0.01
    translating = (translation_speed > 0.01) & ~turning
    active = translating | turning

    forward = jnp.stack((jnp.cos(heading), jnp.sin(heading)))
    leftward = jnp.stack((-forward[1], forward[0]))
    requested_direction = (
        command[0] * forward + command[1] * leftward
    ) / jnp.maximum(translation_speed, 1e-6)
    travel_direction = jnp.where(translating, requested_direction, forward)
    cross_direction = jnp.stack((-travel_direction[1], travel_direction[0]))

    ground_load = jnp.where(contact, loads, 0.0)
    stable_load = jnp.where(last_contact, ground_load, 0.0)
    stable_total = jnp.sum(stable_load)
    supported = stable_total > 0.05 * weight
    slip_speed = jnp.dot(stable_load, foot_speed) / jnp.maximum(stable_total, 1e-6)

    travel_speed = jnp.dot(root_velocity_xy, travel_direction)
    cross_speed = jnp.dot(root_velocity_xy, cross_direction)
    signed_turn_speed = yaw_rate * jnp.sign(command[2])
    # Motion in the wrong direction always counts. Positive credit requires
    # established ground support and subtracts translational sole slip.
    travel_credit = jnp.minimum(travel_speed, 0.0) + jnp.where(
        supported, jnp.maximum(travel_speed - slip_speed, 0.0), 0.0
    )
    turn_credit = jnp.where(supported, signed_turn_speed, jnp.minimum(signed_turn_speed, 0.0))
    sample = jnp.stack((jnp.where(translating, travel_credit, travel_speed), cross_speed, turn_credit))
    updated_window = jnp.concatenate((sample[None, :], velocity_window[:-1]), axis=0)
    updated_count = jnp.minimum(velocity_count + 1, velocity_window.shape[0])
    mean_travel, mean_cross, mean_turn = jnp.sum(updated_window, axis=0) / updated_count

    speed_reference = jnp.where(translating, jnp.maximum(translation_speed, 0.01), 0.05)
    turn_reference = jnp.maximum(jnp.abs(command[2]), 0.01)
    progress_ratio = jnp.where(turning, mean_turn / turn_reference, mean_travel / speed_reference)
    progress = jnp.clip(1.0 - jnp.abs(progress_ratio - 1.0), -1.0, 1.0)
    progress = jnp.where(supported, progress, jnp.minimum(progress, 0.0))
    progress = jnp.where(active, progress, 0.0)

    swing = swing_profile(phase)
    stance_load = ground_load[::-1]
    stance_share = stance_load / jnp.maximum(jnp.sum(ground_load), 1e-6)
    stance_quality = jnp.clip(stance_load / (0.35 * weight), 0.0, 1.0)
    weight_shift = jnp.clip((stance_share - 0.5) / 0.4, 0.0, 1.0) * stance_quality
    clearance_target = jnp.maximum(0.004, swing_height_m * swing)
    lift = jnp.clip((sole_height - 0.001) / (clearance_target - 0.001), 0.0, 1.0)
    clearance = lift * stance_quality * jnp.logical_not(contact)

    displacement = root_xy - origin_xy
    cross_offset = jnp.dot(displacement, cross_direction)
    line_error = (mean_cross**2 + cross_offset**2) / (2.0 * 0.05**2)
    turn_error = (
        mean_travel**2 + mean_cross**2 + jnp.dot(displacement, displacement)
    ) / (2.0 * 0.05**2)
    drift = jnp.clip(jnp.where(turning, turn_error, line_error), 0.0, 4.0)
    heading_error = jnp.clip((_wrapped(yaw - heading) / jnp.deg2rad(15.0)) ** 2, 0.0, 4.0)

    action_delta = action - last_action
    previous_action_delta = last_action - second_last_action
    pose_error = jnp.sum(jnp.abs(joint_positions - home_joints))
    joint_speed = jnp.sum(jnp.abs(joint_velocities))
    still_cost = jnp.nan_to_num(pose_error + joint_speed)
    terms = {
        "alive": jnp.logical_not(failed).astype(action.dtype),
        "tracking_lin_vel": progress,
        "weight_transfer": jnp.where(active, jnp.dot(swing, weight_shift), 0.0),
        "swing_clearance": jnp.where(active, jnp.dot(swing, clearance), 0.0),
        "stance_slip": jnp.where(active & supported, jnp.clip(slip_speed / speed_reference, 0.0, 2.0), 0.0),
        "lateral_error": jnp.where(active, drift, 0.0),
        "heading_error": jnp.where(translating, heading_error, 0.0),
        "action_rate": jnp.mean(action_delta**2),
        "action_acceleration": jnp.mean((action_delta - previous_action_delta) ** 2),
        "torques": jnp.nan_to_num(jnp.sum(actuator_force**2)),
        "roll_stability": _roll_cost(gravity_body, home_gravity, gyro),
        "fall": jnp.asarray(failed, dtype=action.dtype),
        "stand_still": jnp.where(jnp.linalg.norm(command) < 0.01, still_cost, 0.0),
    }
    reward, _ = aggregate_reward(terms, reward_weights, dt)
    return reward, terms, updated_window, updated_count, progress
