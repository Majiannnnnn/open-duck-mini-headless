"""Pack actor inputs and clean privileged signals using the shared layout."""

import jax.numpy as jnp


def imu_yaw(data, scene):
    rotation = data.site_xmat[scene.imu_site_id]
    return jnp.arctan2(rotation[1, 0], rotation[0, 0])


def gravity_in_body(data, scene):
    return -data.site_xmat[scene.imu_site_id, 2]


def _pack(fields, values):
    parts = []
    for field in fields:
        channel = jnp.ravel(jnp.asarray(values[field.name], dtype=jnp.float32))
        if channel.shape != (field.stop - field.start,):
            raise ValueError(f"Observation field {field.name} does not match the contract")
        parts.append(channel)
    return jnp.concatenate(parts)


def observations(data, task, reference_frame, contract, indices, scene, home, active_threshold):
    gyro = data.sensordata[scene.gyro_indices]
    accel = data.sensordata[scene.accelerometer_indices]
    centered_q = data.qpos[indices.qpos_indices] - home
    dq = data.qvel[indices.qvel_indices]
    phase_angle = (2 * jnp.pi / contract.clock.phase_period_steps) * task.phase
    phase = jnp.stack((jnp.cos(phase_angle), jnp.sin(phase_angle)))
    translation_speed = jnp.linalg.norm(task.command[:2])
    translating = (translation_speed > active_threshold) & (jnp.abs(task.command[2]) < active_threshold)
    heading_error = jnp.where(translating, imu_yaw(data, scene) - task.reference_heading, 0.0)
    heading_features = jnp.stack((jnp.sin(heading_error), jnp.cos(heading_error)))

    actor_gyro, actor_accel = gyro, accel
    actor_q = centered_q
    if task.sensor_sample is not None:
        imu = task.imu_history[task.sensor_sample.imu_delay]
        actor_gyro, actor_accel = imu[:3], imu[3:]
        actor_q = centered_q + task.sensor_sample.encoder_bias

    actor = _pack(contract.actor_fields, {
        "gyro": actor_gyro,
        "accelerometer": actor_accel,
        "command_xyz_yaw": task.command,
        "q_minus_default": actor_q,
        "dq_scaled": dq * contract.observations["joint_velocity_scale"],
        "last_action": task.action_history[0],
        "last_last_action": task.action_history[1],
        "last_last_last_action": task.action_history[2],
        "last_motor_targets": task.sent_target,
        "contacts_left_right": task.contact,
        "phase": phase,
        "relative_heading_sin_cos": heading_features,
    })

    heading = task.reference_heading
    forward = jnp.stack((jnp.cos(heading), jnp.sin(heading)))
    left = jnp.stack((-forward[1], forward[0]))
    command_unit = task.command[:2] / jnp.maximum(translation_speed, active_threshold)
    travel = command_unit[0] * forward + command_unit[1] * left
    normal = jnp.stack((-travel[1], travel[0]))
    cross_track = jnp.where(translating, jnp.dot(data.qpos[:2] - task.command_origin_xy, normal), 0.0)

    privileged = _pack(contract.critic_fields[len(contract.actor_fields):], {
        "clean_gyro": gyro,
        "clean_accelerometer": accel,
        "gravity_body": gravity_in_body(data, scene),
        "local_linear_velocity": data.sensordata[scene.local_linear_velocity_indices],
        "global_angular_velocity": data.sensordata[scene.global_angular_velocity_indices],
        "clean_q_minus_default": centered_q,
        "clean_dq": dq,
        "root_height": data.qpos[2],
        "actuator_force": data.actuator_force[indices.actuator_ids],
        "contacts": task.contact,
        "foot_velocities_world": data.sensordata[scene.foot_velocity_indices],
        "foot_air_times": task.feet_air_time,
        "reference_frame": reference_frame,
        "phase_index": task.phase,
        "imitation_phase": phase,
        "cross_track": cross_track,
    })
    return {contract.actor_key: actor, contract.critic_key: jnp.concatenate((actor, privileged))}
