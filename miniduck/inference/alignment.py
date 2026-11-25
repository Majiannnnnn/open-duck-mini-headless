"""Deterministic checks for the training and deployment policy interface."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from miniduck.contract import read_json
from miniduck.env.control import position_targets
from miniduck.env.env import DuckEnv
from miniduck.env.reference import load_reference
from miniduck.inference.runtime import Runtime, pack_actor_numpy


@dataclass(frozen=True)
class AlignmentResult:
    actor_fields: int
    critic_fields: int
    delay_steps: tuple[int, ...]
    deployment_ticks: int
    observations: np.ndarray


def _check(actual, expected, label):
    actual = np.asarray(actual)
    expected = np.asarray(expected, dtype=actual.dtype)
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: expected shape {expected.shape}, got {actual.shape}")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6, err_msg=label)


def _clock_features(phase, period):
    angle = 2 * np.pi * phase / period
    return np.array([np.cos(angle), np.sin(angle)])


def _actor_field(contract, name, observation):
    field = next(field for field in contract.actor_fields if field.name == name)
    return observation[field.indices]


def _rotation(yaw):
    # Use tilted body coordinates so a transposed gravity/sensor axis is visible.
    pitch = 0.22
    roll = -0.16
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array(
        [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    ) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])


def _check_observations(env, state, reference):
    contract, indices, scene = env.contract, env.indices, env.scene
    home = np.asarray(env.home_joints)
    gyro = np.array([0.35, -0.40, 0.62], dtype=np.float32)
    accel = np.array([-0.91, 1.15, 9.72], dtype=np.float32)
    local_velocity = np.array([0.032, -0.044, 0.018], dtype=np.float32)
    angular_velocity = np.array([0.015, -0.028, 0.31], dtype=np.float32)
    foot_velocity = np.array([[0.11, -0.05, 0.04], [-0.04, 0.08, -0.12]], dtype=np.float32)
    joints = home + np.linspace(-0.04, 0.04, 10, dtype=np.float32)
    joint_velocity = np.linspace(-0.72, 1.08, 10, dtype=np.float32)
    actor_history = np.linspace(-0.8, 0.9, 30, dtype=np.float32).reshape(3, 10)
    last_sent = home + np.linspace(0.02, -0.03, 10, dtype=np.float32)
    forces = np.linspace(-0.23, 0.36, 10, dtype=np.float32)
    world_xyz = np.array([0.41, -0.29, 0.157], dtype=np.float32)
    origin = np.array([0.37, -0.31], dtype=np.float32)
    foot_air_time = np.array([0.0, 0.18], dtype=np.float32)
    yaw, reference_heading = -0.37, -0.81
    rotation = _rotation(yaw)
    data = state.data.replace(
        qpos=state.data.qpos.at[:3].set(world_xyz).at[indices.qpos_indices].set(joints),
        qvel=state.data.qvel.at[indices.qvel_indices].set(joint_velocity),
        site_xmat=state.data.site_xmat.at[scene.imu_site_id].set(rotation.astype(np.float32)),
        sensordata=(
            state.data.sensordata.at[scene.gyro_indices].set(gyro)
            .at[scene.accelerometer_indices].set(accel)
            .at[scene.local_linear_velocity_indices].set(local_velocity)
            .at[scene.global_angular_velocity_indices].set(angular_velocity)
            .at[scene.foot_velocity_indices].set(foot_velocity)
        ),
        actuator_force=state.data.actuator_force.at[indices.actuator_ids].set(forces),
    )
    period = contract.clock.phase_period_steps
    fixtures = (
        ((0.06, 0, 0), (True, False), 7),
        ((0, -0.07, 0), (False, True), 22),
        ((0, 0, -0.25), (True, True), 36),
        ((0, 0, 0), (False, False), 0),
    )
    observations = []
    for velocity, contacting, phase in fixtures:
        command = np.array(velocity, dtype=np.float32)
        contacts = np.array(contacting)
        task = state.task._replace(
            command=jnp.asarray(command), phase=jnp.int32(phase),
            reference_heading=jnp.float32(reference_heading), command_origin_xy=jnp.asarray(origin),
            action_history=jnp.asarray(actor_history), sent_target=jnp.asarray(last_sent),
            contact=jnp.asarray(contacts), feet_air_time=jnp.asarray(foot_air_time),
        )
        training = env._observe(data, task)
        actor = np.asarray(training[contract.actor_key])
        critic = np.asarray(training[contract.critic_key])
        runtime = pack_actor_numpy(
            contract, gyro=gyro, accelerometer=accel, command=command, joint_positions=joints,
            joint_velocities=joint_velocity, home=home, action_history=actor_history, last_sent=last_sent,
            contacts=contacts, phase=phase, yaw=yaw, reference_heading=reference_heading,
        )
        translation = bool(command[0] or command[1])
        heading_error = yaw - reference_heading if translation else 0
        phase_features = _clock_features(phase, period)
        expected_actor = {
            "gyro": gyro, "accelerometer": accel, "command_xyz_yaw": command,
            "q_minus_default": joints - home, "dq_scaled": 0.05 * joint_velocity,
            "last_action": actor_history[0], "last_last_action": actor_history[1],
            "last_last_last_action": actor_history[2], "last_motor_targets": last_sent,
            "contacts_left_right": contacts, "phase": phase_features,
            "relative_heading_sin_cos": [np.sin(heading_error), np.cos(heading_error)],
        }
        for field in contract.actor_fields:
            expected = expected_actor[field.name]
            _check(actor[field.indices], expected, f"JAX actor {field.name}")
            _check(runtime[field.indices], expected, f"NumPy actor {field.name}")

        np.testing.assert_array_equal(critic[:contract.actor_size], actor)
        active = np.any(command)
        frame_phase = (phase + 15) % period if active else 0
        # For these single-axis commands the path normal is the left rotation
        # of the travel direction expressed in the original heading frame.
        travel_yaw = reference_heading + np.arctan2(command[1], command[0])
        path_normal = np.array([-np.sin(travel_yaw), np.cos(travel_yaw)])
        cross_track = np.dot(world_xyz[:2] - origin, path_normal) if translation else 0
        expected_suffix = {
            "clean_gyro": gyro, "clean_accelerometer": accel, "gravity_body": -rotation[2],
            "local_linear_velocity": local_velocity, "global_angular_velocity": angular_velocity,
            "clean_q_minus_default": joints - home, "clean_dq": joint_velocity,
            "root_height": [world_xyz[2]], "actuator_force": forces, "contacts": contacts,
            "foot_velocities_world": foot_velocity.ravel(), "foot_air_times": foot_air_time,
            "reference_frame": reference.frame(command, frame_phase), "phase_index": [phase],
            "imitation_phase": phase_features, "cross_track": [cross_track],
        }
        for field in contract.critic_fields[len(contract.actor_fields):]:
            _check(critic[field.indices], expected_suffix[field.name], f"critic {field.name}")
        if actor.dtype != np.float32 or critic.dtype != np.float32 or runtime.dtype != np.float32:
            raise AssertionError("Policy observations must be float32")
        observations.append(runtime)
    return observations


def _expected_target(home, limits, last_sent, action):
    requested = home + 0.25 * np.clip(action, -1, 1)
    limited = np.clip(requested, last_sent - 0.1048, last_sent + 0.1048)
    return np.clip(limited, limits[:, 0], limits[:, 1])


def _check_commands(env, reset, step, initial, reference):
    contract = env.contract
    period = contract.clock.phase_period_steps
    home = np.asarray(env.home_joints)
    limits = np.asarray(env.joint_limits)
    first_action = np.linspace(-1.3, 1.4, contract.action_size, dtype=np.float32)
    commands = (first_action, -first_action, np.ones_like(first_action), np.zeros_like(first_action))
    for delay in (0, 1, 2):
        previous = home
        history = [home] * (delay + 1)
        queue = jnp.broadcast_to(env.home_joints, (delay + 1, contract.action_size))
        for action in commands:
            expected = _expected_target(home, limits, previous, action)
            history = [expected, *history[:-1]]
            sent, applied, queue = position_targets(
                jnp.clip(action, -1, 1), previous, queue, delay, env.home_joints, env.joint_limits,
                scale=contract.action["scale_rad"], max_delta=contract.action["target_rate_rad_s"] * contract.clock.control_dt_s,
            )
            _check(sent, expected, f"delay {delay} sent")
            _check(applied, history[delay], f"delay {delay} applied")
            _check(queue, history, f"delay {delay} target history")
            previous = expected

    # Exercise the real transition at the decision boundary for both training
    # start phases, then trace the delayed target through the MJX ctrl field.
    command = np.asarray(initial.task.first_command)
    for expected_start in (0, 20):
        state = initial if expected_start == 20 else reset(jax.random.key(7), jnp.int32(expected_start))
        before_warmup_end = state.task._replace(steps=jnp.int32(env.command_config["warmup_steps"] - 1))
        state = step(state._replace(task=before_warmup_end), jnp.zeros(contract.action_size))
        if int(state.task.phase) != expected_start:
            raise AssertionError(f"Training start phase {expected_start} was not used")
        _check(
            _actor_field(contract, "phase", state.obs[contract.actor_key]),
            _clock_features(expected_start, period), "training start actor phase",
        )
        frame = next(field for field in contract.critic_fields if field.name == "reference_frame")
        _check(
            state.obs[contract.critic_key][frame.indices],
            reference.frame(command, (expected_start + 15) % period), "training start shifted reference",
        )

    previous = home
    history = [home] * (env.delay_steps + 1)
    for action in commands:
        state = step(state, action)
        expected = _expected_target(home, limits, previous, action)
        history = [expected, *history[:-1]]
        applied = history[env.delay_steps]
        _check(state.task.sent_target, expected, "training sent target")
        _check(state.task.applied_target, applied, "training applied target")
        _check(state.data.ctrl[env.indices.actuator_ids], applied, "MJX control used delayed target")
        _check(
            _actor_field(contract, "last_motor_targets", state.obs[contract.actor_key]),
            expected, "training actor observed sent target",
        )
        previous = expected


def _check_deployment(robot, contract):
    runtime = Runtime(robot)
    home = np.asarray(robot["default_joint_positions_rad"], dtype=np.float32)
    limits = np.asarray(robot["joint_limits_rad"], dtype=np.float32)
    period = contract.clock.phase_period_steps
    gyro, accel = [0.01, -0.03, 0.12], [0.65, 0.12, 9.72]
    velocities = np.linspace(-1.0, 1.2, contract.action_size)
    contacts = [1, 0]
    observations = []
    actions = (
        np.linspace(-1.25, 1.33, contract.action_size),
        np.linspace(0.83, -0.85, contract.action_size),
        np.full(contract.action_size, 0.67),
    )

    def prepare(command, yaw):
        return runtime.prepare(command, gyro, accel, home, velocities, contacts, yaw)

    begin = prepare([0.04, 0, 0], 0.12)
    unsent = runtime.propose(begin, actions[0])
    _check(runtime.last_sent, home, "failed send kept last target")
    _check(runtime.action_history, np.zeros((3, contract.action_size)), "failed send kept action history")
    if runtime.phase != 0 or runtime.mode != 0:
        raise AssertionError("An uncommitted send changed phase or command mode")

    state = prepare([0.04, 0, 0], 0.23)
    _check(state.reference_heading, 0.23, "retry heading latched from fresh sample")
    plan = (
        ([0.04, 0, 0], 0.23, 0.23, 0, 1, actions[0]),
        ([0.075, 0, 0], 0.41, 0.23, 1, 1, actions[1]),
        ([0, 0.06, 0], 0.62, 0.62, 2, 2, actions[2]),
        ([0, -0.06, 0], 0.71, 0.71, 3, -2, actions[0]),
        ([0, 0, 0.30], 0.84, 0.84, 4, 3, actions[1]),
        ([0, 0, 0.30], 1.15, 0.84, 5, 3, actions[2]),
    )
    history = np.zeros((3, contract.action_size), dtype=np.float32)
    previous = home
    for command, yaw, heading, phase, mode, action in plan:
        state = prepare(command, yaw)
        if (state.phase, state.mode) != (phase, mode):
            raise AssertionError(f"Deployment command {command} produced phase/mode {(state.phase, state.mode)}")
        _check(state.reference_heading, heading, "deployment heading latch")
        actor = state.observation
        _check(_actor_field(contract, "phase", actor), _clock_features(phase, period), "deployment actor phase")
        error = yaw - heading if mode != 3 else 0
        _check(
            _actor_field(contract, "relative_heading_sin_cos", actor),
            [np.sin(error), np.cos(error)], "deployment heading features",
        )
        for row, name in enumerate(("last_action", "last_last_action", "last_last_last_action")):
            _check(_actor_field(contract, name, actor), history[row], f"deployment {name}")
        _check(_actor_field(contract, "last_motor_targets", actor), previous, "deployment last sent")
        target = runtime.propose(state, action)
        expected = _expected_target(home, limits, previous, action)
        _check(target.sent, expected, "deployment position limit")
        runtime.commit(target)
        _check(runtime.last_sent, expected, "deployment committed target")
        if runtime.phase != (phase + 1) % period:
            raise AssertionError("Committed active target did not advance the deployment phase")
        observations.append(actor)
        history = np.vstack((np.clip(action, -1, 1), history[:2]))
        previous = expected

    try:
        runtime.commit(unsent)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("Deployment accepted a stale send candidate")

    for phase in range(len(plan), period):
        state = prepare([0, 0, 0.30], 1.35)
        if state.phase != phase:
            raise AssertionError("Deployment phase did not advance once per active tick")
        runtime.commit(runtime.propose(state, actions[2]))
    if runtime.phase != 0 or runtime.mode != 3:
        raise AssertionError("Deployment did not wrap phase while keeping its active mode")

    turned = prepare([0, 0, 0.30], 1.35)
    runtime.commit(runtime.propose(turned, actions[2]))
    state = prepare([0, 0, 0], 1.35)
    _check(_actor_field(contract, "phase", state.observation), [1, 0], "deployment idle phase")
    _check(_actor_field(contract, "relative_heading_sin_cos", state.observation), [0, 1], "deployment idle heading")
    _check(state.observation[29:59], np.zeros(30), "deployment stop resets histories")
    target = runtime.propose(state, actions[0])
    expected = _expected_target(home, limits, runtime.last_sent, np.zeros(contract.action_size))
    _check(target.sent, expected, "deployment idle rate-limited home target")
    runtime.commit(target)

    restart = prepare([-0.06, 0, 0], 1.40)
    if restart.phase != 0 or restart.mode != -1:
        raise AssertionError("Deploying after stop did not start at phase zero")
    _check(restart.observation[29:59], np.zeros(30), "deployment restart action histories")
    _check(restart.reference_heading, 1.40, "deployment restart heading")
    return [*observations, state.observation, restart.observation]


def check_alignment(project_root: Path | str) -> AlignmentResult:
    """Check both observation paths, motor timing and transactional deployment."""
    root = Path(project_root)
    robot = read_json(root / "configs/robot.json")
    env = DuckEnv(root)
    env.delay_steps = 2
    reference = load_reference(root / env.config["reference"]["path"], period_steps=env.clock.phase_period_steps)
    reset = jax.jit(lambda key, phase: env.reset(
        key, command=jnp.array([0.06, 0, 0]), start_phase=phase, floor_friction=0.6, randomize_pose=False,
    ))
    initial = reset(jax.random.key(7), jnp.int32(20))
    fixtures = _check_observations(env, initial, reference)
    _check_commands(env, reset, jax.jit(env.step), initial, reference)
    deployment = _check_deployment(robot, env.contract)
    return AlignmentResult(
        len(env.contract.actor_fields), len(env.contract.critic_fields) - len(env.contract.actor_fields),
        (0, 1, 2), len(deployment), np.stack([*fixtures, *deployment]),
    )
