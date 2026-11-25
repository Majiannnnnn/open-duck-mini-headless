"""Pure single-environment V4 transitions on top of MuJoCo MJX."""

from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from miniduck.contract import load_contract, read_json
from miniduck.env.control import position_targets
from miniduck.env.curriculum import curriculum_at, push_velocity, sample_push_interval, validate_schedule
from miniduck.env.model import load_model
from miniduck.env.obs import gravity_in_body, imu_yaw, observations
from miniduck.env.randomize import (
    PhysicalSample, apply_physical_parameters, load_perturbations, sample_physical_parameters,
    validate_physical_ranges,
)
from miniduck.env.reference import load_reference
from miniduck.env.rewards import aggregate_reward, early_guidance, evaluate_reward, swing_profile
from miniduck.env.scene import foot_contacts_and_loads, load_scene, sole_heights
from miniduck.env.sensors import (
    SensorPerturbation, initial_imu_history, read_imu, sample_sensors, validate_sensor_config,
)


RUNNING, FALL, NONFINITE, TIME_LIMIT = range(4)
_ROOT = Path(__file__).resolve().parents[2]


class Episode(NamedTuple):
    key: jax.Array
    steps: jax.Array
    curriculum_age: jax.Array
    floor_friction: jax.Array
    first_command: jax.Array
    second_command: jax.Array
    switch_step: jax.Array
    start_phase: jax.Array
    command: jax.Array
    phase: jax.Array
    reference_heading: jax.Array
    command_origin_xy: jax.Array
    action_history: jax.Array
    sent_target: jax.Array
    applied_target: jax.Array
    target_queue: jax.Array
    contact: jax.Array
    feet_air_time: jax.Array
    velocity_window: jax.Array
    velocity_count: jax.Array
    physical_sample: PhysicalSample | None
    sensor_sample: SensorPerturbation | None
    imu_history: jax.Array | None
    push_interval: jax.Array | None
    command_scale: jax.Array


class State(NamedTuple):
    data: mjx.Data
    obs: dict[str, jax.Array]
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    termination_reason: jax.Array
    metrics: dict[str, jax.Array]
    task: Episode

    @property
    def done(self) -> jax.Array:
        return self.terminated | self.truncated


class DuckEnv:
    """Single V4 environment. JIT the bound transition functions for rollouts."""

    def __init__(self, project_root: Path | str = _ROOT, *, curriculum: bool | str | Path = False,
                 task_config: dict | None = None):
        root = Path(project_root).resolve()
        robot = read_json(root / "configs/robot.json")
        self.config = read_json(root / "configs/task.json") if task_config is None else deepcopy(task_config)
        self.contract = load_contract(robot)
        self.clock = self.contract.clock
        self.model, self.indices = load_model(root / robot["model"], robot)
        self.scene = load_scene(self.model, self.indices)
        self.mjx_model = mjx.put_model(self.model)
        self._empty_data = mjx.make_data(self.mjx_model)
        self.home_joints = jnp.asarray(robot["default_joint_positions_rad"], dtype=jnp.float32)
        self.joint_limits = jnp.asarray(robot["joint_limits_rad"], dtype=jnp.float32)
        configured_qpos = jnp.asarray(robot["initial_qpos_wxyz"], dtype=jnp.float32)
        self.initial_qpos = configured_qpos.at[self.indices.qpos_indices].set(self.home_joints)
        self._weight = jnp.sum(self.mjx_model.body_mass) * jnp.linalg.norm(self.mjx_model.opt.gravity)

        self.episode_length = self.config["episode_length"]
        self.command_config = self.config["command"]
        self.command_range = jnp.asarray(self.command_config["magnitude_range"], dtype=jnp.float32)
        self.active_threshold = self.contract.active_command_threshold
        self.reset_config = self.config["reset"]
        self.tilt_cosine = jnp.cos(jnp.deg2rad(self.reset_config["home_gravity_tilt_limit_deg"]))
        self.reward_weights = self.config["reward"]["weights"]
        self.swing_height = self.config["reward"]["swing_height_m"]
        self.friction_range = self.config["randomization"]["floor_friction_range"]
        self.delay_steps = self.config["randomization"]["action_delay_steps"]
        if type(self.delay_steps) is not int or self.delay_steps < 0:
            raise ValueError("Actuator delay must be a nonnegative integer number of control steps")
        self.curriculum = None
        self.physical_source = "off"
        self._bank_arrays = None
        self._physical_ranges = {}
        if curriculum is not False:
            profile = root / "configs/curriculum.json" if curriculum is True else root / Path(curriculum)
            self.curriculum = read_json(profile)
            validate_schedule(self.curriculum)
            validate_sensor_config(self.curriculum["sensors"])
            self.physical_source = self.curriculum["physics"]["source"]
            if self.physical_source not in ("off", "sampled", "bank"):
                raise ValueError("Physical curriculum source must be off, sampled or bank")
            if self.physical_source == "sampled":
                self._physical_ranges = validate_physical_ranges(self.curriculum["physics"]["ranges"])
            if self.physical_source == "bank":
                bank = load_perturbations(root / self.config["randomization"]["bank_path"])
                self._bank_arrays = {name: jnp.asarray(values) for name, values in bank.fields.items()}
        self.sensor_config = self.curriculum["sensors"] if self.curriculum and self.curriculum["sensors"]["enabled"] else None
        self.push_config = self.curriculum["push"] if self.curriculum and self.curriculum["push"]["enabled"] else None
        self._guidance = self.curriculum is not None and self.curriculum["guidance"]["weight"] > 0
        self._metric_terms = dict(self.reward_weights)
        if self._guidance:
            self._metric_terms["early_guidance"] = self.curriculum["guidance"]["weight"]

        ref_cfg = self.config["reference"]
        reference = load_reference(root / ref_cfg["path"], period_steps=self.clock.phase_period_steps)
        self.reference_commands = jnp.asarray(reference.commands)
        self.reference_frames = jnp.asarray(reference.frames)
        self.reference_offset = ref_cfg["phase_offset_when_active"]
        self.reference_threshold = ref_cfg["active_command_norm_gt"]
        self._motion_names = (
            "active_steps", "single_support_steps", "double_support_steps", "expected_swing_steps",
            "swing_clearance_m", "progress", "x_steps", "y_steps", "yaw_steps", "x_ratio", "y_ratio", "yaw_ratio",
        )

    def _active(self, command):
        return jnp.max(jnp.abs(command)) > self.active_threshold

    def _command_mode(self, command):
        axis = jnp.argmax(jnp.abs(command))
        return jnp.where(self._active(command), (axis + 1) * jnp.sign(command[axis]), 0).astype(jnp.int32)

    def _sample_command(self, key):
        axis_key, sign_key, magnitude_key = jax.random.split(key, 3)
        axis = jax.random.randint(axis_key, (), 0, 3)
        signed = jnp.where(jax.random.bernoulli(sign_key), 1.0, -1.0)
        low, high = self.command_range[axis]
        magnitude = jax.random.uniform(magnitude_key, (), minval=low, maxval=high)
        return jnp.zeros(3, dtype=jnp.float32).at[axis].set(signed * magnitude)

    def _command_at(self, step, first, second, switch_step):
        motion = jnp.where(step < switch_step, first, second)
        return jnp.where(step >= self.command_config["warmup_steps"], motion, jnp.zeros_like(first))

    def _physics(self, friction, sample=None, strength=1.0):
        model = self.mjx_model
        if sample is not None:
            model = apply_physical_parameters(model, sample, strength)
            nominal = self.mjx_model.geom_friction[self.scene.floor_geom_id, 0]
            friction = jnp.where(strength >= 1.0, friction, nominal + strength * (friction - nominal))
        material = model.geom_friction.at[self.scene.floor_geom_id, 0].set(friction)
        return model.replace(geom_friction=material)

    def _reference_frame(self, command, phase):
        magnitude = jnp.linalg.norm(command)
        phase_offset = jnp.where(magnitude > self.reference_threshold, self.reference_offset, 0)
        frame_idx = (phase + phase_offset) % self.clock.phase_period_steps
        distances = jnp.abs(self.reference_commands - command)
        cell = jnp.argmax(jnp.all(distances == distances.min(axis=0), axis=1))
        return self.reference_frames[cell, frame_idx]

    def _observe(self, data, task):
        return observations(
            data, task, self._reference_frame(task.command, task.phase),
            self.contract, self.indices, self.scene, self.home_joints, self.active_threshold,
        )

    def _course_metrics(self, course, task, impulse):
        if course is None:
            return {}
        return {
            "curriculum/age": task.curriculum_age.astype(jnp.float32),
            "curriculum/command_scale": task.command_scale,
            "curriculum/roll_scale": course.roll_scale,
            "curriculum/guidance_weight": course.guidance_weight,
            "curriculum/physics_strength": course.physics_strength,
            "curriculum/push_strength": course.push_strength,
            "curriculum/push_m_s": impulse,
        }

    def _initial_data(self, model, position_key, yaw_key, randomize_pose):
        qpos = self.initial_qpos
        if randomize_pose:
            xy_jitter = self.reset_config["xy_jitter_m"]
            yaw_jitter = self.reset_config["yaw_jitter_rad"]
            shift = jax.random.uniform(position_key, (2,), minval=-xy_jitter, maxval=xy_jitter)
            yaw = jax.random.uniform(yaw_key, (), minval=-yaw_jitter, maxval=yaw_jitter)
            w, x, y, z = qpos[3:7]
            cosine, sine = jnp.cos(yaw / 2), jnp.sin(yaw / 2)
            rotation = jnp.stack((cosine*w - sine*z, cosine*x - sine*y, cosine*y + sine*x, cosine*z + sine*w))
            qpos = qpos.at[:2].add(shift).at[3:7].set(rotation)
        ctrl = self._empty_data.ctrl.at[self.indices.actuator_ids].set(self.home_joints)
        data = self._empty_data.replace(qpos=qpos, ctrl=ctrl)
        data = mjx.kinematics(model, data)
        floor_gap = jnp.min(sole_heights(model, data, self.scene))
        qpos = qpos.at[2].add(self.reset_config["sole_clearance_m"] - floor_gap)
        return mjx.forward(model, data.replace(qpos=qpos))

    def reset(self, key, curriculum_age=0, *, command=None, start_phase=None, floor_friction=None,
              randomize_pose=True, physical_sample=None) -> State:
        """Sample an episode, optionally retaining the environment's existing physical draw."""
        key, first_key, second_key, switch_key, tick_key, stop_key, phase_key, friction_key, xy_key, yaw_key = (
            jax.random.split(key, 10)
        )
        age = jnp.int32(curriculum_age)
        course = curriculum_at(age, self.curriculum) if self.curriculum is not None else None
        command_scale = jnp.float32(1) if course is None else course.command_scale
        if command is None:
            first = self._sample_command(first_key) * command_scale
            second = jnp.where(
                jax.random.bernoulli(stop_key, self.command_config["stop_probability"]),
                jnp.zeros_like(first), self._sample_command(second_key) * command_scale,
            )
            start, end = self.command_config["switch_after_active_steps"]
            warmup = self.command_config["warmup_steps"]
            sampled_tick = jax.random.randint(tick_key, (), warmup + start, warmup + end + 1)
            switch_tick = jnp.where(
                jax.random.bernoulli(switch_key, self.command_config["switch_probability"]),
                sampled_tick, self.episode_length + 1,
            )
        else:
            first = second = jnp.asarray(command, dtype=jnp.float32)
            if first.shape != (3,):
                raise ValueError("A fixed command must have three channels")
            switch_tick = jnp.int32(self.episode_length + 1)

        if start_phase is None:
            phases = jnp.asarray(self.command_config["start_phases"], dtype=jnp.int32)
            start_phase = jax.random.choice(phase_key, phases)
        start_phase = jnp.int32(start_phase) % self.clock.phase_period_steps
        sensor_sample = None
        imu_history = None
        push_interval = None
        if course is not None:
            key, physics_key, sensor_key, push_key = jax.random.split(key, 4)
            if self.physical_source != "off" and physical_sample is None:
                physical_sample = sample_physical_parameters(
                    self.mjx_model, physics_key, self._physical_ranges,
                    joint_dof_indices=self.indices.qvel_indices, joint_qpos_indices=self.indices.qpos_indices,
                    actuator_ids=self.indices.actuator_ids, bank_arrays=self._bank_arrays,
                )
            if self.sensor_config is not None:
                sensor_sample = sample_sensors(sensor_key, self.sensor_config)
            if self.push_config is not None:
                push_interval = sample_push_interval(push_key, self.push_config["interval_s"], self.clock.control_dt_s)
        if floor_friction is None:
            if self.physical_source == "bank":
                floor_friction = physical_sample.geom_friction[self.scene.floor_geom_id, 0]
            else:
                low, high = self.friction_range
                floor_friction = jax.random.uniform(friction_key, (), minval=low, maxval=high)
        friction = jnp.float32(floor_friction)
        strength = 1.0 if course is None else course.physics_strength
        model = self._physics(friction, physical_sample, strength)
        data = self._initial_data(model, xy_key, yaw_key, randomize_pose)
        contact, _ = foot_contacts_and_loads(model, data, self.scene)
        queue_length = self.delay_steps + 1
        if sensor_sample is not None:
            queue_length = 3
            imu_history = initial_imu_history(
                data.sensordata[self.scene.gyro_indices], data.sensordata[self.scene.accelerometer_indices],
                sensor_sample, queue_length,
            )
        step = jnp.int32(0)
        initial_command = self._command_at(step, first, second, switch_tick)
        phase = jnp.where(self._active(initial_command), start_phase, step)
        task = Episode(
            key, step, age, friction, first, second, switch_tick, start_phase,
            initial_command, phase, imu_yaw(data, self.scene), data.qpos[:2],
            jnp.zeros((3, self.contract.action_size), dtype=jnp.float32),
            self.home_joints, self.home_joints,
            jnp.broadcast_to(self.home_joints, (queue_length, self.contract.action_size)),
            contact, jnp.zeros(2, dtype=jnp.float32),
            jnp.zeros((self.clock.phase_period_steps, 3), dtype=jnp.float32), jnp.int32(0),
            physical_sample, sensor_sample, imu_history, push_interval, command_scale,
        )
        zero = jnp.float32(0)
        metrics = {self._term_key(name): zero for name in self._metric_terms}
        metrics.update({f"reward_raw/{name}": zero for name in self._metric_terms})
        metrics.update({f"reward_contribution/{name}": zero for name in self._metric_terms})
        metrics.update(self._course_metrics(course, task, zero))
        metrics.update({f"motion/{name}": zero for name in self._motion_names})
        metrics.update({"episode/return": zero, "episode/steps": zero})
        return State(
            data, self._observe(data, task), zero, jnp.bool_(False), jnp.bool_(False), jnp.int32(RUNNING),
            metrics, task,
        )

    def _term_key(self, name):
        prefix = "reward" if self._metric_terms[name] > 0 else "cost"
        return f"{prefix}/{name}"

    def _advance(self, model, data, target):
        ctrl = data.ctrl.at[self.indices.actuator_ids].set(target)
        return jax.lax.fori_loop(
            0, self.clock.substeps, lambda _, current: mjx.step(model, current), data.replace(ctrl=ctrl),
        )

    def _motion_metrics(self, task, phase, contact, height, velocity, yaw_rate, progress):
        active = self._active(task.command)
        swing = swing_profile(phase) > 0
        direction = jnp.argmax(jnp.abs(task.command))
        speed = jnp.abs(task.command[direction])
        heading = task.reference_heading
        cosine, sine = jnp.cos(heading), jnp.sin(heading)
        measured = jnp.stack((
            cosine * velocity[0] + sine * velocity[1],
            -sine * velocity[0] + cosine * velocity[1],
            yaw_rate,
        ))
        ratio = measured[direction] * jnp.sign(task.command[direction]) / jnp.maximum(speed, self.active_threshold)
        matches_axis = jnp.arange(3) == direction
        values = (
            active, jnp.sum(contact) == 1, jnp.all(contact), jnp.any(swing),
            jnp.sum(jnp.where(swing, jnp.maximum(height, 0), 0)), progress,
            *matches_axis, *(matches_axis * ratio),
        )
        return {
            f"motion/{name}": jnp.where(active, jnp.asarray(value, dtype=jnp.float32), 0.0)
            for name, value in zip(self._motion_names, values, strict=True)
        }

    def step(self, state: State, action) -> State:
        """Advance one control tick. Reset explicitly after a terminal transition."""
        task = state.task
        action = jnp.asarray(action, dtype=jnp.float32)
        if action.shape != (self.contract.action_size,):
            raise ValueError(f"Expected {self.contract.action_size} action channels")
        valid_input = jnp.all(jnp.isfinite(action)) & self._physical_state_is_finite(state.data)
        active = self._active(task.command)
        bounded = jnp.where(active & valid_input, jnp.clip(action, *self.contract.action["clip"]), 0)
        reward_phase = jnp.where(active, (task.phase + 1) % self.clock.phase_period_steps, 0)
        dt = self.clock.control_dt_s
        delay = task.sensor_sample.action_delay if task.sensor_sample is not None else self.delay_steps
        sent, applied, queue = position_targets(
            bounded, task.sent_target, task.target_queue, delay, self.home_joints, self.joint_limits,
            scale=self.contract.action["scale_rad"], max_delta=self.contract.action["target_rate_rad_s"] * dt,
        )
        course = curriculum_at(task.curriculum_age, self.curriculum) if self.curriculum is not None else None
        strength = 1.0 if course is None else course.physics_strength
        model = self._physics(task.floor_friction, task.physical_sample, strength)
        key = task.key
        impulse_norm = jnp.float32(0)
        start_data = state.data
        if task.push_interval is not None:
            key, impulse = push_velocity(
                key, task.steps, task.push_interval, course.push_strength, self.push_config["delta_velocity_m_s"],
            )
            start_data = start_data.replace(qvel=start_data.qvel.at[:2].add(impulse))
            impulse_norm = jnp.linalg.norm(impulse)
        data = jax.lax.cond(
            valid_input, lambda _: self._advance(model, start_data, applied), lambda _: state.data, None,
        )
        finite = valid_input & self._physical_state_is_finite(data)
        contact, loads = foot_contacts_and_loads(model, data, self.scene)
        height = sole_heights(model, data, self.scene)
        gravity = gravity_in_body(data, self.scene)
        fallen = (gravity[2] > 0) | (jnp.dot(gravity, self.scene.home_gravity) < self.tilt_cosine)
        yaw = imu_yaw(data, self.scene)
        yaw_delta = yaw - imu_yaw(state.data, self.scene)
        yaw_rate = jnp.arctan2(jnp.sin(yaw_delta), jnp.cos(yaw_delta)) / dt
        velocity = (data.qpos[:2] - state.data.qpos[:2]) / dt
        feet_xy = data.geom_xpos[self.scene.foot_geom_ids, :2]
        previous_feet_xy = state.data.geom_xpos[self.scene.foot_geom_ids, :2]
        foot_speed = jnp.linalg.norm(feet_xy - previous_feet_xy, axis=1) / dt

        weight = self._weight
        if task.physical_sample is not None:
            weight = jnp.sum(model.body_mass) * jnp.linalg.norm(model.opt.gravity)
        _, terms, velocity_window, velocity_count, progress = evaluate_reward(
            task.command, task.reference_heading, task.command_origin_xy, data.qpos[:2], velocity, yaw, yaw_rate,
            foot_speed, loads, contact, task.contact, height, reward_phase, gravity, self.scene.home_gravity,
            data.sensordata[self.scene.gyro_indices], bounded, task.action_history[0], task.action_history[1],
            data.qpos[self.indices.qpos_indices], data.qvel[self.indices.qvel_indices], self.home_joints,
            data.actuator_force[self.indices.actuator_ids], fallen, weight, task.velocity_window,
            task.velocity_count, dt, self.swing_height, self.reward_weights,
        )
        weights = self.reward_weights
        if course is not None:
            weights = dict(self.reward_weights)
            weights["roll_stability"] = self.reward_weights["roll_stability"] * course.roll_scale
            if self._guidance:
                frame = self._reference_frame(task.command, reward_phase)
                terms["early_guidance"] = jnp.where(
                    active & ~fallen, early_guidance(frame, data.qpos[self.indices.qpos_indices], contact), 0.0,
                )
                weights["early_guidance"] = course.guidance_weight

        finite &= jnp.all(jnp.isfinite(jnp.stack(tuple(terms.values()))))
        failed = ~finite | fallen
        raw = {name: jnp.where(finite, value, float(name == "fall")) for name, value in terms.items()}
        reward, contributions = aggregate_reward(raw, weights, dt)
        steps = task.steps + 1
        truncated = ~failed & (steps >= self.episode_length)
        reason = jnp.where(~finite, NONFINITE, jnp.where(fallen, FALL, jnp.where(truncated, TIME_LIMIT, RUNNING)))
        metrics = {self._term_key(name): jnp.abs(weights[name]) * value for name, value in raw.items()}
        metrics.update({f"reward_raw/{name}": value for name, value in raw.items()})
        metrics.update({f"reward_contribution/{name}": value for name, value in contributions.items()})
        metrics.update(self._course_metrics(course, task, jnp.where(finite, impulse_norm, 0.0)))
        motion = self._motion_metrics(task, reward_phase, contact, height, velocity, yaw_rate, progress)
        metrics.update({name: jnp.where(finite, value, 0) for name, value in motion.items()})
        metrics.update({"episode/return": state.metrics["episode/return"] + reward, "episode/steps": steps.astype(jnp.float32)})

        command = self._command_at(steps, task.first_command, task.second_command, task.switch_step)
        next_active = self._active(command)
        mode_changed = self._command_mode(command) != self._command_mode(task.command)
        command_changed = jnp.any(command != task.command)
        history = jnp.concatenate((bounded[None], task.action_history[:2]), axis=0)
        phase = jnp.where(next_active, jnp.where(active, reward_phase, task.start_phase), 0)
        imu_history = task.imu_history
        if task.sensor_sample is not None:
            _, _, updated_history = read_imu(
                data.sensordata[self.scene.gyro_indices], data.sensordata[self.scene.accelerometer_indices],
                task.sensor_sample, imu_history,
            )
            imu_history = jnp.where(finite, updated_history, imu_history)
        next_task = task._replace(
            key=key, steps=steps, curriculum_age=task.curriculum_age + 1, command=command, phase=phase,
            imu_history=imu_history,
            reference_heading=jnp.where(mode_changed, yaw, task.reference_heading),
            command_origin_xy=jnp.where(mode_changed, data.qpos[:2], task.command_origin_xy),
            action_history=jnp.where(active != next_active, jnp.zeros_like(history), history),
            sent_target=sent, applied_target=applied, target_queue=queue,
            contact=contact, feet_air_time=jnp.where(contact, 0, task.feet_air_time + dt),
            velocity_window=jnp.where(command_changed, jnp.zeros_like(velocity_window), velocity_window),
            velocity_count=jnp.where(command_changed, 0, velocity_count),
        )
        obs = self._observe(data, next_task)
        safe_obs = jax.tree.map(lambda current, previous: jnp.where(finite, current, previous), obs, state.obs)
        return State(data, safe_obs, reward, failed, truncated, reason, metrics, next_task)

    @staticmethod
    def _physical_state_is_finite(data):
        return (
            jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
            & jnp.all(jnp.isfinite(data.qacc)) & jnp.all(jnp.isfinite(data.sensordata))
        )

    def step_and_reset(self, state: State, action) -> tuple[State, State]:
        """Return the completed transition and the state to pass to the next step."""
        transition = self.step(state, action)
        next_state = jax.lax.cond(
            transition.done,
            lambda item: self.reset(
                item.task.key, item.task.curriculum_age,
                floor_friction=item.task.floor_friction, physical_sample=item.task.physical_sample,
            ),
            lambda item: item,
            transition,
        )
        return transition, next_state
