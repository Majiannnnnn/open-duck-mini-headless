"""Run deterministic acceptance checks for the configured curriculum."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from miniduck.env.curriculum import curriculum_at
from miniduck.env.env import DuckEnv
from miniduck.env.rewards import early_guidance
from miniduck.env.scene import sole_heights


def _same_tree(left, right):
    for first, second in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(first, second)


def _reward_from_log(state):
    values = [float(value) for name, value in state.metrics.items() if name.startswith("reward_contribution/")]
    reconstructed = np.clip(sum(values), -10_000, 10_000)
    np.testing.assert_allclose(reconstructed, state.reward, rtol=1e-6, atol=1e-7)


def check_curriculum(root: Path) -> None:
    env = DuckEnv(root, curriculum=True)
    config = env.curriculum
    schedule = jax.jit(lambda age: curriculum_at(age, config))
    low = schedule(0)
    high_age = max(config[name]["ramp_steps"] for name in ("command", "roll", "physics"))
    high = schedule(high_age)
    np.testing.assert_allclose(
        (low.command_scale, low.roll_scale, low.physics_strength, low.push_strength),
        (config["command"]["initial_scale"], config["roll"]["initial_scale"],
         config["physics"]["initial_strength"], 0),
    )
    np.testing.assert_allclose((high.command_scale, high.roll_scale, high.physics_strength, high.push_strength), 1)
    np.testing.assert_allclose(schedule(config["command"]["ramp_steps"] // 2).command_scale,
                               (config["command"]["initial_scale"] + 1) / 2)

    start, end = config["guidance"]["fade_steps"]
    guidance_config = dict(config, guidance=dict(config["guidance"], weight=2.0))
    guidance = jax.jit(lambda age: curriculum_at(age, guidance_config).guidance_weight)
    np.testing.assert_allclose([guidance(start), guidance((start + end) // 2), guidance(end)], [2, 1, 0])
    frame = jnp.zeros(40).at[:5].set(.2).at[11:16].set(-.3).at[32:34].set(jnp.array([.8, .1]))
    joints = jnp.concatenate((frame[:5], frame[11:16]))
    np.testing.assert_allclose(jax.jit(early_guidance)(frame, joints, jnp.array([True, False])), 1)
    print("Schedule: command, roll, physics, impulse and guidance endpoints verified", flush=True)

    key = jax.random.key(22)
    reset = jax.jit(env.reset)
    state = reset(key, 0)
    repeat = reset(key, 0)
    mature = reset(key, high_age)
    _same_tree(state.task.physical_sample, repeat.task.physical_sample)
    _same_tree(state.task.sensor_sample, repeat.task.sensor_sample)
    _same_tree(state.task.physical_sample, mature.task.physical_sample)
    np.testing.assert_array_equal(state.obs["state"], repeat.obs["state"])
    np.testing.assert_allclose(state.task.first_command, mature.task.first_command * low.command_scale, atol=1e-7)
    for point, item in ((low, state), (high, mature)):
        model = env._physics(item.task.floor_friction, item.task.physical_sample, point.physics_strength)
        if item.task.physical_sample is not None:
            np.testing.assert_array_equal(model.qpos0, item.task.physical_sample.qpos0)
            mass = np.asarray(model.body_mass)
            assert np.all(mass[np.asarray(env.mjx_model.body_mass) > 0] > 0), "Nonpositive physical link mass"
            numpy_gain = np.asarray(model.actuator_gainprm[env.indices.actuator_ids, 0])
            numpy_bias = np.asarray(model.actuator_biasprm[env.indices.actuator_ids, 1])
            np.testing.assert_allclose(numpy_gain, -numpy_bias, atol=1e-6)
        np.testing.assert_allclose(
            min(sole_heights(model, item.data, env.scene)), env.reset_config["sole_clearance_m"], atol=2e-6,
        )
    print("Reset: seeded sampling, fixed joint zero offsets and sole clearance verified", flush=True)

    if state.task.sensor_sample is not None:
        sensor = state.task.sensor_sample
        sensors = state.data.sensordata
        actor = np.asarray(state.obs["state"])
        critic = np.asarray(state.obs["privileged_state"])
        np.testing.assert_array_equal(actor, critic[:75])
        np.testing.assert_allclose(actor[:6], state.task.imu_history[sensor.imu_delay], atol=1e-7)
        np.testing.assert_allclose(critic[75:78], sensors[env.scene.gyro_indices], atol=1e-7)
        np.testing.assert_allclose(critic[78:81], sensors[env.scene.accelerometer_indices], atol=1e-7)
        np.testing.assert_allclose(actor[9:19] - critic[90:100], sensor.encoder_bias, atol=1e-7)

    step = jax.jit(env.step)
    zero = jnp.zeros(env.contract.action_size)
    for _ in range(20):
        state = step(state, zero)
        assert not bool(state.done) and np.isfinite(state.obs["state"]).all(), "Invalid course smoke state"
        _reward_from_log(state)
    assert int(state.task.curriculum_age) == 20
    np.testing.assert_allclose(state.metrics["curriculum/command_scale"], low.command_scale, atol=1e-7)

    entering = mature._replace(task=mature.task._replace(steps=jnp.int32(env.command_config["warmup_steps"] - 1)))
    active = step(entering, zero)
    if active.task.sensor_sample is not None:
        active = active._replace(task=active.task._replace(
            sensor_sample=active.task.sensor_sample._replace(action_delay=jnp.int32(2)),
        ))
    if active.task.push_interval is not None:
        active = active._replace(task=active.task._replace(push_interval=jnp.int32(1)))
    changed = step(active, jnp.full(env.contract.action_size, .35, dtype=jnp.float32))
    if active.task.sensor_sample is not None:
        np.testing.assert_array_equal(changed.task.applied_target, active.task.target_queue[1])
        np.testing.assert_array_equal(changed.obs["state"][59:69], changed.task.sent_target)
        np.testing.assert_array_equal(changed.data.ctrl[env.indices.actuator_ids], changed.task.applied_target)
    if active.task.push_interval is not None:
        lower, upper = env.push_config["delta_velocity_m_s"]
        assert lower <= float(changed.metrics["curriculum/push_m_s"]) <= upper, "Scheduled push missing"
    _reward_from_log(changed)

    last = state._replace(task=state.task._replace(steps=jnp.int32(env.episode_length - 1)))
    terminal, new_state = jax.jit(env.step_and_reset)(last, zero)
    assert bool(terminal.truncated) and int(new_state.task.steps) == 0
    assert int(new_state.task.curriculum_age) == int(last.task.curriculum_age) + 1
    _same_tree(new_state.task.physical_sample, last.task.physical_sample)
    np.testing.assert_array_equal(new_state.task.floor_friction, last.task.floor_friction)
    _reward_from_log(terminal)
    print("Transitions: clean critic, delayed control, scheduled push, reward reconstruction and auto-reset verified")
