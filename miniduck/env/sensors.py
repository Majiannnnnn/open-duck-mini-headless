"""Repeatable per-episode sensor biases and control-tick delays."""

from math import isfinite
from numbers import Real
from typing import NamedTuple

import jax
import jax.numpy as jnp


class SensorPerturbation(NamedTuple):
    encoder_bias: jax.Array
    gyro_bias: jax.Array
    accelerometer_bias: jax.Array
    imu_rotation: jax.Array
    imu_delay: jax.Array
    action_delay: jax.Array


def _amplitude(config: dict, name: str) -> float:
    value = config.get(name, 0.0)
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative amplitude")
    return float(value)


def _validate_delays(config: dict, name: str) -> None:
    allowed = config.get(name, (0,))
    if (
        not isinstance(allowed, (list, tuple))
        or not allowed
        or any(type(step) is not int or not 0 <= step <= 2 for step in allowed)
        or len(set(allowed)) != len(allowed)
    ):
        raise ValueError(f"{name} must contain distinct allowed delays from 0, 1, 2")



def validate_sensor_config(config: dict) -> None:
    """Check static options before building a compiled episode."""
    if not isinstance(config, dict):
        raise ValueError("Sensor perturbation options must be a dict")
    if type(config.get("enabled", False)) is not bool:
        raise ValueError("Sensor perturbation enabled flag must be boolean")
    for name in ("encoder_bias_rad", "gyro_bias_rad_s", "accelerometer_bias_m_s2", "imu_mount_radians"):
        _amplitude(config, name)
    for name in ("imu_delay_steps", "action_delay_steps"):
        _validate_delays(config, name)


def _sample_delay(key, config: dict, name: str) -> jax.Array:
    options = jnp.asarray(config.get(name, (0,)), dtype=jnp.int32)
    return jax.random.choice(key, options)


def _sample_bias(key, config: dict, name: str, width: int) -> jax.Array:
    amplitude = float(config.get(name, 0.0))
    return amplitude * jax.random.uniform(key, (width,), dtype=jnp.float32, minval=-1.0, maxval=1.0)


def _mount_rotation(angles: jax.Array) -> jax.Array:
    """Rotate raw body-frame IMU axes into the simulated sensor axes."""
    cx, cy, cz = jnp.cos(angles)
    sx, sy, sz = jnp.sin(angles)
    rotate_x = jnp.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotate_y = jnp.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotate_z = jnp.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rotate_z @ rotate_y @ rotate_x


def sample_sensors(key, config: dict) -> SensorPerturbation:
    """Draw one episode's independent biases, mounting angles and delays.

    The configuration is static under JIT. Sensor amplitudes specify the
    symmetric uniform range in the units of the corresponding observed signal.
    """
    validate_sensor_config(config)
    if not config.get("enabled", False):
        zero_xyz = jnp.zeros(3, dtype=jnp.float32)
        zero_delay = jnp.int32(0)
        return SensorPerturbation(
            jnp.zeros(10, dtype=jnp.float32),
            zero_xyz,
            zero_xyz,
            jnp.eye(3, dtype=jnp.float32),
            zero_delay,
            zero_delay,
        )

    encoder_key, gyro_key, accel_key, mount_key, imu_key, action_key = jax.random.split(key, 6)
    return SensorPerturbation(
        _sample_bias(encoder_key, config, "encoder_bias_rad", 10),
        _sample_bias(gyro_key, config, "gyro_bias_rad_s", 3),
        _sample_bias(accel_key, config, "accelerometer_bias_m_s2", 3),
        _mount_rotation(_sample_bias(mount_key, config, "imu_mount_radians", 3)),
        _sample_delay(imu_key, config, "imu_delay_steps"),
        _sample_delay(action_key, config, "action_delay_steps"),
    )


def _observed_imu(gyro, accel, sensors: SensorPerturbation) -> jax.Array:
    return jnp.concatenate((
        sensors.imu_rotation @ gyro + sensors.gyro_bias,
        sensors.imu_rotation @ accel + sensors.accelerometer_bias,
    ))


def initial_imu_history(gyro, accel, sensors: SensorPerturbation, queue_length: int) -> jax.Array:
    """Prefill every delay slot with the first valid sensor measurement."""
    if queue_length < 1:
        raise ValueError("IMU history must have at least one slot")
    return jnp.broadcast_to(_observed_imu(gyro, accel, sensors), (queue_length, 6))


def read_imu(gyro, accel, sensors: SensorPerturbation, history: jax.Array):
    """Push one sensor sample and read its configured delay.

    The history is newest first and must cover all allowed IMU delays.
    """
    latest = _observed_imu(gyro, accel, sensors)
    updated = jnp.concatenate((latest[None, :], history[:-1]), axis=0)
    measurement = updated[sensors.imu_delay]
    return measurement[:3], measurement[3:], updated
