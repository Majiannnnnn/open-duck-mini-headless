"""Per-environment curriculum schedules and random velocity impulses."""

from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp


class CurriculumPoint(NamedTuple):
    command_scale: jax.Array
    roll_scale: jax.Array
    guidance_weight: jax.Array
    physics_strength: jax.Array
    push_strength: jax.Array


def validate_schedule(config: dict) -> None:
    if config["schema_version"] != 1:
        raise ValueError("Unsupported curriculum configuration schema")
    for name, initial in (("command", "initial_scale"), ("roll", "initial_scale"), ("physics", "initial_strength")):
        section = config[name]
        if not 0 <= section[initial] <= 1 or type(section["ramp_steps"]) is not int or section["ramp_steps"] <= 0:
            raise ValueError(f"Invalid initial scale or ramp duration for {name}")
    begin, end = config["guidance"]["fade_steps"]
    weight = config["guidance"]["weight"]
    if type(begin) is not int or type(end) is not int or not 0 <= begin < end or not isfinite(weight) or weight < 0:
        raise ValueError("Invalid guidance fade interval or weight")
    push = config["push"]
    if type(push["enabled"]) is not bool:
        raise ValueError("Push enabled must be boolean")
    for field, allow_zero in (("interval_s", False), ("delta_velocity_m_s", True)):
        low, high = push[field]
        if not isfinite(low) or not isfinite(high) or low > high or low < 0 or (low == 0 and not allow_zero):
            raise ValueError(f"Invalid push {field} range")


def curriculum_at(age, config: dict) -> CurriculumPoint:
    """Evaluate deterministic schedules using elapsed control ticks."""
    age = jnp.asarray(age, dtype=jnp.float32)
    command = config["command"]
    roll = config["roll"]
    physics = config["physics"]
    begin, end = config["guidance"]["fade_steps"]

    def ramp(duration):
        return jnp.clip(age / duration, 0.0, 1.0)

    command_scale = command["initial_scale"] + (1.0 - command["initial_scale"]) * ramp(command["ramp_steps"])
    roll_scale = roll["initial_scale"] + (1.0 - roll["initial_scale"]) * ramp(roll["ramp_steps"])
    guidance_weight = config["guidance"]["weight"] * (1.0 - jnp.clip((age - begin) / (end - begin), 0.0, 1.0))
    push_strength = ramp(physics["ramp_steps"])
    strength = physics["initial_strength"] + (1.0 - physics["initial_strength"]) * push_strength
    return CurriculumPoint(command_scale, roll_scale, guidance_weight, strength, push_strength)


def sample_push_interval(key, interval_s, dt):
    seconds = jax.random.uniform(key, (), minval=interval_s[0], maxval=interval_s[1])
    return jnp.maximum(jnp.rint(seconds / dt).astype(jnp.int32), 1)


def push_velocity(key, step, interval, strength, speed_range):
    """Sample a horizontal delta-v at scheduled ticks."""
    next_key, direction_key, speed_key = jax.random.split(key, 3)
    angle = jax.random.uniform(direction_key, (), maxval=2.0 * jnp.pi)
    speed = jax.random.uniform(speed_key, (), minval=speed_range[0], maxval=speed_range[1])
    direction = jnp.stack((jnp.cos(angle), jnp.sin(angle)))
    impulse = jnp.where((step + 1) % interval == 0, speed * strength, 0.0)
    return next_key, impulse * direction
