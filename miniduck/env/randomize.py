"""Sample persistent physical parameters for single-environment training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dataclasses import dataclass
import hashlib
import json
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import jax
    from mujoco import mjx


_SCHEMA = "miniduck_training_plant_rehearsal_v1"
_FIELD_SHAPES = {
    "geom_friction": (35, 3),
    "body_mass": (14,),
    "body_inertia": (14, 3),
    "body_ipos": (14, 3),
    "dof_frictionloss": (16,),
    "dof_damping": (16,),
    "dof_armature": (16,),
    "qpos0": (17,),
    "actuator_gainprm": (10, 10),
    "actuator_biasprm": (10, 10),
    "actuator_forcerange": (10, 2),
}


@dataclass(frozen=True)
class PerturbationBank:
    fields: dict[str, NDArray[np.float32]]
    metadata: dict[str, Any]

    @property
    def size(self) -> int:
        return len(self.metadata["source_indices"])

    def sample(self, rng: np.random.Generator, num_envs: int) -> dict[str, NDArray[np.float32]]:
        """Draw one intact plant per environment, keeping its fields together.

        Keep the selected row for the lifetime of its environment. The classroom
        training mixture also includes other randomization and derived plants.
        """
        indices = rng.integers(self.size, size=index(num_envs))
        return {name: values[indices] for name, values in self.fields.items()}


def load_perturbations(path: Path) -> PerturbationBank:
    """Validate model-specific physical fields and their per-plant identities."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(_FIELD_SHAPES) | {"metadata_json"}:
            raise ValueError(f"Invalid perturbation archive fields in {path}")

        text = archive["metadata_json"]
        if text.shape != () or text.dtype.kind not in ("U", "S"):
            raise ValueError("Perturbation metadata must be a scalar JSON string")
        metadata = json.loads(text.item())
        if not isinstance(metadata, dict) or metadata.get("schema") != _SCHEMA:
            raise ValueError(f"Expected perturbation metadata schema {_SCHEMA}")

        indices = metadata.get("source_indices")
        sample_count = metadata.get("source_sample_count")
        if (
            not isinstance(indices, list)
            or len(indices) != 6
            or type(sample_count) is not int
            or any(type(i) is not int or not 0 <= i < sample_count for i in indices)
            or len(set(indices)) != len(indices)
            or metadata.get("fields") != list(_FIELD_SHAPES)
        ):
            raise ValueError("Invalid development plant indices or metadata field order")

        fields = {}
        for name, shape in _FIELD_SHAPES.items():
            array = archive[name]
            if (
                array.shape != (len(indices), *shape)
                or array.dtype != np.float32
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"Invalid perturbation array {name}")
            fields[name] = array

    row_digests = []
    for row in range(len(indices)):
        digest = hashlib.sha256()
        for name, values in fields.items():
            digest.update(name.encode("ascii"))
            digest.update(values[row].tobytes())
        row_digests.append(digest.hexdigest())
    if metadata.get("plant_field_sha256") != row_digests:
        raise ValueError("Perturbation plant fields differ from their archived SHA-256 values")

    return PerturbationBank(fields, metadata)


_PHYSICAL_RANGES = {
    "body_mass_scale": True,
    "body_com_offset_m": False,
    "joint_friction_scale": True,
    "joint_damping_scale": True,
    "joint_armature_scale": True,
    "actuator_gain_scale": True,
    "joint_zero_offset_rad": False,
}


class PhysicalSample(NamedTuple):
    """Absolute MJX parameters sampled once for an environment."""

    geom_friction: jax.Array
    body_mass: jax.Array
    body_inertia: jax.Array
    body_ipos: jax.Array
    dof_frictionloss: jax.Array
    dof_damping: jax.Array
    dof_armature: jax.Array
    qpos0: jax.Array
    actuator_gainprm: jax.Array
    actuator_biasprm: jax.Array
    actuator_forcerange: jax.Array


def validate_physical_ranges(ranges: Mapping[str, Sequence[float]]) -> dict[str, tuple[float, float]]:
    """Check configured sampling bounds once, before tracing a rollout."""
    if set(ranges) != set(_PHYSICAL_RANGES):
        missing = sorted(set(_PHYSICAL_RANGES) - set(ranges))
        extra = sorted(set(ranges) - set(_PHYSICAL_RANGES))
        raise ValueError(f"Unexpected physical range keys. Missing {missing}, extra {extra}")

    result = {}
    for name, positive in _PHYSICAL_RANGES.items():
        bounds = ranges[name]
        if (
            not isinstance(bounds, (list, tuple)) or len(bounds) != 2
            or any(type(value) not in (int, float) or not np.isfinite(value) for value in bounds)
        ):
            raise ValueError(f"{name} requires two finite numbers")
        low, high = bounds
        if low > high or (positive and low <= 0):
            raise ValueError(f"Invalid sampling bounds for {name}: {bounds}")
        result[name] = (float(low), float(high))
    return result


def sample_physical_parameters(
    model: mjx.Model,
    key: jax.Array,
    ranges: Mapping[str, tuple[float, float]],
    *,
    joint_dof_indices: Sequence[int],
    joint_qpos_indices: Sequence[int],
    actuator_ids: Sequence[int],
    bank_arrays: Mapping[str, NDArray[np.float32]] | None = None,
) -> PhysicalSample:
    """Draw one model from validated bounds or one intact archived plant."""
    import jax
    import jax.numpy as jnp

    if bank_arrays is not None:
        row = jax.random.randint(key, (), 0, bank_arrays["qpos0"].shape[0])
        return PhysicalSample(*(jnp.asarray(bank_arrays[name])[row] for name in PhysicalSample._fields))

    keys = dict(zip(_PHYSICAL_RANGES, jax.random.split(key, len(_PHYSICAL_RANGES)), strict=True))

    def draw(name, values):
        low, high = ranges[name]
        return jax.random.uniform(keys[name], values.shape, dtype=values.dtype, minval=low, maxval=high)

    mass_scale = draw("body_mass_scale", model.body_mass)
    inertia = model.body_inertia * mass_scale[:, None]
    center = jnp.where(
        model.body_mass[:, None] > 0,
        model.body_ipos + draw("body_com_offset_m", model.body_ipos),
        model.body_ipos,
    )

    dofs = jnp.asarray(joint_dof_indices, dtype=jnp.int32)
    positions = jnp.asarray(joint_qpos_indices, dtype=jnp.int32)
    actuators = jnp.asarray(actuator_ids, dtype=jnp.int32)

    def scale_joints(field, name):
        nominal = field[dofs]
        return field.at[dofs].set(nominal * draw(name, nominal))

    joint_friction = scale_joints(model.dof_frictionloss, "joint_friction_scale")
    joint_damping = scale_joints(model.dof_damping, "joint_damping_scale")
    joint_armature = scale_joints(model.dof_armature, "joint_armature_scale")
    qpos0 = model.qpos0.at[positions].add(draw("joint_zero_offset_rad", model.qpos0[positions]))

    nominal_gain = model.actuator_gainprm[actuators, 0]
    scaled_gain = nominal_gain * draw("actuator_gain_scale", nominal_gain)
    gain = model.actuator_gainprm.at[actuators, 0].set(scaled_gain)
    # The named position servos use force = kp * (target - position) - kd * velocity.
    bias = model.actuator_biasprm.at[actuators, 1].set(-scaled_gain)
    return PhysicalSample(
        geom_friction=model.geom_friction,
        body_mass=model.body_mass * mass_scale,
        body_inertia=inertia,
        body_ipos=center,
        dof_frictionloss=joint_friction,
        dof_damping=joint_damping,
        dof_armature=joint_armature,
        qpos0=qpos0,
        actuator_gainprm=gain,
        actuator_biasprm=bias,
        actuator_forcerange=model.actuator_forcerange,
    )


def apply_physical_parameters(model: mjx.Model, sample: PhysicalSample, strength: jax.Array) -> mjx.Model:
    """Interpolate continuous parameters while retaining the sampled joint zeros."""
    import jax.numpy as jnp

    fraction = jnp.clip(strength, 0.0, 1.0)
    updated = {"qpos0": sample.qpos0}
    for name in PhysicalSample._fields:
        if name == "qpos0":
            continue
        nominal, target = getattr(model, name), getattr(sample, name)
        interpolated = nominal + fraction * (target - nominal)
        updated[name] = jnp.where(fraction == 1.0, target, interpolated)

    mass = updated["body_mass"]
    for child in range(model.nbody - 1, 0, -1):
        mass = mass.at[model.body_parentid[child]].add(mass[child])
    updated["body_subtreemass"] = mass
    return model.replace(**updated)
