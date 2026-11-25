"""Write and verify atomic, portable Brax PPO parameter snapshots."""

import json
from operator import index
from pathlib import Path
from tempfile import TemporaryDirectory

from brax.training.agents.ppo import checkpoint as brax_checkpoint
import jax
from ml_collections import ConfigDict
import numpy as np


_PARAMETER_GROUPS = ("normalizer", "actor", "critic")
_INITIALIZER_FIELDS = (
    "policy_network_kernel_init_fn",
    "value_network_kernel_init_fn",
    "q_network_kernel_init_fn",
    "mean_kernel_init_fn",
)


def _parameters(params):
    if not isinstance(params, (tuple, list)) or len(params) != len(_PARAMETER_GROUPS):
        raise ValueError("A PPO checkpoint needs normalizer, actor and critic parameters")
    groups = {}
    for name, group in zip(_PARAMETER_GROUPS, params, strict=True):
        leaves, _ = jax.tree_util.tree_flatten_with_path(group)
        if not leaves:
            raise ValueError(f"Missing {name} parameters")
        arrays = {jax.tree_util.keystr(key): np.asarray(value) for key, value in leaves}
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f"Nonfinite {name} parameters")
        groups[name] = arrays
    return groups


def _portable_config(network_config):
    # Brax 0.14.2 reads present initializer keys as registry names.
    # Omitting a null initializer lets the constructor supply its default.
    config = network_config.to_dict()
    kwargs = config["network_factory_kwargs"]
    for key in _INITIALIZER_FIELDS:
        if key in kwargs and kwargs[key] is None:
            del kwargs[key]
    return ConfigDict(config)


def save_checkpoint(directory: Path, step: int, params, network_config) -> Path:
    """Save a complete parameter snapshot, then publish its numbered directory."""
    if isinstance(step, bool):
        raise ValueError("Checkpoint step must be a nonnegative integer")
    step = index(step)
    if step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    _parameters(params)
    config = _portable_config(network_config)

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{step:012d}"
    destination = directory / name
    if destination.exists():
        raise FileExistsError(destination)

    with TemporaryDirectory(prefix=".checkpoint-", dir=directory) as temporary:
        brax_checkpoint.save(Path(temporary), step, params, config)
        saved = Path(temporary) / name
        metadata = {"environment_steps": step}
        (saved / "training_step.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )
        saved.rename(destination)
    return destination


def verify_checkpoint(path: Path, *, params, make_policy, observation) -> float:
    """Check exact parameter recovery and run a deterministic policy round trip."""
    path = Path(path).resolve()
    expected = _parameters(params)
    restored = brax_checkpoint.load(path)
    actual = _parameters(restored)
    for name, values in expected.items():
        if set(values) != set(actual[name]):
            raise ValueError(f"Restored {name} parameter structure differs from the snapshot")
        for key, value in values.items():
            received = actual[name][key]
            if value.dtype != received.dtype or not np.array_equal(value, received):
                raise ValueError(f"Restored {name} parameter {key} differs from the snapshot")

    rng = jax.random.key(0)
    original = make_policy(params, deterministic=True)
    reloaded = brax_checkpoint.load_policy(path, deterministic=True)
    action = np.asarray(original(observation, rng)[0])
    restored_action = np.asarray(reloaded(observation, rng)[0])
    for value in (action, restored_action):
        if value.shape != (10,) or not np.isfinite(value).all():
            raise ValueError("Checkpoint verification expected one finite 10-channel action")
    difference = float(np.max(np.abs(action - restored_action)))
    if difference > 1e-6:
        raise ValueError(f"Checkpoint policy action changed by {difference:.8g}")
    return difference
