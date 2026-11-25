"""Export a verified deterministic V4 actor from a project PPO checkpoint."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np

from miniduck.contract import load_contract, read_json
from miniduck.inference.policy import PolicyBundle, load_policy_bundle


_ABSOLUTE_TOLERANCE = 2e-5


def _tensor(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite float32 tensor of shape {shape}")
    return array


def _actor_parameters(source, contract):
    from brax.training.agents.ppo import checkpoint

    config = read_json(source / "ppo_network_config.json")
    network = config["network_factory_kwargs"]
    widths = (contract.actor_size, 512, 256, 128, 2 * contract.action_size)
    if (
        config["observation_size"] != {
            contract.actor_key: [contract.actor_size], contract.critic_key: [contract.critic_size],
        }
        or config["action_size"] != contract.action_size
        or config["normalize_observations"] is not True
        or network["policy_obs_key"] != contract.actor_key
        or network["policy_hidden_layer_sizes"] != list(widths[1:-1])
        or network["activation"] != "silu"
        or network["distribution_type"] != "tanh_normal"
        or network.get("mean_clip_scale") is not None
    ):
        raise ValueError("Checkpoint does not describe the V4 normalized SiLU and tanh actor")

    statistics, actor, _ = checkpoint.load(source)
    shape = (contract.actor_size,)
    mean = _tensor(statistics.mean[contract.actor_key], shape, "Actor mean")
    std = _tensor(statistics.std[contract.actor_key], shape, "Actor standard deviation")
    if np.any(std <= 0):
        raise ValueError("Actor standard deviations must be positive")

    layers = actor["params"]
    names = [f"hidden_{i}" for i in range(len(widths) - 1)]
    if set(layers) != set(names):
        raise ValueError("Checkpoint actor layers do not match the V4 architecture")
    weights, biases = [], []
    shapes = zip(widths[:-1], widths[1:], strict=True)
    for name, (input_size, output_size) in zip(names, shapes, strict=True):
        layer = layers[name]
        weights.append(_tensor(layer["kernel"], (input_size, output_size), f"{name} kernel"))
        biases.append(_tensor(layer["bias"], (output_size,), f"{name} bias"))
    return mean, std, tuple(weights), tuple(biases)


def _onnx_bytes(contract, mean, std, weights, biases):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    initializers = {
        "actor_mean": mean,
        "actor_std": std,
        "action_indices": np.arange(contract.action_size, dtype=np.int64),
    }
    nodes = [
        helper.make_node("Sub", ["observation", "actor_mean"], ["centered"]),
        helper.make_node("Div", ["centered", "actor_std"], ["normalized"]),
    ]
    previous = "normalized"
    for i, (kernel, bias) in enumerate(zip(weights, biases, strict=True)):
        weight_name, bias_name = f"kernel_{i}", f"bias_{i}"
        initializers.update({weight_name: kernel, bias_name: bias})
        product, affine = f"product_{i}", f"affine_{i}"
        nodes.append(helper.make_node("MatMul", [previous, weight_name], [product]))
        nodes.append(helper.make_node("Add", [product, bias_name], [affine]))
        previous = affine
        if i < len(weights) - 1:
            sigmoid, activated = f"sigmoid_{i}", f"activated_{i}"
            nodes.append(helper.make_node("Sigmoid", [affine], [sigmoid]))
            nodes.append(helper.make_node("Mul", [affine, sigmoid], [activated]))
            previous = activated

    nodes.extend((
        helper.make_node("Gather", [previous, "action_indices"], ["action_mean"], axis=1),
        helper.make_node("Tanh", ["action_mean"], ["action"]),
    ))
    graph = helper.make_graph(
        nodes, "duck_v4_actor",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, contract.actor_size])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, contract.action_size])],
        [numpy_helper.from_array(value, name) for name, value in initializers.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              ir_version=9, producer_name="miniduck")
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def _parity(source, model, digest, contract, mean, std, weights, biases):
    from brax.training.agents.ppo import checkpoint
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
    bundle = PolicyBundle(contract, digest, session, mean, std, weights, biases)

    probes = bundle.parity_inputs(seed=20261005)
    infer = jax.jit(checkpoint.load_policy(source, deterministic=True))
    expected, _ = infer({contract.actor_key: jnp.asarray(probes)}, jax.random.key(0))
    expected = _tensor(expected, (len(probes), contract.action_size), "Brax verification actions")
    actual = np.stack([bundle.infer_onnx(row) for row in probes])
    numpy_actions = np.stack([bundle.infer_numpy(row) for row in probes])
    onnx_error = float(np.max(np.abs(expected - actual)))
    numpy_error = float(np.max(np.abs(numpy_actions - actual)))
    if max(onnx_error, numpy_error) > _ABSOLUTE_TOLERANCE:
        raise ValueError(f"Policy export parity failed: Brax–ONNX {onnx_error:.8g}, NumPy–ONNX {numpy_error:.8g}")
    return len(probes), onnx_error, numpy_error


def _metadata(source, step, robot, contract, run_config, digest, samples, error, numpy_error):
    clock = contract.clock
    command = run_config["task"]["command"]
    magnitudes = command["magnitude_range"]
    coordinates = ("x", "y", "yaw")
    return {
        "schema": robot["policy"]["source_schema"],
        "dt": clock.control_dt_s,
        "action_clip": contract.action["clip"][1],
        "action_scale": contract.action["scale_rad"],
        "max_motor_velocity": contract.action["target_rate_rad_s"],
        "joint_order": list(contract.joint_order),
        "default_joint_positions": robot["default_joint_positions_rad"],
        "joint_limits": robot["joint_limits_rad"],
        "initial_qpos_wxyz": robot["initial_qpos_wxyz"],
        "dof_vel_scale": contract.observations["joint_velocity_scale"],
        "phase_order": contract.observations["phase_order"],
        "phase_period_steps": clock.phase_period_steps,
        "heading_reference": contract.observations["heading_reference"],
        "observation_layout": {field.name: [field.start, field.stop] for field in contract.actor_fields},
        "command_ranges": {name: [-high, high] for name, (_, high) in zip(coordinates, magnitudes, strict=True)},
        "checkpoint": str(source),
        "checkpoint_step": step,
        "training_variant": run_config["baseline"],
        "onnx_sha256": digest,
        "export_parity": {"samples": samples, "max_abs_error": error, "atol": _ABSOLUTE_TOLERANCE},
        "numpy_parity": {"samples": samples, "max_abs_error": numpy_error, "atol": _ABSOLUTE_TOLERANCE},
        "hardware_accepted": False,
    }


def _publish_files(destination, model, metadata, robot, mean, std, weights, biases):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".export-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        staged.write_bytes(model)
        staged.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        arrays = {
            "onnx_sha256": np.array(metadata["onnx_sha256"]),
            "max_abs_error": np.float64(metadata["numpy_parity"]["max_abs_error"]),
            "mean": mean,
            "std": std,
            "layers": np.int64(len(weights)),
        }
        arrays.update({f"w{i}": value for i, value in enumerate(weights)})
        arrays.update({f"b{i}": value for i, value in enumerate(biases)})
        np.savez(staged.with_suffix(".npz"), **arrays)
        load_policy_bundle(staged.with_suffix(""), robot)

        # Publish the ONNX file last and create each path without replacing any file.
        published = []
        try:
            for suffix in (".json", ".npz", ".onnx"):
                final = destination.with_suffix(suffix)
                final.hardlink_to(staged.with_suffix(suffix))
                published.append(final)
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            raise


def export_actor(checkpoint: Path, destination: Path) -> dict:
    """Export ONNX plus matching contract and NumPy bundle after CPU parity checks."""
    source = Path(checkpoint).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.suffix != ".onnx":
        raise ValueError("Export destination must have an .onnx suffix")
    for suffix in (".onnx", ".json", ".npz"):
        target = destination.with_suffix(suffix)
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)

    run = source.parent.parent
    robot = read_json(run / "policy_contract.json")
    contract = load_contract(robot)
    run_config = read_json(run / "run_config.json")
    step = read_json(source / "training_step.json")["environment_steps"]
    if type(step) is not int or step < 0 or source.name != f"{step:012d}":
        raise ValueError("Checkpoint directory and recorded environment step disagree")

    mean, std, weights, biases = _actor_parameters(source, contract)
    model = _onnx_bytes(contract, mean, std, weights, biases)
    digest = sha256(model).hexdigest()
    samples, error, numpy_error = _parity(source, model, digest, contract, mean, std, weights, biases)
    metadata = _metadata(source, step, robot, contract, run_config, digest, samples, error, numpy_error)
    _publish_files(destination, model, metadata, robot, mean, std, weights, biases)
    return {
        "onnx": str(destination),
        "contract": str(destination.with_suffix(".json")),
        "npz": str(destination.with_suffix(".npz")),
        "checkpoint_step": step,
        "onnx_sha256": digest,
        "parity_samples": samples,
        "parity_max_abs_error": error,
        "numpy_max_abs_error": numpy_error,
    }
