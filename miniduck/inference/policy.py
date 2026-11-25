"""Load and run the frozen V4 policy in ONNX Runtime and NumPy."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from miniduck.contract import PolicyContract, load_contract, read_json

if TYPE_CHECKING:
    from onnxruntime import InferenceSession


Float32Array = NDArray[np.float32]
_HIDDEN_WIDTHS = (512, 256, 128)
# The source schema declares only slice bounds, so check its units here.
_V4_ACTOR_UNITS = {
    "gyro": "rad/s",
    "accelerometer": "m/s²",
    "command_xyz_yaw": "m/s, m/s, rad/s",
    "q_minus_default": "rad",
    "dq_scaled": "scaled rad/s",
    "last_action": "1",
    "last_last_action": "1",
    "last_last_last_action": "1",
    "last_motor_targets": "rad",
    "contacts_left_right": "0/1",
    "phase": "1",
    "relative_heading_sin_cos": "1",
}


@dataclass(frozen=True)
class PolicyBundle:
    """The same deterministic actor in two inference formats."""

    contract: PolicyContract
    onnx_sha256: str
    _session: "InferenceSession"
    _mean: Float32Array
    _std: Float32Array
    _weights: tuple[Float32Array, ...]
    _biases: tuple[Float32Array, ...]

    def _observation(self, observation: Float32Array) -> Float32Array:
        return _float_array(observation, (self.contract.actor_size,), "Raw actor observation")

    def _action(self, action: Float32Array) -> Float32Array:
        action = _float_array(action, (self.contract.action_size,), "Policy action")
        lower, upper = self.contract.action["clip"]
        if np.any((action < lower) | (action > upper)):
            raise ValueError("Policy action exceeds the contracted bounds")
        return action

    def infer_onnx(self, observation: Float32Array) -> Float32Array:
        """Feed raw actor channels to the graph, which applies its own normalization."""
        batch = self._observation(observation)[None, :]
        action, = self._session.run(["action"], {"observation": batch})
        return self._action(action[0])

    def infer_numpy(self, observation: Float32Array) -> Float32Array:
        """Evaluate the exported running statistics and deterministic actor."""
        value = (self._observation(observation) - self._mean) / self._std
        for weight, bias in zip(self._weights[:-1], self._biases[:-1], strict=True):
            affine = value @ weight + bias
            value = affine * np.exp(-np.logaddexp(np.float32(0.0), -affine))
        parameters = value @ self._weights[-1] + self._biases[-1]
        return self._action(np.tanh(parameters[:self.contract.action_size]))

    def parity_inputs(self, seed: int = 20261112, samples_per_scale: int = 256) -> Float32Array:
        """Build deterministic raw inputs around the actor's training mean."""
        rng = np.random.default_rng(seed)
        sample_shape = (samples_per_scale, self.contract.actor_size)
        return np.concatenate((
            self._mean[None, :],
            np.zeros_like(self._mean)[None, :],
            self._mean + self._std * rng.standard_normal(sample_shape, dtype=np.float32),
            self._mean + 4.0 * self._std * rng.standard_normal(sample_shape, dtype=np.float32),
        ))


def _float_array(value, shape: tuple[int, ...], label: str) -> Float32Array:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite float32 values with shape {shape}")
    return array


def _validate_metadata(metadata: dict, robot: dict, contract: PolicyContract) -> None:
    if {field.name: field.unit for field in contract.actor_fields} != _V4_ACTOR_UNITS:
        raise ValueError("Robot actor units do not match the V4 source schema")
    expected = {
        "schema": robot["policy"]["source_schema"],
        "dt": contract.clock.control_dt_s,
        "action_clip": contract.action["clip"][1],
        "action_scale": contract.action["scale_rad"],
        "max_motor_velocity": contract.action["target_rate_rad_s"],
        "joint_order": list(contract.joint_order),
        "default_joint_positions": robot["default_joint_positions_rad"],
        "joint_limits": robot["joint_limits_rad"],
        "initial_qpos_wxyz": robot["initial_qpos_wxyz"],
        "dof_vel_scale": contract.observations["joint_velocity_scale"],
        "phase_order": contract.observations["phase_order"],
        "phase_period_steps": contract.clock.phase_period_steps,
        "heading_reference": contract.observations["heading_reference"],
        "observation_layout": {field.name: [field.start, field.stop] for field in contract.actor_fields},
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"Policy metadata field {field!r} does not match the robot contract")


def _load_numpy_weights(path: Path, onnx_digest: str, contract: PolicyContract):
    widths = (contract.actor_size, *_HIDDEN_WIDTHS, 2 * contract.action_size)
    layer_count = len(widths) - 1
    keys = {"onnx_sha256", "max_abs_error", "mean", "std", "layers"}
    keys.update(f"{kind}{index}" for index in range(layer_count) for kind in ("w", "b"))

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != keys:
            raise ValueError("Unexpected keys in the V4 NumPy bundle")
        digest, layers, error = (archive[name] for name in ("onnx_sha256", "layers", "max_abs_error"))
        if digest.shape != () or digest.dtype.kind != "U" or str(digest) != onnx_digest:
            raise ValueError("NumPy bundle belongs to a different ONNX policy")
        if layers.shape != () or layers.dtype.kind not in "iu" or int(layers) != layer_count:
            raise ValueError("Unexpected number of V4 actor layers")
        if error.shape != () or error.dtype.kind != "f" or not np.isfinite(error) or float(error) < 0:
            raise ValueError("Invalid NumPy export error metadata")

        mean = _float_array(archive["mean"], (contract.actor_size,), "Normalizer mean")
        std = _float_array(archive["std"], (contract.actor_size,), "Normalizer standard deviation")
        if np.any(std <= 0):
            raise ValueError("Normalizer standard deviations must be positive")
        weights = tuple(
            _float_array(archive[f"w{i}"], (input_width, width), f"Layer {i} weight")
            for i, (input_width, width) in enumerate(zip(widths[:-1], widths[1:], strict=True))
        )
        biases = tuple(
            _float_array(archive[f"b{i}"], (width,), f"Layer {i} bias")
            for i, width in enumerate(widths[1:])
        )
    return mean, std, weights, biases


def _check_onnx_io(session: "InferenceSession", contract: PolicyContract) -> None:
    for ports, expected_name, width in (
        (session.get_inputs(), "observation", contract.actor_size),
        (session.get_outputs(), "action", contract.action_size),
    ):
        if len(ports) != 1:
            raise ValueError(f"Expected one ONNX {expected_name} port")
        port = ports[0]
        if port.name != expected_name or port.type != "tensor(float)" or port.shape != [1, width]:
            raise ValueError(f"ONNX {expected_name} must be a float32 tensor of shape [1, {width}]")


def load_policy_bundle(stem: Path, robot: dict) -> PolicyBundle:
    """Load and verify the matching .json, .onnx, and .npz files for one actor."""
    stem = Path(stem)
    contract = load_contract(robot)
    metadata = read_json(stem.with_name(stem.name + ".json"))
    _validate_metadata(metadata, robot, contract)
    onnx_model = stem.with_name(stem.name + ".onnx").read_bytes()
    digest = sha256(onnx_model).hexdigest()
    if metadata.get("onnx_sha256") != digest:
        raise ValueError("ONNX policy SHA-256 does not match its metadata")
    mean, std, weights, biases = _load_numpy_weights(
        stem.with_name(stem.name + ".npz"), digest, contract,
    )

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_model, sess_options=options, providers=["CPUExecutionProvider"])
    _check_onnx_io(session, contract)
    return PolicyBundle(contract, digest, session, mean, std, weights, biases)
