"""NumPy actor input and transactional target sender for the V4 runtime."""

from dataclasses import dataclass
from operator import index
from threading import Lock

import numpy as np
from numpy.typing import NDArray

from miniduck.contract import PolicyContract, load_contract


FloatArray = NDArray[np.float32]


def _finite_array(value, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values with shape {shape}")
    return array


def _snapshot(array: FloatArray) -> FloatArray:
    # Immutable storage keeps a prepared tick stable while a send is pending.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def pack_actor_numpy(
    contract: PolicyContract,
    *,
    gyro,
    accelerometer,
    command,
    joint_positions,
    joint_velocities,
    home,
    action_history,
    last_sent,
    contacts,
    phase: int,
    yaw,
    reference_heading,
) -> FloatArray:
    """Assemble the raw actor input against the shared field definitions."""
    size = contract.action_size
    gyro = _finite_array(gyro, (3,), "gyro")
    accelerometer = _finite_array(accelerometer, (3,), "accelerometer")
    command = _finite_array(command, (3,), "command")
    joint_positions = _finite_array(joint_positions, (size,), "joint_positions")
    joint_velocities = _finite_array(joint_velocities, (size,), "joint_velocities")
    home = _finite_array(home, (size,), "home")
    history = _finite_array(action_history, (3, size), "action_history")
    last_sent = _finite_array(last_sent, (size,), "last_sent")
    contacts = np.asarray(contacts)
    if contacts.shape != (2,) or not ((contacts == 0) | (contacts == 1)).all():
        raise ValueError("contacts must give left/right flags, each 0 or 1")
    yaw = float(_finite_array(yaw, (), "yaw"))
    reference_heading = float(_finite_array(reference_heading, (), "reference_heading"))

    period = contract.clock.phase_period_steps
    phase = index(phase)
    if not 0 <= phase < period:
        raise ValueError(f"phase must be between 0 and {period - 1}")
    angle = phase * (2.0 * np.pi / period)
    threshold = contract.active_command_threshold
    translating = np.linalg.norm(command[:2]) > threshold and abs(command[2]) < threshold
    heading_error = yaw - reference_heading if translating else 0.0
    fields = {
        "gyro": gyro,
        "accelerometer": accelerometer,
        "command_xyz_yaw": command,
        "q_minus_default": joint_positions - home,
        "dq_scaled": joint_velocities * contract.observations["joint_velocity_scale"],
        "last_action": history[0],
        "last_last_action": history[1],
        "last_last_last_action": history[2],
        "last_motor_targets": last_sent,
        "contacts_left_right": contacts,
        "phase": np.array([np.cos(angle), np.sin(angle)]),
        "relative_heading_sin_cos": np.array([np.sin(heading_error), np.cos(heading_error)]),
    }
    observation = np.empty(contract.actor_size, dtype=np.float32)
    for layout in contract.actor_fields:
        value = fields[layout.name]
        if value.shape != (layout.stop - layout.start,):
            raise ValueError(f"Actor field {layout.name} does not match its configured width")
        observation[layout.indices] = value
    if not np.isfinite(observation).all():
        raise ValueError("Assembled actor observation contains non-finite values")
    return observation


@dataclass(frozen=True, eq=False)
class _RuntimeState:
    mode: int
    phase: int
    reference_heading: float
    action_history: FloatArray
    last_sent: FloatArray


@dataclass(frozen=True, eq=False)
class PreparedTick:
    """An observation and its candidate mode and heading for one tick."""

    observation: FloatArray
    phase: int
    mode: int
    reference_heading: float
    _history: FloatArray
    _base: _RuntimeState


@dataclass(frozen=True, eq=False)
class TargetTick:
    """The bounded action and constrained target awaiting a send."""

    sent: FloatArray
    action: FloatArray
    _prepared: PreparedTick


class Runtime:
    """Assemble observations and commit a target after the caller sends it."""

    def __init__(self, robot: dict):
        self.contract = load_contract(robot)
        size = self.contract.action_size
        self.home = _snapshot(_finite_array(robot["default_joint_positions_rad"], (size,), "home"))
        self.joint_limits = _snapshot(_finite_array(robot["joint_limits_rad"], (size, 2), "joint limits"))
        lower, upper = self.joint_limits.T
        if np.any(lower >= upper) or np.any(self.home < lower) or np.any(self.home > upper):
            raise ValueError("Home joint angles must lie within the joint limits")

        self._zero_history = _snapshot(np.zeros((3, size), dtype=np.float32))
        self._update = Lock()
        self.reset()

    @property
    def mode(self) -> int:
        return self._state.mode

    @property
    def phase(self) -> int:
        return self._state.phase

    @property
    def reference_heading(self) -> float:
        return self._state.reference_heading

    @property
    def action_history(self) -> FloatArray:
        return self._state.action_history

    @property
    def last_sent(self) -> FloatArray:
        return self._state.last_sent

    def reset(self, last_sent=None) -> None:
        """Clear control history, optionally keeping a known submitted target."""
        if last_sent is None:
            target = self.home
        else:
            target = _finite_array(last_sent, (self.contract.action_size,), "last_sent")
            lower, upper = self.joint_limits.T
            if np.any(target < lower) or np.any(target > upper):
                raise ValueError("last_sent must lie within the joint limits")
            target = _snapshot(target)
        with self._update:
            self._state = _RuntimeState(0, 0, 0.0, self._zero_history, target)

    def prepare(self, command, gyro, accelerometer, joint_positions, joint_velocities, contacts, yaw) -> PreparedTick:
        """Build the next observation without changing committed control history."""
        base = self._state
        command = _finite_array(command, (3,), "command")
        yaw = float(_finite_array(yaw, (), "yaw"))
        axis = int(np.argmax(np.abs(command)))
        speed = command[axis]
        active = abs(speed) > self.contract.active_command_threshold
        mode = (axis + 1) * int(np.sign(speed)) if active else 0
        was_active = base.mode != 0
        phase = base.phase if active and was_active else 0
        history = base.action_history if active == was_active else self._zero_history
        reference = base.reference_heading if mode == base.mode else yaw
        observation = pack_actor_numpy(
            self.contract,
            gyro=gyro, accelerometer=accelerometer, command=command,
            joint_positions=joint_positions, joint_velocities=joint_velocities,
            home=self.home, action_history=history, last_sent=base.last_sent,
            contacts=contacts, phase=phase, yaw=yaw, reference_heading=reference,
        )
        return PreparedTick(_snapshot(observation), phase, mode, reference, history, base)

    def propose(self, prepared: PreparedTick, action) -> TargetTick:
        """Calculate a target to send, using zero action while stationary."""
        self._require_current(prepared)
        action = _finite_array(action, (self.contract.action_size,), "action")
        config = self.contract.action
        bounded = np.clip(action, *config["clip"]) if prepared.mode else np.zeros_like(action)
        requested = self.home + config["scale_rad"] * bounded
        last = prepared._base.last_sent
        delta = config["target_rate_rad_s"] * self.contract.clock.control_dt_s
        rate_limited = np.clip(requested, last - delta, last + delta)
        sent = np.clip(rate_limited, *self.joint_limits.T)
        return TargetTick(_snapshot(sent), _snapshot(bounded), prepared)

    def _require_current(self, prepared: PreparedTick) -> None:
        if not isinstance(prepared, PreparedTick) or prepared._base is not self._state:
            raise ValueError("Tick belongs to a different or superseded runtime state")

    def commit(self, target_tick: TargetTick) -> None:
        """Record a successful send. Discard a failed or skipped send without calling this."""
        if not isinstance(target_tick, TargetTick):
            raise ValueError("commit requires a proposed target tick")
        prepared = target_tick._prepared
        phase = (prepared.phase + 1) % self.contract.clock.phase_period_steps if prepared.mode else 0
        history = _snapshot(np.concatenate((target_tick.action[None, :], prepared._history[:2]), axis=0))
        with self._update:
            self._require_current(prepared)
            self._state = _RuntimeState(prepared.mode, phase, prepared.reference_heading, history, target_tick.sent)

