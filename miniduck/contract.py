"""Shared interpretation of the robot and policy configuration."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Clock:
    control_dt_s: float
    physics_dt_s: float
    substeps: int
    phase_period_steps: int
    phase_period_s: float


@dataclass(frozen=True)
class ObservationField:
    name: str
    start: int
    stop: int
    unit: str
    meaning: str

    @property
    def indices(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class PolicyContract:
    joint_order: tuple[str, ...]
    clock: Clock
    actor_key: str
    critic_key: str
    actor_fields: tuple[ObservationField, ...]
    critic_fields: tuple[ObservationField, ...]
    action_size: int
    action: dict
    observations: dict

    @property
    def active_command_threshold(self) -> float:
        return self.observations.get("active_command_threshold", 0.01)

    @property
    def actor_size(self) -> int:
        return self.actor_fields[-1].stop

    @property
    def critic_size(self) -> int:
        return self.critic_fields[-1].stop


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _check_layout(fields: tuple[ObservationField, ...], expected_size: int) -> None:
    cursor = 0
    names = set()
    for field in fields:
        if (
            field.start != cursor or field.stop <= field.start
            or not field.name or field.name in names or not field.unit or not field.meaning
        ):
            raise ValueError(f"Invalid observation field {field.name} at index {cursor}")
        names.add(field.name)
        cursor = field.stop
    if cursor != expected_size:
        raise ValueError(f"Observation layout has {cursor} fields, expected {expected_size} channels")


def load_contract(robot: dict) -> PolicyContract:
    """Load the left-first, 75/175/10 heading policy contract."""
    if robot["schema_version"] != 1:
        raise ValueError("Unsupported robot configuration schema")

    clock = Clock(**robot["clock"])
    if clock != Clock(0.02, 0.002, 10, 40, 0.8):
        raise ValueError("The v1 protocol requires 0.02 s control, 0.002 s physics, and a 40-step phase")

    order = tuple(robot["joint_order"])
    expected = tuple(
        f"{side}_{joint}"
        for side in ("left", "right")
        for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
    )
    if order != expected:
        raise ValueError("Unexpected joint order; the v1 protocol uses left leg then right leg")

    policy = robot["policy"]
    sizes = policy["sizes"]
    if sizes != {"actor": 75, "critic": 175, "action": len(order)}:
        raise ValueError("The v1 policy requires 75 actor, 175 critic, and 10 action channels")
    actor = tuple(ObservationField(**field) for field in policy["actor_fields"])
    critic = actor + tuple(ObservationField(**field) for field in policy["critic_extra_fields"])
    _check_layout(actor, sizes["actor"])
    _check_layout(critic, sizes["critic"])

    obs = policy["observations"]
    if (
        obs["joint_velocity_scale"] != 0.05
        or obs.get("active_command_threshold", 0.01) != 0.01
        or obs["phase_order"] != ["cos", "sin"]
        or obs["heading_order"] != ["sin", "cos"]
        or obs["normalization"]["method"] != "running_mean_std"
        or obs["normalization"]["runtime_pre_normalize"] is not False
    ):
        raise ValueError("Unexpected feature scaling, feature order, or normalization placement")

    action = policy["action"]
    if (
        action["clip"] != [-1.0, 1.0] or action["scale_rad"] != 0.25
        or action["target_rate_rad_s"] != 5.24
        or action["target_order"] != [
            "clip_action", "scale_and_add_home", "rate_limit_from_previous_target", "clip_joint_limits",
        ]
    ):
        raise ValueError("Unexpected action range, scale or position-target limits")
    if robot["default_joint_positions_rad"] != robot["initial_qpos_wxyz"][7:]:
        # The parent snapshot writes the qpos and the home angles at different precision.
        if len(robot["default_joint_positions_rad"]) != len(order) or any(
            abs(home - initial) > 1e-7
            for home, initial in zip(robot["default_joint_positions_rad"], robot["initial_qpos_wxyz"][7:], strict=True)
        ):
            raise ValueError("Default joint angles differ from the configured home qpos")

    return PolicyContract(
        order, clock, policy["actor_key"], policy["critic_key"], actor, critic, sizes["action"], action, obs,
    )
