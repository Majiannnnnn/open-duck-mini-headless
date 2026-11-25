"""Load the flat MuJoCo model and map it to the shared joint order."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ModelIndices:
    """Indices in protocol order, independent of MJCF declaration order."""

    joint_ids: NDArray[np.intp]
    qpos_indices: NDArray[np.intp]
    qvel_indices: NDArray[np.intp]
    actuator_ids: NDArray[np.intp]
    home_key_id: int


def _named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"MuJoCo model is missing {kind.name} {name!r}")
    return index


def load_model(path: Path, robot: dict) -> tuple[mujoco.MjModel, ModelIndices]:
    """Verify compiled joints, servos, and home pose before using the model."""

    model = mujoco.MjModel.from_xml_path(str(path))
    joint_order = robot["joint_order"]
    count = len(joint_order)
    if count != 10 or len(set(joint_order)) != count:
        raise ValueError("The robot protocol requires ten unique joint names")
    if (model.nq, model.nv, model.njnt, model.nu) != (17, 16, 11, 10):
        raise ValueError("The model must have a floating base and ten actuated joints")

    root_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, robot["floating_joint"])
    if (
        model.jnt_type[root_id] != mujoco.mjtJoint.mjJNT_FREE
        or model.jnt_qposadr[root_id] != 0
        or model.jnt_dofadr[root_id] != 0
    ):
        raise ValueError("The floating joint must occupy the first qpos and qvel entries")
    floor_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("The flat-terrain floor must be a MuJoCo plane")

    joint_ids = np.array(
        [_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_order],
        dtype=np.intp,
    )
    actuator_ids = np.array(
        [_named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in joint_order],
        dtype=np.intp,
    )
    qpos_indices = model.jnt_qposadr[joint_ids].astype(np.intp)
    qvel_indices = model.jnt_dofadr[joint_ids].astype(np.intp)
    if (
        not np.all(model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_HINGE)
        or not np.all(model.jnt_limited[joint_ids])
        or not np.array_equal(np.sort(qpos_indices), np.arange(7, 17))
        or not np.array_equal(np.sort(qvel_indices), np.arange(6, 16))
    ):
        raise ValueError("Each protocol joint must be a limited one-DOF hinge")

    limits = np.asarray(robot["joint_limits_rad"], dtype=float)
    if limits.shape != (count, 2) or not np.allclose(
        model.jnt_range[joint_ids], limits, rtol=0.0, atol=1e-8
    ):
        raise ValueError("Compiled joint limits differ from the robot configuration")

    gain = model.actuator_gainprm[actuator_ids, 0]
    bias = model.actuator_biasprm[actuator_ids, :3]
    gear = model.actuator_gear[actuator_ids]
    if (
        not np.all(model.actuator_trntype[actuator_ids] == mujoco.mjtTrn.mjTRN_JOINT)
        or not np.array_equal(model.actuator_trnid[actuator_ids, 0], joint_ids)
        or not np.allclose(gear, [1, 0, 0, 0, 0, 0], rtol=0.0, atol=1e-8)
        or not np.all(model.actuator_dyntype[actuator_ids] == mujoco.mjtDyn.mjDYN_NONE)
        or not np.all(model.actuator_gaintype[actuator_ids] == mujoco.mjtGain.mjGAIN_FIXED)
        or not np.all(model.actuator_biastype[actuator_ids] == mujoco.mjtBias.mjBIAS_AFFINE)
        or not np.all(np.isfinite(gain) & (gain > 0))
        or not np.all(np.isfinite(bias))
        or not np.allclose(bias[:, 0], 0, rtol=0.0, atol=1e-8)
        or not np.allclose(bias[:, 1], -gain, rtol=0.0, atol=1e-8)
        or not np.all(bias[:, 2] <= 0)
    ):
        raise ValueError("Each named actuator must be a position servo on its matching joint")
    if not np.all(model.actuator_ctrllimited[actuator_ids]) or not np.allclose(
        model.actuator_ctrlrange[actuator_ids], limits, rtol=0.0, atol=1e-8
    ):
        raise ValueError("Position actuator control ranges must match the configured joint limits")

    home_key_id = _named_id(model, mujoco.mjtObj.mjOBJ_KEY, robot["home_keyframe"])
    initial_qpos = np.asarray(robot["initial_qpos_wxyz"], dtype=float)
    actual_home = model.key_qpos[home_key_id, np.r_[0:7, qpos_indices]]
    if (
        initial_qpos.shape != (17,)
        or not np.allclose(actual_home, initial_qpos, rtol=0.0, atol=1e-7)
        or not np.allclose(
            model.key_ctrl[home_key_id, actuator_ids], initial_qpos[7:], rtol=0.0, atol=1e-7
        )
    ):
        raise ValueError("The MJCF home keyframe differs from the configured initial pose")

    model.opt.timestep = robot["clock"]["physics_dt_s"]
    return model, ModelIndices(joint_ids, qpos_indices, qvel_indices, actuator_ids, home_key_id)
