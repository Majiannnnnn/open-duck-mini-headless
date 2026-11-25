"""Named robot sensors and contact measurements shared by the V4 environment."""

from typing import NamedTuple

import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import support
import numpy as np
from numpy.typing import NDArray

from miniduck.env.model import ModelIndices


class SceneIndices(NamedTuple):
    imu_site_id: int
    gyro_indices: NDArray[np.int32]
    accelerometer_indices: NDArray[np.int32]
    local_linear_velocity_indices: NDArray[np.int32]
    global_angular_velocity_indices: NDArray[np.int32]
    foot_velocity_indices: NDArray[np.int32]
    foot_geom_ids: NDArray[np.int32]
    floor_geom_id: int
    home_gravity: NDArray[np.float64]
    sole_vertices: tuple[NDArray[np.float32], NDArray[np.float32]]


def _sensor_indices(
    model: mujoco.MjModel, name: str, site_id: int, sensor_type: mujoco.mjtSensor
) -> NDArray[np.int32]:
    sensor_id = model.sensor(name).id
    if (
        model.sensor_type[sensor_id] != sensor_type
        or model.sensor_dim[sensor_id] != 3
        or model.sensor_objtype[sensor_id] != mujoco.mjtObj.mjOBJ_SITE
        or model.sensor_objid[sensor_id] != site_id
    ):
        raise ValueError(f"Sensor {name!r} must be a three-axis signal at its named site")
    address = model.sensor_adr[sensor_id]
    return np.arange(address, address + 3, dtype=np.int32)


def load_scene(model: mujoco.MjModel, indices: ModelIndices) -> SceneIndices:
    """Read sensor addresses and compiled collision mesh vertices."""
    imu = model.site("imu").id
    gyro = _sensor_indices(model, "gyro", imu, mujoco.mjtSensor.mjSENS_GYRO)
    accel = _sensor_indices(model, "accelerometer", imu, mujoco.mjtSensor.mjSENS_ACCELEROMETER)
    local_velocity = _sensor_indices(model, "local_linvel", imu, mujoco.mjtSensor.mjSENS_VELOCIMETER)
    angular_velocity = _sensor_indices(model, "global_angvel", imu, mujoco.mjtSensor.mjSENS_FRAMEANGVEL)

    foot_velocities = []
    foot_geoms = []
    soles = []
    for side in ("left", "right"):
        foot = f"{side}_foot"
        site_id = model.site(foot).id
        foot_velocities.append(_sensor_indices(
            model, f"{foot}_global_linvel", site_id, mujoco.mjtSensor.mjSENS_FRAMELINVEL,
        ))
        geom_id = model.geom(f"{foot}_bottom_tpu").id
        mesh_id = model.geom_dataid[geom_id]
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH or mesh_id < 0:
            raise ValueError(f"{foot} must have a collision mesh")
        first_vertex = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        foot_geoms.append(geom_id)
        soles.append(model.mesh_vert[first_vertex:first_vertex + count].copy())

    floor = model.geom("floor").id
    if (
        model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE or model.geom_condim[floor] != 3
        or np.any(model.geom_priority[foot_geoms] > model.geom_priority[floor])
        or np.any(
            (model.geom_priority[foot_geoms] == model.geom_priority[floor])
            & (model.geom_condim[foot_geoms] > 3)
        )
    ):
        raise ValueError("V4 expects three-dimensional sole contacts with a plane floor")

    home_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, home_data, indices.home_key_id)
    mujoco.mj_kinematics(model, home_data)
    home_gravity = -home_data.site_xmat[imu].reshape(3, 3)[2].copy()
    return SceneIndices(
        imu, gyro, accel, local_velocity, angular_velocity, np.stack(foot_velocities),
        np.array(foot_geoms, dtype=np.int32), floor, home_gravity, (soles[0], soles[1]),
    )


def sole_heights(model: mjx.Model, data: mjx.Data, scene: SceneIndices) -> jnp.ndarray:
    """Lowest point of each compiled sole mesh above the horizontal plane."""
    del model
    geom_xpos = jnp.asarray(data.geom_xpos)
    geom_xmat = jnp.asarray(data.geom_xmat).reshape(-1, 3, 3)
    heights = geom_xpos[scene.foot_geom_ids, 2] - geom_xpos[scene.floor_geom_id, 2]
    vertical = geom_xmat[scene.foot_geom_ids, 2, :]
    offsets = jnp.stack([
        jnp.min(vertices @ axis) for vertices, axis in zip(scene.sole_vertices, vertical, strict=True)
    ])
    return heights + offsets


def foot_contacts_and_loads(
    model: mjx.Model, data: mjx.Data, scene: SceneIndices
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return floor contact flags and normal loads in left/right order."""
    contact = data._impl.contact
    feet = scene.foot_geom_ids
    first, second = contact.geom.T
    floor_first = (first == scene.floor_geom_id)[:, None]
    floor_second = (second == scene.floor_geom_id)[:, None]
    pairs = (floor_first & (second[:, None] == feet)) | (floor_second & (first[:, None] == feet))
    touching = jnp.any(pairs & (contact.dist <= 0)[:, None], axis=0)

    # MJX 3.6 decodes normal force from its solver rows using the chosen cone type.
    wrenches, rows = support.contact_force_dim(model, data, 3)
    loads = jnp.sum(jnp.maximum(wrenches[:, :1], 0) * pairs[rows], axis=0)
    return touching, loads
