"""Run the shared actor and target sender against native MuJoCo."""

from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from miniduck.contract import read_json
from miniduck.env.model import load_model
from miniduck.env.scene import load_scene
from miniduck.inference.policy import load_policy_bundle
from miniduck.inference.runtime import Runtime


_VIDEO_FPS = 25
_VIDEO_WIDTH = 800
_VIDEO_HEIGHT = 600
_WARMUP_SECONDS = 1.0
_FALL_UPRIGHT_COSINE = 0.5


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if binary is not None:
        return binary
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as error:
        raise RuntimeError("Recording requires FFmpeg or imageio-ffmpeg") from error
    return get_ffmpeg_exe()


@contextmanager
def _recording(model, output: Path):
    """Publish a video once the renderer and encoder have both finished."""
    output.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, _VIDEO_WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, _VIDEO_HEIGHT)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 0.78
    camera.azimuth = 130
    camera.elevation = -15
    with TemporaryDirectory(prefix=".recording-", dir=output.parent) as directory:
        video = Path(directory) / "simulation.mp4"
        command = [
            _ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{_VIDEO_WIDTH}x{_VIDEO_HEIGHT}",
            "-framerate", str(_VIDEO_FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
        ]
        with mujoco.Renderer(model, height=_VIDEO_HEIGHT, width=_VIDEO_WIDTH) as renderer:
            with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as encoder:
                last_frame = None

                def capture(data, *, write=True):
                    nonlocal last_frame
                    if data is not None:
                        camera.lookat[:] = data.qpos[:3] + (0, 0, 0.035)
                        renderer.update_scene(data, camera=camera)
                        last_frame = renderer.render().tobytes()
                    if write:
                        if last_frame is None:
                            raise RuntimeError("No valid simulation frame has been rendered")
                        encoder.stdin.write(last_frame)

                try:
                    yield capture
                except BrokenPipeError as error:
                    diagnostic = encoder.stderr.read().decode(errors="replace").strip()
                    raise RuntimeError(f"FFmpeg stopped accepting frames: {diagnostic}") from error
                finally:
                    encoder.stdin.close()
                diagnostic = encoder.stderr.read().decode(errors="replace").strip()
                if encoder.wait() or not video.is_file() or video.stat().st_size == 0:
                    raise RuntimeError(f"FFmpeg did not write an MP4: {diagnostic}")
        os.link(video, output)


def _sole_height(data, scene) -> float:
    heights = []
    for geom, vertices in zip(scene.foot_geom_ids, scene.sole_vertices, strict=True):
        z_axis = data.geom_xmat[geom].reshape(3, 3)[2]
        heights.append(data.geom_xpos[geom, 2] + np.min(vertices @ z_axis))
    return float(min(heights) - data.geom_xpos[scene.floor_geom_id, 2])


def _foot_contacts(data, scene) -> np.ndarray:
    pairs = data.contact.geom
    floor = scene.floor_geom_id
    touches = np.any(pairs == floor, axis=1) & (data.contact.dist <= 0)
    return np.any(touches[:, None] & np.any(pairs[:, :, None] == scene.foot_geom_ids, axis=1), axis=0)


def _yaw(rotation) -> float:
    return math.atan2(rotation[1, 0], rotation[0, 0])


def _rollout(model, data, indices, scene, runtime, infer, command, motion_ticks, capture):
    clock = runtime.contract.clock
    warmup_ticks = round(_WARMUP_SECONDS / clock.control_dt_s)
    stride = round((1.0 / _VIDEO_FPS) / clock.control_dt_s)
    no_command = np.zeros(3, dtype=np.float32)
    no_action = np.zeros(runtime.contract.action_size, dtype=np.float32)
    origin = None
    displacement = np.zeros(2)
    motion_yaw = 0.0
    previous_yaw = 0.0
    yaw_change = 0.0
    fall_tick = None
    termination_reason = "duration_elapsed"
    invalid_physics = False
    physical_ticks = 0
    commanded_ticks = 0
    target_changes = 0
    video_frames = 0

    if capture:
        capture(data, write=False)
    for tick in range(warmup_ticks + motion_ticks):
        motion_tick = tick - warmup_ticks
        rotation = data.site_xmat[scene.imu_site_id].reshape(3, 3)
        if motion_tick == 0 and not invalid_physics:
            origin = data.qpos[:2].copy()
            motion_yaw = previous_yaw = _yaw(rotation)
        if capture and motion_tick >= 0 and motion_tick % stride == 0:
            capture(None if invalid_physics else data)
            video_frames += 1
        if invalid_physics:
            continue

        moving = motion_tick >= 0 and fall_tick is None
        prepared = runtime.prepare(
            command if moving else no_command,
            data.sensordata[scene.gyro_indices],
            data.sensordata[scene.accelerometer_indices],
            data.qpos[indices.qpos_indices],
            data.qvel[indices.qvel_indices],
            _foot_contacts(data, scene),
            _yaw(rotation),
        )
        action = infer(prepared.observation) if prepared.mode else no_action
        target = runtime.propose(prepared, action)
        target_changes += int(np.any(target.sent != runtime.last_sent))
        data.ctrl[indices.actuator_ids] = target.sent
        runtime.commit(target)
        commanded_ticks += int(prepared.mode != 0)
        physical_ticks += 1
        previous_time = data.time
        mujoco.mj_step(model, data, nstep=clock.substeps)
        mujoco.mj_forward(model, data)

        finite = (
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.sensordata).all()
            and math.isclose(data.time - previous_time, clock.control_dt_s, abs_tol=1e-9)
        )
        if not finite:
            invalid_physics = True
            termination_reason = "invalid_physics"
            continue
        rotation = data.site_xmat[scene.imu_site_id].reshape(3, 3)
        if rotation[2, 2] < _FALL_UPRIGHT_COSINE and fall_tick is None:
            fall_tick = tick + 1
            termination_reason = "fall"
        if origin is not None:
            displacement = data.qpos[:2] - origin
            yaw = _yaw(rotation)
            yaw_change += math.remainder(yaw - previous_yaw, 2 * math.pi)
            previous_yaw = yaw

    c = math.cos(motion_yaw)
    s = math.sin(motion_yaw)
    local_displacement = [c * displacement[0] + s * displacement[1], -s * displacement[0] + c * displacement[1]]
    return {
        "termination_reason": termination_reason,
        "fell": fall_tick is not None,
        "fall_time_seconds": None if fall_tick is None else fall_tick * clock.control_dt_s,
        "warmup_seconds": _WARMUP_SECONDS,
        "elapsed_seconds": physical_ticks * clock.control_dt_s,
        "active_seconds": max(physical_ticks - warmup_ticks, 0) * clock.control_dt_s,
        "commanded_seconds": commanded_ticks * clock.control_dt_s,
        "control_ticks": physical_ticks,
        "active_ticks": max(physical_ticks - warmup_ticks, 0),
        "physics_steps": physical_ticks * clock.substeps,
        "physics_steps_per_control_tick": clock.substeps,
        "inference_calls": commanded_ticks,
        "actuator_updates": physical_ticks,
        "target_change_ticks": target_changes,
        "displacement_xy_m": displacement.tolist(),
        "displacement_forward_left_m": local_displacement,
        "yaw_change_rad": yaw_change,
        "video_frames": video_frames,
    }


def run_simulation(
    project_root, policy_stem, *, command, duration, video_path=None, report_path=None,
) -> dict:
    """Run one fixed command after standing warmup and optionally record its MP4."""
    root = Path(project_root).resolve()
    robot = read_json(root / "configs/robot.json")
    task = read_json(root / "configs/task.json")
    runtime = Runtime(robot)
    command = np.asarray(command, dtype=np.float32)
    ranges = np.asarray(task["command"]["magnitude_range"], dtype=np.float32)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError("Command must contain three finite numbers: vx, vy, yaw_rate")
    if np.count_nonzero(command) > 1 or np.any(np.abs(command) > ranges[:, 1]):
        raise ValueError("Use a single-axis command within the V4 speed limits")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Simulation duration must be finite and positive")
    clock = runtime.contract.clock
    motion_ticks = round(duration / clock.control_dt_s)
    if not motion_ticks:
        raise ValueError("Simulation duration must include at least one control tick")

    video = None if video_path is None else (root / Path(video_path)).resolve()
    report_file = None if report_path is None else (root / Path(report_path)).resolve()
    if video is not None and video.suffix.lower() != ".mp4":
        raise ValueError("The recording output must have an .mp4 extension")
    if video is not None and video == report_file:
        raise ValueError("Video and report must use separate output paths")
    for path in (video, report_file):
        if path is not None and path.exists():
            raise FileExistsError(path)

    policy_stem = (root / Path(policy_stem)).resolve()
    bundle = load_policy_bundle(policy_stem, robot)
    model, indices = load_model(root / robot["model"], robot)
    scene = load_scene(model, indices)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, indices.home_key_id)
    mujoco.mj_kinematics(model, data)
    data.qpos[2] += task["reset"]["sole_clearance_m"] - _sole_height(data, scene)
    mujoco.mj_forward(model, data)
    with _recording(model, video) if video is not None else nullcontext() as capture:
        result = _rollout(model, data, indices, scene, runtime, bundle.infer_onnx, command, motion_ticks, capture)
    result.update({
        "policy": str(policy_stem.with_name(policy_stem.name + ".onnx")),
        "policy_sha256": bundle.onnx_sha256,
        "command": command.tolist(),
        "requested_active_seconds": float(duration),
        "control_dt_seconds": clock.control_dt_s,
        "physics_dt_seconds": clock.physics_dt_s,
        "heading_source": "mujoco_imu_site",
        "video_path": None if video is None else str(video),
        "video_fps": None if video is None else _VIDEO_FPS,
    })
    if report_file is not None:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, allow_nan=False, indent=2)
            stream.write("\n")
    return result
