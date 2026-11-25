"""Validate the robot assets and policy runtime in WSL."""

import argparse
import json
from hashlib import sha256
from math import isclose
from pathlib import Path

import mujoco
import numpy as np

from miniduck.contract import load_contract, read_json
from miniduck.env.model import load_model
from miniduck.env.randomize import load_perturbations
from miniduck.env.reference import CONTACTS, load_reference


ROOT = Path(__file__).resolve().parent.parent


def _verify(root: Path) -> None:
    robot = read_json(root / "configs/robot.json")
    task = read_json(root / "configs/task.json")
    training = read_json(root / "configs/ppo.json")
    contract = load_contract(robot)
    clock = contract.clock

    asset_root = root / "assets"
    manifest = read_json(asset_root / "manifest.json")["files"]
    required = (robot["model"], task["reference"]["path"], task["randomization"]["bank_path"])
    if any(path not in manifest for path in required):
        raise ValueError("A configured asset is missing from assets/manifest.json")
    for relative, origin in manifest.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(asset_root) or sha256(path.read_bytes()).hexdigest() != origin["sha256"]:
            raise ValueError(f"Asset failed SHA-256 verification: {relative}")

    network = training["network"]
    ppo = training["ppo"]
    if (
        network["policy_obs_key"] != contract.actor_key
        or network["value_obs_key"] != contract.critic_key
        or ppo["normalize_observations"] is not True
        or ppo["action_repeat"] != 1
        or ppo["unroll_length"] != clock.phase_period_steps
        or ppo["episode_length"] != task["episode_length"]
        or ppo["num_envs"] <= 0
        or ppo["batch_size"] * ppo["num_minibatches"] % ppo["num_envs"] != 0
    ):
        raise ValueError("PPO observation settings, control cadence or batch sizes conflict with the contract")

    model, indices = load_model(root / robot["model"], robot)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, indices.home_key_id)
    mujoco.mj_step(model, data, nstep=clock.substeps)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not isclose(data.time, clock.control_dt_s):
        raise ValueError("MuJoCo could not advance a finite state by one control period")

    reference_config = task["reference"]
    reference = load_reference(root / reference_config["path"], period_steps=clock.phase_period_steps)
    reference_frame_width = next(field.stop - field.start for field in contract.critic_fields if field.name == "reference_frame")
    if (
        tuple(reference_config["shape"]) != reference.frames.shape
        or reference.frames.shape[-1] != reference_frame_width
        or not isclose(reference.dt, clock.control_dt_s)
        or not np.allclose(reference.home[3:], robot["initial_qpos_wxyz"][3:], atol=1e-7, rtol=0.0)
        or reference_config["field_slices"]["contacts_left_right"] != [CONTACTS.start, CONTACTS.stop]
        or reference_config["contact_threshold_gt"] != 0.5
    ):
        raise ValueError("Reference gait shape, clock, rotation, joints or contact definition conflicts with the contract")

    phase = clock.phase_period_steps + 1
    if not np.array_equal(reference.frame(reference.commands[0], phase), reference.frames[0, 1]):
        raise ValueError("Reference lookup failed command and wrapped-phase check")

    bank = load_perturbations(root / task["randomization"]["bank_path"])
    nominal = sha256()
    for field, samples in bank.fields.items():
        values = np.ascontiguousarray(getattr(model, field), dtype=samples.dtype)
        nominal.update(field.encode("ascii"))
        nominal.update(values.tobytes())
    if nominal.hexdigest() != bank.metadata["base_model_sha256"]:
        raise ValueError("The perturbation bank belongs to a different nominal physical model")

    sampled = bank.sample(np.random.default_rng(0), 2)
    matching_rows = np.ones((2, bank.size), dtype=bool)
    for field, values in sampled.items():
        rows = values[:, None] == bank.fields[field][None, :]
        matching_rows &= rows.reshape(2, bank.size, -1).all(axis=-1)
    if not matching_rows.any(axis=1).all():
        raise ValueError("Perturbation sampling mixed fields across plant rows")

    print(f"Assets: {len(manifest)} files verified")
    print(f"Model: floating base, {model.nu} position actuators, home loaded")
    print(f"Clock: {clock.substeps} x {clock.physics_dt_s:g} s = {data.time:g} s per control, "
          f"{clock.phase_period_steps} steps = {clock.phase_period_s:g} s per gait cycle")
    print(f"Contract: actor={contract.actor_size}, critic={contract.critic_size}, actions={contract.action_size}")
    print(f"Reference: {' x '.join(map(str, reference.frames.shape))}, contacts {CONTACTS.start}:{CONTACTS.stop}")
    print(f"Perturbations: {bank.size} plant rows, {len(bank.fields)} fields, nominal model verified")



def _fixture_stem(root: Path) -> Path:
    directory = root / "fixtures/directional_v4"
    manifest = read_json(directory / "manifest.json")
    stem = manifest["stem"]
    if not stem or stem != Path(stem).name:
        raise ValueError("Invalid policy fixture stem")
    for extension in ("onnx", "npz", "json"):
        name = f"{stem}.{extension}"
        digest = sha256((directory / name).read_bytes()).hexdigest()
        if digest != manifest["files"][name]["sha256"]:
            raise ValueError(f"Policy fixture failed SHA-256 verification: {name}")
    return directory / stem


def _verify_alignment(root: Path, policy_stem: Path | None) -> None:
    from miniduck.inference.alignment import check_alignment
    from miniduck.inference.policy import load_policy_bundle

    robot = read_json(root / "configs/robot.json")
    bundle = load_policy_bundle(policy_stem or _fixture_stem(root), robot)
    result = check_alignment(root)
    observations = np.concatenate((result.observations, bundle.parity_inputs()), axis=0)
    tolerance = 1e-5
    max_error = 0.0
    for row, observation in enumerate(observations):
        differences = np.abs(bundle.infer_onnx(observation) - bundle.infer_numpy(observation))
        channel = int(np.argmax(differences))
        error = float(differences[channel])
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(
                f"Policy parity failed at sample {row}, action channel {channel}: "
                f"absolute error {error:.3g}, limit {tolerance:g}"
            )
        max_error = max(error, max_error)

    print(f"Observation: {result.actor_fields} actor fields, {result.critic_fields} critic-only fields")
    print(f"Targets: verified {', '.join(map(str, result.delay_steps))} control-tick delays")
    print(f"Runtime: {result.deployment_ticks} scripted ticks checked, including retry and phase wrap")
    print(f"Policy: ONNX vs NumPy on {len(observations)} raw inputs, max absolute error {max_error:.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Verify model, data and policy contract")
    commands.add_parser("validate-curriculum", help="Verify curriculum, perturbations and reward metrics")
    alignment = commands.add_parser("validate-alignment", help="Compare training and runtime features and policy outputs")
    alignment.add_argument("--policy-stem", type=Path, help="Common path of compatible ONNX, NPZ and JSON files")
    from miniduck.learning.train import add_training_arguments

    add_training_arguments(commands.add_parser("train", help="Plan or run Brax PPO training"))

    export = commands.add_parser("export", help="Export and cross-check a deterministic PPO actor")
    export.add_argument("checkpoint", type=Path, help="Brax PPO checkpoint directory")
    export.add_argument("destination", type=Path, help="New ONNX output file")
    record = commands.add_parser("record", help="Run fixed-command native MuJoCo and record an MP4")
    record.add_argument("--policy", type=Path, required=True, help="ONNX file with its matching sidecars")
    record.add_argument("--command", dest="policy_command", type=float, nargs=3, required=True,
                        metavar=("VX", "VY", "YAW_RATE"))
    record.add_argument("--duration", type=float, default=6.0, help="Seconds of fixed command after warmup")
    record.add_argument("--video", type=Path, required=True, help="New MP4 output file")
    record.add_argument("--report", type=Path, help="New JSON file for this simulation's results")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            _verify(ROOT)
        elif args.command == "validate-alignment":
            _verify_alignment(ROOT, args.policy_stem)
        elif args.command == "train":
            from miniduck.learning.train import run_training

            run_training(ROOT, args)
        elif args.command == "export":
            from miniduck.inference.export import export_actor

            result = export_actor(args.checkpoint, args.destination)
            print(json.dumps(result, indent=2, default=str))
        elif args.command == "record":
            from miniduck.inference.simulate import run_simulation

            if args.policy.suffix.lower() != ".onnx":
                raise ValueError("Expected an ONNX policy file")
            report = run_simulation(
                ROOT, args.policy.with_suffix(""), command=args.policy_command, duration=args.duration,
                video_path=args.video, report_path=args.report,
            )
            print(json.dumps(report, indent=2, default=str))
        else:
            from miniduck.env.curriculum_checks import check_curriculum

            check_curriculum(ROOT)
    except (AssertionError, ImportError, KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")


if __name__ == "__main__":
    main()
