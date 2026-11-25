"""Resolve and run the project's Brax PPO training plan."""

from argparse import ArgumentParser, Namespace
from importlib.metadata import version
import json
from pathlib import Path

from miniduck.contract import load_contract, read_json
from miniduck.learning.networks import validate_network


def add_training_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="New run directory, for example runs/ppo-smoke")
    parser.add_argument("--config", type=Path, default=Path("configs/ppo.json"), help="PPO configuration path")
    parser.add_argument("--seed", type=int, help="Override the configuration seed")
    parser.add_argument("--curriculum", nargs="?", const=Path("configs/curriculum.json"), type=Path,
                        help="Enable the default curriculum or specify its configuration path")
    parser.add_argument("--smoke", action="store_true", help="Run a 160-transition PPO integration check")
    parser.add_argument("--plan", action="store_true", help="Print the resolved plan")


def _validate_sampling(ppo: dict, task: dict, phase_steps: int) -> tuple[int, int]:
    positive = ("num_envs", "num_eval_envs", "num_timesteps", "num_evals", "unroll_length", "batch_size",
                "num_minibatches", "num_updates_per_batch", "episode_length", "training_metrics_steps")
    for key in positive:
        if type(ppo[key]) is not int or ppo[key] < 1:
            raise ValueError(f"PPO {key} must be a positive integer")
    if type(ppo["seed"]) is not int or not 0 <= ppo["seed"] < 2**32:
        raise ValueError("PPO seed must be a wrapped 32-bit unsigned integer")
    resets = ppo["num_resets_per_eval"]
    if type(resets) is not int or resets < 0:
        raise ValueError("PPO num_resets_per_eval must be a nonnegative integer")
    warmup = task["command"]["warmup_steps"]
    if type(warmup) is not int or not 0 <= warmup < ppo["episode_length"]:
        raise ValueError("Warmup must finish before the time limit")
    if ppo["episode_length"] != task["episode_length"]:
        raise ValueError("PPO and task episode lengths must agree")
    if ppo["action_repeat"] != 1 or ppo["unroll_length"] != phase_steps:
        raise ValueError("V4 uses one action per control tick and a full gait cycle per unroll")
    if ppo["normalize_observations"] is not True or ppo["deterministic_eval"] is not True:
        raise ValueError("V4 uses running observation normalization and deterministic evaluation")
    if ppo["bootstrap_on_timeout"] is not False:
        raise ValueError("The V4 adapter uses Brax's default truncation handling")

    trajectories = ppo["batch_size"] * ppo["num_minibatches"]
    if trajectories % ppo["num_envs"]:
        raise ValueError("PPO minibatches must contain an integer number of parallel-environment rollouts")
    batch_transitions = trajectories * ppo["unroll_length"]
    intervals = max(ppo["num_evals"] - 1, 1)
    quantum = batch_transitions * intervals * max(resets, 1)
    if ppo["num_timesteps"] % quantum:
        raise ValueError(f"Choose a sample budget divisible by {quantum:,} to end on a complete training interval")
    return batch_transitions, intervals


def resolve_plan(root: Path, args: Namespace) -> tuple[dict, dict]:
    root = root.resolve()
    config_path = (root / args.config).resolve()
    robot = read_json(root / "configs/robot.json")
    task = read_json(root / "configs/task.json")
    settings = read_json(config_path)
    contract = load_contract(robot)
    ppo = dict(settings["ppo"])
    ppo.setdefault("bootstrap_on_timeout", False)
    if args.seed is not None:
        ppo["seed"] = args.seed
    if args.smoke:
        ppo.update({
            "num_envs": 4, "num_eval_envs": 2, "batch_size": 2, "num_minibatches": 2,
            "num_updates_per_batch": 1, "num_timesteps": 160, "num_evals": 2, "num_resets_per_eval": 0,
            "episode_length": 24, "log_training_metrics": True, "training_metrics_steps": 160,
            "max_devices_per_host": 1,
        })
        task["episode_length"] = 24
        task["command"]["warmup_steps"] = 4

    network = settings["network"]
    validate_network(network, contract)
    batch, intervals = _validate_sampling(ppo, task, contract.clock.phase_period_steps)
    curriculum_path = (root / args.curriculum).resolve() if args.curriculum is not None else None
    curriculum = read_json(curriculum_path) if curriculum_path is not None else None
    if curriculum is not None and ppo["num_resets_per_eval"]:
        raise ValueError("Use num_resets_per_eval=0 with curriculum to preserve environment ages and samples")
    eval_age = 0 if curriculum is None else max(
        curriculum["physics"]["ramp_steps"], curriculum["command"]["ramp_steps"],
        curriculum["roll"]["ramp_steps"], curriculum["guidance"]["fade_steps"][1],
    )
    samples_between_saves = ppo["num_timesteps"] // intervals
    plan = {
        "source": str(config_path),
        "output": str((root / args.output).resolve()),
        "baseline": task["baseline"],
        "smoke": args.smoke,
        "from_random_weights": True,
        "ppo": ppo,
        "network": network,
        "task": task,
        "curriculum": curriculum,
        "curriculum_path": str(curriculum_path) if curriculum_path is not None else None,
        "evaluation_curriculum_age": eval_age,
        "transitions_per_batch": batch,
        "expected_checkpoint_steps": [samples_between_saves * i for i in range(1, intervals + 1)],
    }
    return plan, robot


def run_training(root: Path, args: Namespace) -> None:
    plan, robot = resolve_plan(root, args)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return

    import jax
    import jax.numpy as jnp
    import numpy as np
    from brax.training.agents.ppo import checkpoint as brax_checkpoint
    from brax.training.agents.ppo import train as ppo_train

    from miniduck.env.env import DuckEnv
    from miniduck.learning.checkpoint import save_checkpoint, verify_checkpoint
    from miniduck.learning.metrics import RunLog
    from miniduck.learning.networks import make_network_factory
    from miniduck.learning.wrappers import BraxDuckEnv, wrap_for_brax

    versions = {name: version(name) for name in ("brax", "jax", "mujoco-mjx", "flax", "optax", "orbax-checkpoint")}
    if versions["brax"] != "0.14.2":
        raise RuntimeError("The V4 adapter was verified with Brax 0.14.2")
    plan["dependencies"] = versions
    plan["backend"] = jax.default_backend()

    physical_env = DuckEnv(root, curriculum=plan["curriculum_path"] or False, task_config=plan["task"])
    training_env = BraxDuckEnv(physical_env)
    evaluation_env = BraxDuckEnv(physical_env, curriculum_age=plan["evaluation_curriculum_age"])
    ppo = plan["ppo"]
    factory = make_network_factory(plan["network"])
    descriptor = brax_checkpoint.network_config(
        training_env.observation_size, training_env.action_size, ppo["normalize_observations"], factory,
    )
    probe = {name: jnp.linspace(-0.3, 0.3, shape[0], dtype=jnp.float32)
             for name, shape in training_env.observation_size.items()}
    policy_contract = {
        **robot,
        "training_baseline": plan["baseline"],
        "normalizer_source": "running statistics from PPO rollouts",
    }
    log = RunLog(Path(plan["output"]), plan, policy_contract)
    initial = None

    def report(step, metrics):
        log.record_metrics(step, metrics)

    def retain(step, make_policy, params):
        nonlocal initial
        step = int(step)
        if step == 0:
            initial = jax.tree.map(lambda value: np.array(value, copy=True), params)
            return

        path = save_checkpoint(log.output / "checkpoints", step, params, descriptor)
        maximum_error = verify_checkpoint(path, params=params, make_policy=make_policy, observation=probe)
        log.record_checkpoint(path, step, maximum_error)
        print(f"PPO checkpoint: step={step:,}, reload max action error={maximum_error:.3g}", flush=True)

    try:
        print(f"PPO start: {plan['backend']}, {ppo['num_envs']} environments, {ppo['num_timesteps']:,} transitions",
              flush=True)
        _, params, final_metrics = ppo_train.train(
            environment=training_env, eval_env=evaluation_env, network_factory=factory,
            wrap_env_fn=wrap_for_brax, progress_fn=report, policy_params_fn=retain, **ppo,
        )
        jax.effects_barrier()
        if initial is None:
            raise RuntimeError("Brax did not report initial parameters")
        names = ("normalizer", "actor", "critic")
        changed = {}
        for name, baseline, trained in zip(names, initial, params, strict=True):
            pairs = zip(jax.tree.leaves(baseline), jax.tree.leaves(trained), strict=True)
            changed[f"verification/{name}_updated"] = int(any(not np.array_equal(a, b) for a, b in pairs))
        if not all(changed.values()):
            raise RuntimeError(f"Training did not update all required parameter groups: {changed}")

        step = log.progress["actual_new_actor_samples"]
        losses = {key: value for key, value in final_metrics.items() if key.startswith("training/") and "loss" in key}
        if not losses or not all(np.isfinite(np.asarray(value)).all() for value in losses.values()):
            raise RuntimeError("Training did not report finite PPO losses")
        log.record_metrics(step, changed)
        log.complete(step)
        print(f"PPO complete: {step:,} transitions, actor/critic and normalizer updated", flush=True)
    except BaseException as exc:
        log.fail(exc)
        raise
