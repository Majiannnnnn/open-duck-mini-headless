"""Adapt the DuckEnv transition API to Brax training and evaluation."""

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State
from brax.envs.wrappers.training import VmapWrapper

from miniduck.env.env import DuckEnv, State as DuckState


class BraxDuckEnv(Env):
    """Expose a single rollout lane with DuckEnv's own episode reset semantics."""

    def __init__(self, env: DuckEnv, *, curriculum_age: int = 0):
        if type(curriculum_age) is not int or curriculum_age < 0:
            raise ValueError("Initial curriculum age must be a nonnegative integer")
        self.env = env
        self.curriculum_age = curriculum_age

    @property
    def episode_length(self):
        return self.env.episode_length

    @property
    def action_size(self):
        return self.env.contract.action_size

    @property
    def observation_size(self):
        contract = self.env.contract
        return {
            contract.actor_key: (contract.actor_size,),
            contract.critic_key: (contract.critic_size,),
        }

    @property
    def backend(self):
        return "mjx"

    @staticmethod
    def _step_metrics(state: DuckState):
        # DuckEnv already accumulates these counters. Brax sums the per-step values below.
        return {
            name: value for name, value in state.metrics.items()
            if name not in ("episode/return", "episode/steps")
        }

    def reset(self, rng):
        current = self.env.reset(rng, self.curriculum_age)
        metrics = self._step_metrics(current)
        zero = jnp.zeros_like(current.reward)
        return self._brax_state(
            current,
            current,
            metrics,
            {**jax.tree.map(jnp.zeros_like, metrics), "sum_reward": zero, "length": zero},
        )

    def step(self, state: State, action):
        transition, current = self.env.step_and_reset(state.pipeline_state, action)
        metrics = self._step_metrics(transition)
        prior = state.info["episode_metrics"]
        continuation_mask = 1.0 - state.info["episode_done"]
        episode_metrics = {
            name: prior[name] * continuation_mask + value
            for name, value in metrics.items()
        }
        episode_metrics["sum_reward"] = transition.metrics["episode/return"]
        episode_metrics["length"] = transition.task.steps.astype(jnp.float32)
        return self._brax_state(transition, current, metrics, episode_metrics)

    @staticmethod
    def _brax_state(transition: DuckState, current: DuckState, metrics, episode_metrics):
        done = transition.done.astype(jnp.float32)
        return State(
            pipeline_state=current,
            obs=current.obs,
            reward=transition.reward,
            done=done,
            metrics=metrics,
            info={
                "steps": transition.task.steps.astype(jnp.float32),
                "truncation": transition.truncated.astype(jnp.float32),
                "episode_done": done,
                "episode_metrics": episode_metrics,
                "terminated": transition.terminated,
                "truncated": transition.truncated,
                "termination_reason": transition.termination_reason,
                "terminal_observation": jax.tree.map(
                    lambda value: jnp.where(transition.done, value, jnp.zeros_like(value)),
                    transition.obs,
                ),
            },
        )


def wrap_for_brax(env: BraxDuckEnv, episode_length, action_repeat, randomization_fn=None):
    """Vectorize reset and step with a different key for every rollout lane."""
    if episode_length != env.episode_length:
        raise ValueError("Brax and DuckEnv episode lengths must match")
    if action_repeat != 1:
        raise ValueError("DuckEnv already advances one full control tick per action")
    if randomization_fn is not None:
        raise ValueError("Set per-environment randomization in DuckEnv")
    return VmapWrapper(env)
