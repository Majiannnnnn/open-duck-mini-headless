"""Policy actions and position targets in the shared joint order."""

import jax.numpy as jnp


def position_targets(action, previous, queue, delay_steps, home, limits, *, scale, max_delta):
    """Return the sent target, actuator target, and updated delay history."""
    requested = home + scale * action
    rate_limited = jnp.clip(requested, previous - max_delta, previous + max_delta)
    sent = jnp.clip(rate_limited, limits[:, 0], limits[:, 1])
    queue = jnp.concatenate((sent[None], queue[:-1]), axis=0)
    return sent, queue[delay_steps], queue
