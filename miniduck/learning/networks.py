"""Brax network factory for the V4 policy and value function."""

from functools import partial

from miniduck.contract import PolicyContract


def validate_network(settings: dict, contract: PolicyContract) -> None:
    if (
        settings["policy_obs_key"] != contract.actor_key
        or settings["value_obs_key"] != contract.critic_key
        or settings["activation"] != "silu"
        or settings["distribution_type"] != "tanh_normal"
        or settings["use_distributional_critic"] is not False
    ):
        raise ValueError("Expected the V4 actor/critic keys, SiLU, tanh normal and scalar value")
    for key in ("policy_hidden_layer_sizes", "value_hidden_layer_sizes"):
        widths = settings[key]
        if not isinstance(widths, list) or not widths or any(type(width) is not int or width < 1 for width in widths):
            raise ValueError(f"{key} must be a nonempty list of positive widths")


def make_network_factory(settings: dict):
    import jax
    from brax.training.agents.ppo.networks import make_ppo_networks

    options = dict(settings)
    activation = options.pop("activation")
    if activation != "silu":
        raise ValueError(f"Unsupported PPO activation: {activation}")
    for key in ("policy_hidden_layer_sizes", "value_hidden_layer_sizes"):
        options[key] = tuple(options[key])
    return partial(make_ppo_networks, activation=jax.nn.silu, **options)
