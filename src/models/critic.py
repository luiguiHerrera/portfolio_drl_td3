"""Critic network scaffold for TD3 portfolio allocation.

This module will define the value estimators used by TD3 to evaluate
state-action pairs. TD3 requires two independent critics to reduce
overestimation bias, but their architecture and inputs should be finalized only
after the environment state and action representations are validated.
"""


class Critic:
    """State-action value estimator interface."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Critic architecture has not been implemented yet.")

    def forward(self, state, action):
        """Estimate the value of a state-action pair."""
        raise NotImplementedError("Critic forward pass has not been implemented yet.")
