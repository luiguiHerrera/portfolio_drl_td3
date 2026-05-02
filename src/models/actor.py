"""Actor network scaffold for TD3 portfolio allocation.

This module will define the policy model that maps market states to portfolio
weights. The initial design target is a long-only, fully invested action space,
so the final actor output is expected to be non-negative and sum to one,
potentially through a softmax transformation.
"""


class Actor:
    """Policy network interface for portfolio weights."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Actor architecture has not been implemented yet.")

    def forward(self, state):
        """Map a state representation to portfolio weights."""
        raise NotImplementedError("Actor forward pass has not been implemented yet.")
