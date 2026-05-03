"""Actor network for TD3 portfolio allocation.

This module defines the policy model that maps state observations to portfolio
weights. The actor produces logits internally and applies a softmax output so
the resulting actions are long-only and fully invested.
"""

import torch
from torch import nn


class ActorNetwork(nn.Module):
    """MLP actor that outputs portfolio weights."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Map state observations to long-only, fully invested portfolio weights."""
        logits = self.network(state)

        return torch.softmax(logits, dim=-1)
