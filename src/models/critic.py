"""Critic network for TD3 portfolio allocation.

This module defines a state-action value estimator for TD3. TD3 will use two
independent instances of this network to reduce overestimation bias.
"""

import torch
from torch import nn


class CriticNetwork(nn.Module):
    """MLP critic that estimates Q(s, a)."""

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
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Estimate Q-values for state-action pairs."""
        state_action = torch.cat((state, action), dim=-1)

        return self.network(state_action)
