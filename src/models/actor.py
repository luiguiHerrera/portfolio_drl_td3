"""Actor network for TD3 portfolio allocation.

This module defines the policy model that maps state observations to portfolio
weights. The actor produces logits internally and applies a softmax output so
the resulting actions are long-only and fully invested. Linear layers are
initialized explicitly so the initial policy is not excessively concentrated by
random logits.
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
        softmax_temperature: float = 1.0,
    ):
        super().__init__()

        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive.")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.softmax_temperature = softmax_temperature
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._initialize_weights()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Map state observations to long-only, fully invested portfolio weights."""
        logits = self.network(state)

        return torch.softmax(logits / self.softmax_temperature, dim=-1)

    def _initialize_weights(self) -> None:
        """Initialize hidden layers for ReLU and output logits near zero."""
        linear_layers = [module for module in self.network if isinstance(module, nn.Linear)]
        hidden_layers = linear_layers[:-1]
        output_layer = linear_layers[-1]

        for layer in hidden_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        nn.init.uniform_(output_layer.weight, -1e-3, 1e-3)
        nn.init.zeros_(output_layer.bias)
