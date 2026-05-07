"""Replay buffer for off-policy TD3 training.

This module stores transitions collected from the portfolio environment and
provides random mini-batches for later TD3 updates. It deliberately uses NumPy
arrays only; conversion to PyTorch tensors should happen inside the future agent
or training code.
"""

import numpy as np


class ReplayBuffer:
    """Fixed-size circular experience buffer."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int,
        seed: int | None = None,
    ):
        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if max_size <= 0:
            raise ValueError("max_size must be positive.")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((max_size, state_dim), dtype=float)
        self.actions = np.zeros((max_size, action_dim), dtype=float)
        self.rewards = np.zeros((max_size, 1), dtype=float)
        self.next_states = np.zeros((max_size, state_dim), dtype=float)
        self.dones = np.zeros((max_size, 1), dtype=bool)

    def add(self, state, action, reward, next_state, done) -> None:
        """Store one transition in the circular buffer."""
        state_array = np.asarray(state, dtype=float)
        action_array = np.asarray(action, dtype=float)
        next_state_array = np.asarray(next_state, dtype=float)
        reward_value = self._validate_scalar_numeric(reward, "reward")
        done_value = self._validate_scalar_boolean(done)

        self._validate_vector_shape(state_array, self.state_dim, "state")
        self._validate_vector_shape(action_array, self.action_dim, "action")
        self._validate_vector_shape(next_state_array, self.state_dim, "next_state")

        self.states[self.ptr] = state_array
        self.actions[self.ptr] = action_array
        self.rewards[self.ptr] = reward_value
        self.next_states[self.ptr] = next_state_array
        self.dones[self.ptr] = done_value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> dict:
        """Sample a mini-batch of transitions without replacement."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.size < batch_size:
            raise ValueError("batch_size cannot exceed current buffer size.")

        indexes = self.rng.choice(self.size, size=batch_size, replace=False)

        return {
            "states": self.states[indexes],
            "actions": self.actions[indexes],
            "rewards": self.rewards[indexes],
            "next_states": self.next_states[indexes],
            "dones": self.dones[indexes],
        }

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self.size

    @staticmethod
    def _validate_vector_shape(array: np.ndarray, expected_dim: int, name: str) -> None:
        expected_shape = (expected_dim,)
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}.")

    @staticmethod
    def _validate_scalar_numeric(value, name: str) -> float:
        value_array = np.asarray(value)
        if value_array.shape != ():
            raise ValueError(f"{name} must be a scalar numeric value.")

        scalar = value_array.item()
        if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, (int, float, np.number)):
            raise ValueError(f"{name} must be a scalar numeric value.")

        return float(scalar)

    @staticmethod
    def _validate_scalar_boolean(value) -> bool:
        value_array = np.asarray(value)
        if value_array.shape != ():
            raise ValueError("done must be a scalar boolean value.")

        scalar = value_array.item()
        if isinstance(scalar, (bool, np.bool_)):
            return bool(scalar)
        if isinstance(scalar, (int, np.integer)) and scalar in (0, 1):
            return bool(scalar)

        raise ValueError("done must be a scalar boolean value.")
