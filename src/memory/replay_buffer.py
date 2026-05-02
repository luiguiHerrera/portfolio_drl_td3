"""Replay buffer scaffold for off-policy TD3 training.

This module will store transitions collected from the portfolio environment and
provide mini-batches for TD3 updates. The transition schema should be finalized
after the state representation, reward components, and terminal conditions are
defined.
"""


class ReplayBuffer:
    """Experience storage interface for off-policy learning."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Replay buffer storage has not been implemented yet.")

    def add(self, transition):
        """Store one environment transition."""
        raise NotImplementedError("Replay buffer insertion has not been implemented yet.")

    def sample(self, batch_size):
        """Sample a mini-batch of transitions."""
        raise NotImplementedError("Replay buffer sampling has not been implemented yet.")
