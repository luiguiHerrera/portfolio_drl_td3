"""TD3 agent scaffold for portfolio allocation.

This module will coordinate the actor, twin critics, target networks,
exploration noise, replay buffer interaction, and update schedule. It remains a
non-functional contract until the environment, reward, and model interfaces are
validated.
"""


class TD3Agent:
    """Training and inference coordinator for the TD3 components."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TD3 agent initialization has not been implemented yet.")

    def select_action(self, state, explore=False):
        """Return portfolio weights for a given state."""
        raise NotImplementedError("TD3 action selection has not been implemented yet.")

    def update(self, replay_buffer):
        """Update actor and critics from replay buffer samples."""
        raise NotImplementedError("TD3 update logic has not been implemented yet.")
