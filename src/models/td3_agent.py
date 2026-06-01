"""TD3 agent core for portfolio allocation.

This module coordinates the actor, twin critics, target networks, optimizers,
action selection, and a single sampled-batch TD3 update. It intentionally does
not implement environment interaction, replay-buffer sampling, or a training
loop.
"""

import numpy as np
import torch
from torch import nn

from src.models.actor import ActorNetwork
from src.models.critic import CriticNetwork
from src.utils.action_projection import project_torch_portfolio_actions


class TD3Agent:
    """Core TD3 components and one-step update logic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        device: str | None = None,
        max_weight_cap: float | None = None,
    ):
        self._validate_init_args(
            state_dim,
            action_dim,
            hidden_dim,
            actor_learning_rate,
            critic_learning_rate,
            gamma,
            tau,
            policy_noise,
            noise_clip,
            policy_delay,
            max_weight_cap,
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_weight_cap = None if max_weight_cap is None else float(max_weight_cap)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.total_it = 0

        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.hard_update(self.actor, self.actor_target)
        self.hard_update(self.critic_1, self.critic_1_target)
        self.hard_update(self.critic_2, self.critic_2_target)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_learning_rate,
        )
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(),
            lr=critic_learning_rate,
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(),
            lr=critic_learning_rate,
        )
        self.loss_fn = nn.MSELoss()

    def select_action(self, state) -> np.ndarray:
        """Return portfolio weights for a single NumPy state observation."""
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},).")

        was_training = self.actor.training
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state_array, dtype=torch.float32, device=self.device)
            action = self.actor(state_tensor).cpu().numpy()
        if was_training:
            self.actor.train()

        return action

    def soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Move target parameters toward source parameters using tau."""
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.mul_(1.0 - self.tau)
                target_param.add_(self.tau * source_param)

    def hard_update(self, source: nn.Module, target: nn.Module) -> None:
        """Copy source parameters exactly into target parameters."""
        target.load_state_dict(source.state_dict())

    def train_step(self, batch: dict) -> dict:
        """Run one TD3 update from an already sampled replay batch."""
        tensors = self._batch_to_tensors(batch)
        states = tensors["states"]
        actions = tensors["actions"]
        rewards = tensors["rewards"]
        next_states = tensors["next_states"]
        dones = tensors["dones"]

        self.total_it += 1

        with torch.no_grad():
            target_actions = self._target_policy_actions(next_states)

            target_q1 = self.critic_1_target(next_states, target_actions)
            target_q2 = self.critic_2_target(next_states, target_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1.0 - dones) * target_q

        current_q1 = self.critic_1(states, actions)
        critic_1_loss = self.loss_fn(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        current_q2 = self.critic_2(states, actions)
        critic_2_loss = self.loss_fn(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        actor_loss_value = None
        if self.total_it % self.policy_delay == 0:
            actor_actions = self._actor_policy_actions(states)
            actor_loss = -self.critic_1(states, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic_1, self.critic_1_target)
            self.soft_update(self.critic_2, self.critic_2_target)
            actor_loss_value = float(actor_loss.detach().cpu())

        return {
            "critic_1_loss": float(critic_1_loss.detach().cpu()),
            "critic_2_loss": float(critic_2_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
        }

    def _batch_to_tensors(self, batch: dict) -> dict:
        """Convert a NumPy replay-buffer batch to torch tensors on the agent device."""
        required_keys = ("states", "actions", "rewards", "next_states", "dones")
        missing_keys = [key for key in required_keys if key not in batch]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise KeyError(f"batch is missing required keys: {missing}")

        return {
            "states": torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device),
            "rewards": torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device),
            "next_states": torch.as_tensor(
                batch["next_states"],
                dtype=torch.float32,
                device=self.device,
            ),
            "dones": torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device),
        }

    def _project_to_simplex_like_weights(self, actions: torch.Tensor) -> torch.Tensor:
        """Clamp actions non-negative and normalize each row to sum to one."""
        return project_torch_portfolio_actions(actions)

    def _project_to_feasible_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Project actions to the simplex and active max-weight cap, if any."""
        return project_torch_portfolio_actions(actions, max_weight=self.max_weight_cap)

    def _target_policy_actions(self, next_states: torch.Tensor) -> torch.Tensor:
        """Return target actions after TD3 smoothing and feasible-set projection."""
        target_actions = self.actor_target(next_states)
        noise = torch.randn_like(target_actions) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        return self._project_to_feasible_actions(target_actions + noise)

    def _actor_policy_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Return actor actions projected to the feasible action set."""
        return self._project_to_feasible_actions(self.actor(states))

    @staticmethod
    def _validate_init_args(
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        gamma: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_delay: int,
        max_weight_cap: float | None,
    ) -> None:
        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive.")
        if critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive.")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in the interval (0, 1].")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in the interval (0, 1].")
        if policy_noise < 0.0:
            raise ValueError("policy_noise must be non-negative.")
        if noise_clip < 0.0:
            raise ValueError("noise_clip must be non-negative.")
        if policy_delay <= 0:
            raise ValueError("policy_delay must be positive.")
        if max_weight_cap is not None:
            cap = float(max_weight_cap)
            if cap <= 0.0 or cap > 1.0:
                raise ValueError("max_weight_cap must be in (0, 1].")
            if cap * action_dim < 1.0:
                raise ValueError("max_weight_cap is infeasible for the action dimension.")
