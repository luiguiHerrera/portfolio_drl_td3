"""Tests for the TD3 agent core utilities."""

import unittest

import numpy as np
import torch

from src.models.td3_agent import TD3Agent


class TD3AgentTests(unittest.TestCase):
    def setUp(self):
        self.state_dim = 6
        self.action_dim = 5

    def test_constructor_initializes_all_networks(self):
        agent = self._agent()

        self.assertIsNotNone(agent.actor)
        self.assertIsNotNone(agent.actor_target)
        self.assertIsNotNone(agent.critic_1)
        self.assertIsNotNone(agent.critic_2)
        self.assertIsNotNone(agent.critic_1_target)
        self.assertIsNotNone(agent.critic_2_target)
        self.assertIsNotNone(agent.actor_optimizer)
        self.assertIsNotNone(agent.critic_1_optimizer)
        self.assertIsNotNone(agent.critic_2_optimizer)

    def test_target_policy_smoothing_config_is_separate_from_behavior_exploration(self):
        agent = self._agent(policy_noise=0.11, noise_clip=0.22)

        self.assertEqual(agent.policy_noise, 0.11)
        self.assertEqual(agent.noise_clip, 0.22)
        self.assertFalse(hasattr(agent, "exploration_noise"))

    def test_target_networks_initially_match_online_networks(self):
        agent = self._agent()

        self._assert_modules_match(agent.actor, agent.actor_target)
        self._assert_modules_match(agent.critic_1, agent.critic_1_target)
        self._assert_modules_match(agent.critic_2, agent.critic_2_target)

    def test_select_action_returns_action_shape(self):
        agent = self._agent()
        state = np.ones(self.state_dim, dtype=np.float32)

        action = agent.select_action(state)

        self.assertEqual(action.shape, (self.action_dim,))

    def test_select_action_is_non_negative_and_sums_to_one(self):
        agent = self._agent()
        state = np.random.randn(self.state_dim).astype(np.float32)

        action = agent.select_action(state)

        self.assertTrue((action >= 0.0).all())
        self.assertAlmostEqual(float(action.sum()), 1.0, places=6)

    def test_batch_to_tensors_returns_correct_shapes(self):
        agent = self._agent()
        tensors = agent._batch_to_tensors(self._batch(batch_size=4))

        self.assertEqual(tensors["states"].shape, (4, self.state_dim))
        self.assertEqual(tensors["actions"].shape, (4, self.action_dim))
        self.assertEqual(tensors["rewards"].shape, (4, 1))
        self.assertEqual(tensors["next_states"].shape, (4, self.state_dim))
        self.assertEqual(tensors["dones"].shape, (4, 1))
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in tensors.values()))

    def test_soft_update_changes_target_params_toward_source_params(self):
        agent = self._agent(tau=0.5)
        for parameter in agent.actor.parameters():
            parameter.data.fill_(1.0)
        for parameter in agent.actor_target.parameters():
            parameter.data.zero_()

        agent.soft_update(agent.actor, agent.actor_target)

        for parameter in agent.actor_target.parameters():
            self.assertTrue(torch.allclose(parameter, torch.full_like(parameter, 0.5)))

    def test_train_step_returns_expected_loss_keys(self):
        agent = self._agent()

        losses = agent.train_step(self._batch(batch_size=4))

        self.assertEqual(set(losses.keys()), {"critic_1_loss", "critic_2_loss", "actor_loss"})

    def test_train_step_increments_total_it(self):
        agent = self._agent()

        agent.train_step(self._batch(batch_size=4))

        self.assertEqual(agent.total_it, 1)

    def test_actor_loss_is_none_on_non_delayed_update(self):
        agent = self._agent(policy_delay=2)

        losses = agent.train_step(self._batch(batch_size=4))

        self.assertIsNone(losses["actor_loss"])

    def test_actor_loss_is_not_none_on_delayed_update(self):
        agent = self._agent(policy_delay=2)

        agent.train_step(self._batch(batch_size=4))
        losses = agent.train_step(self._batch(batch_size=4))

        self.assertIsNotNone(losses["actor_loss"])

    def test_simplex_projection_produces_valid_weights(self):
        agent = self._agent()
        actions = torch.tensor(
            [
                [0.2, -0.3, 0.5, 0.0, 0.1],
                [-1.0, -2.0, -3.0, -4.0, -5.0],
            ]
        )

        projected = agent._project_to_simplex_like_weights(actions)

        self.assertTrue(torch.all(projected >= 0.0))
        self.assertTrue(torch.allclose(projected.sum(dim=1), torch.ones(2)))

    def test_feasible_projection_respects_max_weight_cap(self):
        agent = self._agent(max_weight_cap=0.40)
        actions = torch.tensor([[0.90, 0.05, 0.05, 0.0, 0.0]])

        projected = agent._project_to_feasible_actions(actions)

        self.assertTrue(torch.all(projected >= 0.0))
        self.assertTrue(torch.all(projected <= 0.40 + 1e-6))
        self.assertTrue(torch.allclose(projected.sum(dim=1), torch.ones(1)))

    def test_target_policy_smoothing_respects_max_weight_cap(self):
        torch.manual_seed(7)
        agent = self._agent(max_weight_cap=0.40, policy_noise=0.50, noise_clip=0.50)
        next_states = torch.randn(8, self.state_dim)

        target_actions = agent._target_policy_actions(next_states)

        self.assertTrue(torch.all(target_actions >= 0.0))
        self.assertTrue(torch.all(target_actions <= 0.40 + 1e-6))
        self.assertTrue(torch.allclose(target_actions.sum(dim=1), torch.ones(8)))

    def test_actor_loss_actions_respect_max_weight_cap(self):
        agent = self._agent(max_weight_cap=0.40)
        states = torch.randn(8, self.state_dim)

        actor_actions = agent._actor_policy_actions(states)

        self.assertTrue(torch.all(actor_actions >= 0.0))
        self.assertTrue(torch.all(actor_actions <= 0.40 + 1e-6))
        self.assertTrue(torch.allclose(actor_actions.sum(dim=1), torch.ones(8)))

    def test_invalid_max_weight_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_weight_cap"):
            self._agent(max_weight_cap=0.10)

    def _agent(self, **kwargs) -> TD3Agent:
        return TD3Agent(
            self.state_dim,
            self.action_dim,
            hidden_dim=16,
            device="cpu",
            **kwargs,
        )

    def _batch(self, batch_size: int) -> dict:
        actions = np.random.rand(batch_size, self.action_dim).astype(np.float32)
        actions = actions / actions.sum(axis=1, keepdims=True)

        return {
            "states": np.random.randn(batch_size, self.state_dim).astype(np.float32),
            "actions": actions,
            "rewards": np.random.randn(batch_size, 1).astype(np.float32),
            "next_states": np.random.randn(batch_size, self.state_dim).astype(np.float32),
            "dones": np.zeros((batch_size, 1), dtype=bool),
        }

    def _assert_modules_match(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.allclose(source_param, target_param))


if __name__ == "__main__":
    unittest.main()
