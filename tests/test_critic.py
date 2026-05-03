"""Tests for the TD3 critic network."""

import unittest

import torch

from src.models.critic import CriticNetwork


class CriticNetworkTests(unittest.TestCase):
    def setUp(self):
        self.state_dim = 6
        self.action_dim = 5

    def test_constructor_rejects_non_positive_dimensions(self):
        with self.assertRaises(ValueError):
            CriticNetwork(0, self.action_dim)
        with self.assertRaises(ValueError):
            CriticNetwork(self.state_dim, 0)
        with self.assertRaises(ValueError):
            CriticNetwork(self.state_dim, self.action_dim, hidden_dim=0)

    def test_forward_with_single_state_action_returns_scalar_shape(self):
        critic = CriticNetwork(self.state_dim, self.action_dim)
        state = torch.ones(self.state_dim)
        action = torch.ones(self.action_dim)

        q_value = critic(state, action)

        self.assertEqual(tuple(q_value.shape), (1,))

    def test_forward_with_batch_state_action_returns_batch_shape(self):
        critic = CriticNetwork(self.state_dim, self.action_dim)
        states = torch.ones(4, self.state_dim)
        actions = torch.ones(4, self.action_dim)

        q_values = critic(states, actions)

        self.assertEqual(tuple(q_values.shape), (4, 1))

    def test_forward_rejects_mismatched_batch_sizes(self):
        critic = CriticNetwork(self.state_dim, self.action_dim)
        states = torch.ones(4, self.state_dim)
        actions = torch.ones(3, self.action_dim)

        with self.assertRaises(RuntimeError):
            critic(states, actions)

    def test_forward_rejects_invalid_state_last_dimension(self):
        critic = CriticNetwork(self.state_dim, self.action_dim)
        state = torch.ones(self.state_dim + 1)
        action = torch.ones(self.action_dim)

        with self.assertRaises(RuntimeError):
            critic(state, action)

    def test_forward_rejects_invalid_action_last_dimension(self):
        critic = CriticNetwork(self.state_dim, self.action_dim)
        state = torch.ones(self.state_dim)
        action = torch.ones(self.action_dim + 1)

        with self.assertRaises(RuntimeError):
            critic(state, action)

    def test_output_is_torch_tensor(self):
        critic = CriticNetwork(self.state_dim, self.action_dim)
        state = torch.ones(self.state_dim)
        action = torch.ones(self.action_dim)

        q_value = critic(state, action)

        self.assertIsInstance(q_value, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
