"""Tests for the TD3 actor network."""

import unittest

import torch

from src.models.actor import ActorNetwork


class ActorNetworkTests(unittest.TestCase):
    def setUp(self):
        self.state_dim = 6
        self.action_dim = 5

    def test_constructor_rejects_non_positive_dimensions(self):
        with self.assertRaises(ValueError):
            ActorNetwork(0, self.action_dim)
        with self.assertRaises(ValueError):
            ActorNetwork(self.state_dim, 0)
        with self.assertRaises(ValueError):
            ActorNetwork(self.state_dim, self.action_dim, hidden_dim=0)
        with self.assertRaises(ValueError):
            ActorNetwork(self.state_dim, self.action_dim, softmax_temperature=0.0)

    def test_constructor_initializes_valid_actor(self):
        actor = ActorNetwork(self.state_dim, self.action_dim, softmax_temperature=0.5)

        self.assertEqual(actor.state_dim, self.state_dim)
        self.assertEqual(actor.action_dim, self.action_dim)
        self.assertEqual(actor.hidden_dim, 128)
        self.assertEqual(actor.softmax_temperature, 0.5)

    def test_forward_with_single_state_returns_action_shape(self):
        actor = ActorNetwork(self.state_dim, self.action_dim)
        state = torch.ones(self.state_dim)

        action = actor(state)

        self.assertEqual(tuple(action.shape), (self.action_dim,))

    def test_forward_with_batch_state_returns_batch_action_shape(self):
        actor = ActorNetwork(self.state_dim, self.action_dim)
        states = torch.ones(4, self.state_dim)

        actions = actor(states)

        self.assertEqual(tuple(actions.shape), (4, self.action_dim))

    def test_output_weights_are_non_negative(self):
        actor = ActorNetwork(self.state_dim, self.action_dim)
        state = torch.randn(self.state_dim)

        action = actor(state)

        self.assertTrue(torch.all(action >= 0.0))

    def test_output_weights_sum_to_one_for_single_state(self):
        actor = ActorNetwork(self.state_dim, self.action_dim)
        state = torch.randn(self.state_dim)

        action = actor(state)

        self.assertTrue(torch.allclose(action.sum(), torch.tensor(1.0), atol=1e-6))

    def test_output_weights_sum_to_one_for_each_batch_row(self):
        actor = ActorNetwork(self.state_dim, self.action_dim)
        states = torch.randn(4, self.state_dim)

        actions = actor(states)

        self.assertTrue(torch.allclose(actions.sum(dim=1), torch.ones(4), atol=1e-6))

    def test_output_does_not_contain_nan_or_inf(self):
        actor = ActorNetwork(self.state_dim, self.action_dim)
        states = torch.randn(4, self.state_dim)

        actions = actor(states)

        self.assertFalse(torch.isnan(actions).any())
        self.assertFalse(torch.isinf(actions).any())

    def test_initial_policy_is_not_extremely_concentrated(self):
        actor = ActorNetwork(self.state_dim, self.action_dim)
        states = torch.randn(16, self.state_dim)

        actions = actor(states)

        self.assertLess(float(actions.max()), 0.8)

    def test_forward_rejects_invalid_last_dimension(self):
        actor = ActorNetwork(self.state_dim, self.action_dim)
        state = torch.ones(self.state_dim + 1)

        with self.assertRaises(RuntimeError):
            actor(state)


if __name__ == "__main__":
    unittest.main()
