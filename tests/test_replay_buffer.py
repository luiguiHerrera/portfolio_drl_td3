"""Tests for the NumPy replay buffer."""

import unittest

import numpy as np

from src.memory.replay_buffer import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def setUp(self):
        self.state_dim = 3
        self.action_dim = 2

    def test_constructor_initializes_empty_buffer(self):
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=5, seed=42)

        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.ptr, 0)
        self.assertEqual(buffer.states.shape, (5, self.state_dim))
        self.assertEqual(buffer.actions.shape, (5, self.action_dim))
        self.assertEqual(buffer.rewards.shape, (5, 1))
        self.assertEqual(buffer.next_states.shape, (5, self.state_dim))
        self.assertEqual(buffer.dones.shape, (5, 1))

    def test_add_increases_length(self):
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=5)

        buffer.add(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.4, 0.6]),
            1.0,
            np.array([2.0, 3.0, 4.0]),
            False,
        )

        self.assertEqual(len(buffer), 1)

    def test_circular_overwrite_works(self):
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=2)

        for value in range(3):
            buffer.add(
                np.full(self.state_dim, value),
                np.full(self.action_dim, value),
                value,
                np.full(self.state_dim, value + 1),
                False,
            )

        self.assertEqual(len(buffer), 2)
        self.assertEqual(buffer.ptr, 1)
        self.assertTrue((buffer.states[0] == np.full(self.state_dim, 2)).all())

    def test_sample_returns_all_expected_keys(self):
        buffer = self._filled_buffer(size=3)

        batch = buffer.sample(batch_size=2)

        self.assertEqual(
            set(batch.keys()),
            {"states", "actions", "rewards", "next_states", "dones"},
        )

    def test_sampled_arrays_have_correct_shapes(self):
        buffer = self._filled_buffer(size=4)

        batch = buffer.sample(batch_size=3)

        self.assertEqual(batch["states"].shape, (3, self.state_dim))
        self.assertEqual(batch["actions"].shape, (3, self.action_dim))
        self.assertEqual(batch["rewards"].shape, (3, 1))
        self.assertEqual(batch["next_states"].shape, (3, self.state_dim))
        self.assertEqual(batch["dones"].shape, (3, 1))

    def test_sample_rejects_batch_size_larger_than_current_size(self):
        buffer = self._filled_buffer(size=2)

        with self.assertRaises(ValueError):
            buffer.sample(batch_size=3)

    def test_sample_rejects_non_positive_batch_size(self):
        buffer = self._filled_buffer(size=2)

        with self.assertRaises(ValueError):
            buffer.sample(batch_size=0)

    def test_add_rejects_wrong_state_shape(self):
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=5)

        with self.assertRaises(ValueError):
            buffer.add(
                np.array([1.0, 2.0]),
                np.array([0.4, 0.6]),
                1.0,
                np.array([2.0, 3.0, 4.0]),
                False,
            )

    def test_add_rejects_wrong_action_shape(self):
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=5)

        with self.assertRaises(ValueError):
            buffer.add(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.4, 0.5, 0.1]),
                1.0,
                np.array([2.0, 3.0, 4.0]),
                False,
            )

    def test_add_rejects_wrong_next_state_shape(self):
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=5)

        with self.assertRaises(ValueError):
            buffer.add(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.4, 0.6]),
                1.0,
                np.array([2.0, 3.0]),
                False,
            )

    def test_constructor_rejects_non_positive_dimensions_or_max_size(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(0, self.action_dim, max_size=5)
        with self.assertRaises(ValueError):
            ReplayBuffer(self.state_dim, 0, max_size=5)
        with self.assertRaises(ValueError):
            ReplayBuffer(self.state_dim, self.action_dim, max_size=0)

    def test_add_rejects_non_scalar_reward(self):
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=5)

        with self.assertRaisesRegex(ValueError, "reward must be a scalar numeric value"):
            buffer.add(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.4, 0.6]),
                np.array([1.0, 2.0]),
                np.array([2.0, 3.0, 4.0]),
                False,
            )

    def test_add_rejects_non_numeric_reward(self):
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=5)

        with self.assertRaisesRegex(ValueError, "reward must be a scalar numeric value"):
            buffer.add(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.4, 0.6]),
                "bad",
                np.array([2.0, 3.0, 4.0]),
                False,
            )

    def test_add_rejects_non_scalar_done(self):
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=5)

        with self.assertRaisesRegex(ValueError, "done must be a scalar boolean value"):
            buffer.add(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.4, 0.6]),
                1.0,
                np.array([2.0, 3.0, 4.0]),
                [True, False],
            )

    def test_add_rejects_done_not_convertible_to_boolean(self):
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=5)

        with self.assertRaisesRegex(ValueError, "done must be a scalar boolean value"):
            buffer.add(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.4, 0.6]),
                1.0,
                np.array([2.0, 3.0, 4.0]),
                2,
            )

    def test_circular_behavior_with_max_size_three(self):
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=3)

        for value in range(5):
            buffer.add(
                np.full(self.state_dim, value),
                np.full(self.action_dim, value),
                value,
                np.full(self.state_dim, value + 1),
                False,
            )

        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.ptr, 2)

    def test_sampling_is_reproducible_with_same_seed(self):
        buffer_1 = self._filled_buffer(size=5, seed=123)
        buffer_2 = self._filled_buffer(size=5, seed=123)

        batch_1 = buffer_1.sample(batch_size=3)
        batch_2 = buffer_2.sample(batch_size=3)

        for key in ("states", "actions", "rewards", "next_states", "dones"):
            np.testing.assert_array_equal(batch_1[key], batch_2[key])

    def _filled_buffer_with_seed(self, size: int, seed: int | None) -> ReplayBuffer:
        buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=size, seed=seed)
        for value in range(size):
            buffer.add(
                np.full(self.state_dim, value),
                np.full(self.action_dim, value),
                float(value),
                np.full(self.state_dim, value + 1),
                value % 2 == 0,
            )

        return buffer

    def _filled_buffer(self, size: int, seed: int | None = None) -> ReplayBuffer:
        return self._filled_buffer_with_seed(size=size, seed=seed)


if __name__ == "__main__":
    unittest.main()
