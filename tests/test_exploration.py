import unittest

import numpy as np

from src.train.exploration import (
    apply_behavior_exploration_noise,
    project_to_capped_simplex,
    project_to_simplex,
)


class ExplorationTests(unittest.TestCase):
    def test_zero_noise_preserves_normalized_action(self):
        action = np.array([0.4, 0.3, 0.2, 0.1])

        projected = apply_behavior_exploration_noise(action, noise_std=0.0)

        np.testing.assert_allclose(projected, action)

    def test_positive_noise_changes_action_under_fixed_seed(self):
        action = np.array([0.4, 0.3, 0.2, 0.1])
        rng = np.random.default_rng(7)

        projected = apply_behavior_exploration_noise(action, noise_std=0.05, rng=rng)

        self.assertFalse(np.allclose(projected, action))
        self.assertAlmostEqual(projected.sum(), 1.0)
        self.assertTrue((projected >= 0.0).all())

    def test_noisy_action_respects_max_weight_cap(self):
        action = np.array([0.9, 0.1, 0.0, 0.0])
        rng = np.random.default_rng(7)

        projected = apply_behavior_exploration_noise(
            action,
            noise_std=0.20,
            rng=rng,
            max_weight=0.40,
        )

        self.assertAlmostEqual(projected.sum(), 1.0)
        self.assertTrue((projected >= 0.0).all())
        self.assertLessEqual(float(projected.max()), 0.40 + 1e-12)

    def test_project_to_simplex_handles_all_negative_action(self):
        projected = project_to_simplex(np.array([-1.0, -2.0, -3.0]))

        np.testing.assert_allclose(projected, np.array([1 / 3, 1 / 3, 1 / 3]))

    def test_project_to_capped_simplex_rejects_infeasible_cap(self):
        with self.assertRaisesRegex(ValueError, "infeasible"):
            project_to_capped_simplex(np.array([0.5, 0.5, 0.0]), max_weight=0.20)

    def test_negative_exploration_noise_rejected(self):
        with self.assertRaisesRegex(ValueError, "noise_std"):
            apply_behavior_exploration_noise(np.array([0.5, 0.5]), noise_std=-0.01)


if __name__ == "__main__":
    unittest.main()
