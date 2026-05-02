"""Tests for the minimal portfolio environment."""

import unittest

import numpy as np
import pandas as pd

from src.env.portfolio_env import PortfolioEnv


class PortfolioEnvTests(unittest.TestCase):
    def setUp(self):
        self.returns = pd.DataFrame(
            {
                "SPY": [0.01, 0.02, -0.01],
                "TLT": [0.00, 0.01, 0.01],
                "GLD": [0.02, -0.01, 0.00],
                "BTC-USD": [0.03, -0.02, 0.04],
                "CASH": [0.00, 0.00, 0.00],
            }
        )

    def test_reset_returns_observation_with_correct_shape(self):
        env = PortfolioEnv(self.returns)

        observation = env.reset()

        self.assertEqual(observation.shape, (5,))

    def test_action_weights_are_normalized_and_non_negative(self):
        env = PortfolioEnv(self.returns)
        env.reset()

        _, _, _, info = env.step(np.array([1.0, -1.0, 2.0, 0.0, 1.0]))

        self.assertTrue((info["weights"] >= 0.0).all())
        self.assertAlmostEqual(info["weights"].sum(), 1.0)

    def test_zero_action_falls_back_to_equal_weights(self):
        env = PortfolioEnv(self.returns)
        env.reset()

        _, _, _, info = env.step(np.zeros(5))

        np.testing.assert_allclose(info["weights"], np.full(5, 0.2))

    def test_portfolio_value_updates_after_one_step(self):
        env = PortfolioEnv(self.returns, initial_cash=100000.0, transaction_cost=0.001)
        env.reset()

        _, reward, _, info = env.step(np.full(5, 1.0))

        expected_return = self.returns.iloc[0].mean()
        expected_value = 100000.0 * (1.0 + expected_return)
        self.assertAlmostEqual(reward, expected_return)
        self.assertAlmostEqual(info["portfolio_value"], expected_value)

    def test_done_becomes_true_after_consuming_all_rows(self):
        env = PortfolioEnv(self.returns)
        env.reset()

        done = False
        for _ in range(len(self.returns)):
            _, _, done, _ = env.step(np.full(5, 1.0))

        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
