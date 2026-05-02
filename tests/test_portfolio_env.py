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
            },
            index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
        )
        self.features = pd.DataFrame(
            {
                "SPY_ret_1w": [0.10, 0.20, 0.30],
                "SPY_vol_4w": [0.01, 0.02, 0.03],
                "CASH_ret_1w": [0.00, 0.00, 0.00],
            },
            index=self.returns.index,
        )

    def test_reset_returns_observation_with_correct_shape(self):
        env = PortfolioEnv(self.returns)

        observation = env.reset()

        self.assertEqual(observation.shape, (5,))

    def test_reset_returns_feature_observation_when_features_are_provided(self):
        env = PortfolioEnv(self.returns, features=self.features)

        observation = env.reset()

        self.assertEqual(observation.shape, (3,))
        np.testing.assert_allclose(observation, self.features.iloc[0].to_numpy())

    def test_returns_and_features_are_aligned_by_shared_index(self):
        extra_index = pd.date_range("2023-12-29", periods=5, freq="W-FRI")
        features = pd.DataFrame(
            {
                "SPY_ret_1w": [9.0, 0.10, 0.20, 0.30, 8.0],
                "SPY_vol_4w": [9.0, 0.01, 0.02, 0.03, 8.0],
            },
            index=extra_index,
        )

        env = PortfolioEnv(self.returns, features=features)

        self.assertTrue(env.returns.index.equals(self.returns.index))
        self.assertTrue(env.features.index.equals(self.returns.index))
        np.testing.assert_allclose(env.reset(), features.loc[self.returns.index[0]].to_numpy())

    def test_alignment_preserves_returns_order_when_features_index_is_unordered(self):
        unordered_features = self.features.iloc[[2, 0, 1]]

        env = PortfolioEnv(self.returns, features=unordered_features)

        self.assertTrue(env.returns.index.equals(self.returns.index))
        self.assertTrue(env.features.index.equals(self.returns.index))

    def test_step_uses_returns_not_features_for_portfolio_return(self):
        env = PortfolioEnv(self.returns, features=self.features, transaction_cost=0.0)
        env.reset()

        _, reward, _, info = env.step(np.full(5, 1.0))

        expected_return = self.returns.iloc[0].mean()
        self.assertAlmostEqual(info["portfolio_return"], expected_return)
        self.assertAlmostEqual(reward, expected_return)

    def test_done_observation_uses_feature_dimension(self):
        env = PortfolioEnv(self.returns, features=self.features)
        env.reset()

        done = False
        observation = None
        for _ in range(len(self.returns)):
            observation, _, done, _ = env.step(np.full(5, 1.0))

        self.assertTrue(done)
        self.assertEqual(observation.shape, (3,))

    def test_no_shared_index_between_returns_and_features_raises_value_error(self):
        features = self.features.copy()
        features.index = pd.date_range("2030-01-04", periods=3, freq="W-FRI")

        with self.assertRaises(ValueError):
            PortfolioEnv(self.returns, features=features)

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

    def test_step_after_done_raises_runtime_error(self):
        env = PortfolioEnv(self.returns)
        env.reset()

        for _ in range(len(self.returns)):
            env.step(np.full(5, 1.0))

        with self.assertRaises(RuntimeError):
            env.step(np.full(5, 1.0))


if __name__ == "__main__":
    unittest.main()
