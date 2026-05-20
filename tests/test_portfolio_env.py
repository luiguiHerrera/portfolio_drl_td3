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

    def test_observation_shape_unchanged_when_auxiliary_features_are_provided(self):
        auxiliary_features = self._auxiliary_features([1.0, 0.0, 1.0])
        env = PortfolioEnv(
            self.returns,
            features=self.features,
            auxiliary_features=auxiliary_features,
        )

        observation = env.reset()

        self.assertEqual(observation.shape, (3,))
        self.assertEqual(env.observation_dim, len(self.features.columns))
        self.assertEqual(list(env.auxiliary_features.columns), ["risk_off_state"])

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

    def test_default_reward_config_preserves_net_return_behavior(self):
        env = PortfolioEnv(self.returns, transaction_cost=0.01)
        env.reset()

        _, reward, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertAlmostEqual(
            reward,
            info["portfolio_return"] - info["transaction_cost"],
        )

    def test_reward_unchanged_when_mandate_penalty_missing_or_false(self):
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        baseline_env = PortfolioEnv(self.returns, transaction_cost=0.0)
        disabled_env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_mandate_penalty": False,
                "lambda_mandate": 100.0,
            },
        )
        baseline_env.reset()
        disabled_env.reset()

        _, baseline_reward, _, baseline_info = baseline_env.step(action)
        _, disabled_reward, _, disabled_info = disabled_env.step(action)

        self.assertAlmostEqual(disabled_reward, baseline_reward)
        self.assertNotIn("mandate_penalty", baseline_info)
        self.assertNotIn("mandate_penalty", disabled_info)

    def test_reward_unchanged_when_cash_risk_off_penalty_missing_or_false(self):
        action = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        baseline_env = PortfolioEnv(self.returns, transaction_cost=0.0)
        disabled_env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_cash_risk_off_penalty": False,
                "normal_cash_max": 0.10,
                "cash_penalty_weight": 100.0,
            },
        )
        baseline_env.reset()
        disabled_env.reset()

        _, baseline_reward, _, baseline_info = baseline_env.step(action)
        _, disabled_reward, _, disabled_info = disabled_env.step(action)

        self.assertAlmostEqual(disabled_reward, baseline_reward)
        self.assertNotIn("cash_penalty", baseline_info)
        self.assertNotIn("cash_penalty", disabled_info)

    def test_cash_penalty_enabled_reduces_reward_above_normal_cash_outside_risk_off(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_cash_risk_off_penalty": True,
                "normal_cash_max": 0.10,
                "cash_penalty_weight": 1.0,
                "cash_risk_off_state": False,
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))

        self.assertAlmostEqual(info["cash_breach"], 0.90)
        self.assertAlmostEqual(info["cash_penalty"], 0.90)
        self.assertAlmostEqual(reward, info["portfolio_return"] - 0.90)

    def test_cash_penalty_does_not_penalize_high_cash_when_risk_off_state_true(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_cash_risk_off_penalty": True,
                "normal_cash_max": 0.10,
                "cash_penalty_weight": 1.0,
                "cash_risk_off_state": True,
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))

        self.assertEqual(info["cash_breach"], 0.0)
        self.assertEqual(info["cash_penalty"], 0.0)
        self.assertAlmostEqual(reward, info["portfolio_return"])

    def test_cash_penalty_uses_static_risk_off_state_without_dynamic_column(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            auxiliary_features=self._auxiliary_features([0.0, 0.0, 0.0]),
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_cash_risk_off_penalty": True,
                "normal_cash_max": 0.10,
                "cash_penalty_weight": 1.0,
                "cash_risk_off_state": True,
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))

        self.assertEqual(info["cash_penalty"], 0.0)
        self.assertNotIn("cash_risk_off_column", info)
        self.assertAlmostEqual(reward, info["portfolio_return"])

    def test_cash_penalty_uses_dynamic_auxiliary_risk_off_state(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            auxiliary_features=self._auxiliary_features([1.0, 0.0, 0.0]),
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_cash_risk_off_penalty": True,
                "normal_cash_max": 0.10,
                "cash_penalty_weight": 1.0,
                "cash_risk_off_column": "risk_off_state",
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))

        self.assertEqual(info["cash_breach"], 0.0)
        self.assertEqual(info["cash_penalty"], 0.0)
        self.assertTrue(info["cash_risk_off_state"])
        self.assertEqual(info["cash_risk_off_column"], "risk_off_state")
        self.assertAlmostEqual(reward, info["portfolio_return"])

    def test_cash_penalty_penalizes_high_cash_outside_dynamic_risk_off(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            auxiliary_features=self._auxiliary_features([0.0, 1.0, 1.0]),
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_cash_risk_off_penalty": True,
                "normal_cash_max": 0.10,
                "cash_penalty_weight": 1.0,
                "cash_risk_off_column": "risk_off_state",
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))

        self.assertAlmostEqual(info["cash_breach"], 0.90)
        self.assertAlmostEqual(info["cash_penalty"], 0.90)
        self.assertFalse(info["cash_risk_off_state"])
        self.assertAlmostEqual(reward, info["portfolio_return"] - 0.90)

    def test_cash_risk_off_column_requires_auxiliary_features(self):
        with self.assertRaisesRegex(ValueError, "auxiliary_features are required"):
            PortfolioEnv(
                self.returns,
                reward_config={
                    "use_cash_risk_off_penalty": True,
                    "cash_risk_off_column": "risk_off_state",
                },
            )

    def test_cash_risk_off_column_requires_existing_auxiliary_column(self):
        auxiliary_features = self._auxiliary_features([0.0, 1.0, 0.0])

        with self.assertRaisesRegex(ValueError, "cash risk-off column"):
            PortfolioEnv(
                self.returns,
                auxiliary_features=auxiliary_features,
                reward_config={
                    "use_cash_risk_off_penalty": True,
                    "cash_risk_off_column": "missing_state",
                },
            )

    def test_cash_risk_off_column_rejects_nan_values(self):
        auxiliary_features = self._auxiliary_features([0.0, np.nan, 1.0])

        with self.assertRaisesRegex(ValueError, "contains NaN"):
            PortfolioEnv(
                self.returns,
                auxiliary_features=auxiliary_features,
                reward_config={
                    "use_cash_risk_off_penalty": True,
                    "cash_risk_off_column": "risk_off_state",
                },
            )

    def test_auxiliary_features_are_aligned_by_matching_dates(self):
        auxiliary_features = self._auxiliary_features([0.0, 1.0, 0.0])
        auxiliary_features = auxiliary_features.iloc[[2, 0, 1]]

        env = PortfolioEnv(
            self.returns,
            features=self.features,
            auxiliary_features=auxiliary_features,
        )

        self.assertTrue(env.returns.index.equals(self.returns.index))
        self.assertTrue(env.features.index.equals(self.returns.index))
        self.assertTrue(env.auxiliary_features.index.equals(self.returns.index))
        self.assertEqual(env.auxiliary_features.iloc[1]["risk_off_state"], 1.0)

    def test_info_includes_cash_penalty_fields_when_enabled(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_cash_risk_off_penalty": True,
                "normal_cash_max": 0.10,
                "cash_penalty_weight": 1.0,
                "cash_risk_off_state": False,
            },
        )
        env.reset()

        _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))

        for key in (
            "cash_penalty",
            "cash_breach",
            "normal_cash_max",
            "cash_risk_off_state",
        ):
            self.assertIn(key, info)

    def test_custom_reward_config_turnover_penalty_reduces_reward(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "lambda_turnover": 0.5,
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertAlmostEqual(reward, info["portfolio_return"] - 0.5 * info["turnover"])

    def test_turnover_penalty_none_keeps_transaction_cost_active(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.01,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 1.0,
                "lambda_turnover": 100.0,
                "turnover_penalty_mode": "none",
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertEqual(info["turnover_penalty"], 0.0)
        self.assertAlmostEqual(
            reward,
            info["portfolio_return"] - info["transaction_cost"],
        )

    def test_excess_linear_turnover_penalty_ignores_turnover_below_free_band(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "lambda_turnover": 0.5,
                "turnover_penalty_mode": "excess_linear",
                "turnover_free_band": 0.10,
            },
        )
        env.reset()

        _, reward, _, info = env.step(
            np.array([0.22, 0.195, 0.195, 0.195, 0.195])
        )

        self.assertLess(info["turnover"], 0.10)
        self.assertEqual(info["turnover_excess"], 0.0)
        self.assertEqual(info["turnover_penalty"], 0.0)
        self.assertAlmostEqual(reward, info["portfolio_return"])

    def test_excess_linear_turnover_penalty_uses_only_excess_above_free_band(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "lambda_turnover": 0.5,
                "turnover_penalty_mode": "excess_linear",
                "turnover_free_band": 0.10,
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

        expected_excess = info["turnover"] - 0.10
        expected_penalty = 0.5 * expected_excess
        self.assertAlmostEqual(info["turnover_excess"], expected_excess)
        self.assertAlmostEqual(info["turnover_penalty"], expected_penalty)
        self.assertAlmostEqual(reward, info["portfolio_return"] - expected_penalty)

    def test_excess_quadratic_turnover_penalty_adds_quadratic_excess(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "lambda_turnover": 0.5,
                "turnover_penalty_mode": "excess_quadratic",
                "turnover_free_band": 0.10,
                "turnover_quadratic_weight": 0.2,
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

        expected_excess = info["turnover"] - 0.10
        expected_penalty = 0.5 * expected_excess + 0.2 * expected_excess ** 2
        self.assertAlmostEqual(info["turnover_penalty"], expected_penalty)
        self.assertAlmostEqual(reward, info["portfolio_return"] - expected_penalty)

    def test_info_includes_turnover_penalty_fields(self):
        env = PortfolioEnv(self.returns)
        env.reset()

        _, _, _, info = env.step(np.full(5, 1.0))

        for key in (
            "turnover_penalty",
            "turnover_penalty_mode",
            "turnover_free_band",
            "turnover_excess",
        ):
            self.assertIn(key, info)

    def test_custom_reward_config_concentration_penalty_reduces_reward(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "lambda_concentration": 0.1,
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertAlmostEqual(
            reward,
            info["portfolio_return"] - 0.1 * info["concentration"],
        )

    def test_mandate_penalty_enabled_reduces_reward_when_breaches_exist(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_mandate_penalty": True,
                "lambda_mandate": 1.0,
                "mandate_profile": "moderate",
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([0.95, 0.05, 0.0, 0.0, 0.0]))

        self.assertGreater(info["mandate_penalty"], 0.0)
        self.assertLess(reward, info["portfolio_return"])

    def test_aggressive_profile_does_not_penalize_all_in_max_weight(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_mandate_penalty": True,
                "lambda_mandate": 1.0,
                "mandate_profile": "aggressive",
            },
        )
        env.reset()

        _, _, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertEqual(info["mandate_max_weight_breach"], 0.0)

    def test_moderate_profile_penalizes_high_max_weight(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_mandate_penalty": True,
                "lambda_mandate": 1.0,
                "mandate_profile": "moderate",
            },
        )
        env.reset()

        _, _, _, info = env.step(np.array([0.95, 0.05, 0.0, 0.0, 0.0]))

        self.assertAlmostEqual(info["mandate_max_weight_breach"], 0.15)

    def test_volatility_penalty_uses_only_observed_returns(self):
        returns = pd.DataFrame(
            {
                "SPY": [0.01, 0.01, -0.50],
                "TLT": [0.00, 0.00, 0.00],
                "GLD": [0.00, 0.00, 0.00],
                "BTC-USD": [0.00, 0.00, 0.00],
                "CASH": [0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
        )
        env = PortfolioEnv(
            returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_mandate_penalty": True,
                "lambda_mandate": 1.0,
                "mandate_profile": "moderate",
                "mandate_penalty_weights": {
                    "drawdown_breach": 0.0,
                    "volatility_breach": 1.0,
                    "max_weight_breach": 0.0,
                    "effective_assets_breach": 0.0,
                    "turnover_breach": 0.0,
                },
                "mandate_volatility_window": 2,
            },
        )
        env.reset()

        _, _, _, first_info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        _, _, _, second_info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        _, _, _, third_info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertEqual(first_info["mandate_volatility_breach"], 0.0)
        self.assertEqual(second_info["mandate_volatility_breach"], 0.0)
        self.assertGreater(third_info["mandate_volatility_breach"], 0.0)

    def test_info_includes_mandate_debug_fields_when_enabled(self):
        env = PortfolioEnv(
            self.returns,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "use_mandate_penalty": True,
                "lambda_mandate": 1.0,
                "mandate_profile": "moderate",
            },
        )
        env.reset()

        _, _, _, info = env.step(np.array([0.95, 0.05, 0.0, 0.0, 0.0]))

        for key in [
            "mandate_penalty",
            "mandate_drawdown_breach",
            "mandate_volatility_breach",
            "mandate_max_weight_breach",
            "mandate_effective_assets_breach",
            "mandate_turnover_breach",
        ]:
            self.assertIn(key, info)

    def test_info_includes_risk_aware_reward_fields(self):
        env = PortfolioEnv(self.returns)
        env.reset()

        _, reward, _, info = env.step(np.full(5, 1.0))

        self.assertIn("financial_net_return", info)
        self.assertIn("drawdown", info)
        self.assertIn("concentration", info)
        self.assertIn("reward", info)
        self.assertIn("peak_portfolio_value", info)
        self.assertAlmostEqual(info["reward"], reward)

    def test_portfolio_value_updates_with_financial_net_return_not_risk_reward(self):
        returns = pd.DataFrame(
            {
                "SPY": [0.10],
                "TLT": [0.00],
                "GLD": [0.00],
                "BTC-USD": [0.00],
                "CASH": [0.00],
            },
            index=pd.date_range("2024-01-05", periods=1, freq="W-FRI"),
        )
        env = PortfolioEnv(
            returns,
            initial_cash=100000.0,
            transaction_cost=0.0,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "lambda_concentration": 0.2,
            },
        )
        env.reset()

        _, reward, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertAlmostEqual(info["financial_net_return"], 0.10)
        self.assertAlmostEqual(reward, -0.10)
        self.assertAlmostEqual(info["portfolio_value"], 110000.0)

    def test_peak_portfolio_value_updates_after_gains(self):
        env = PortfolioEnv(self.returns, initial_cash=100000.0, transaction_cost=0.0)
        env.reset()

        _, _, _, info = env.step(np.full(5, 1.0))

        self.assertGreater(info["peak_portfolio_value"], 100000.0)
        self.assertAlmostEqual(info["peak_portfolio_value"], info["portfolio_value"])

    def test_drawdown_is_based_on_financial_portfolio_value(self):
        returns = pd.DataFrame(
            {
                "SPY": [0.10, -0.20, 0.00],
                "TLT": [0.00, 0.00, 0.00],
                "GLD": [0.00, 0.00, 0.00],
                "BTC-USD": [0.00, 0.00, 0.00],
                "CASH": [0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
        )
        env = PortfolioEnv(returns, initial_cash=100000.0, transaction_cost=0.0)
        env.reset()

        env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        _, _, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertAlmostEqual(info["portfolio_value"], 88000.0)
        self.assertAlmostEqual(info["peak_portfolio_value"], 110000.0)
        self.assertAlmostEqual(info["drawdown"], 0.2)

    def test_step_portfolio_return_depends_on_current_action(self):
        returns = pd.DataFrame(
            {
                "SPY": [0.10],
                "TLT": [0.00],
                "GLD": [0.00],
                "BTC-USD": [0.00],
                "CASH": [0.00],
            },
            index=pd.date_range("2024-01-05", periods=1, freq="W-FRI"),
        )
        env = PortfolioEnv(returns, transaction_cost=0.0)
        env.reset()

        _, reward, _, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

        self.assertAlmostEqual(info["portfolio_return"], 0.10)
        self.assertAlmostEqual(reward, 0.10)
        self.assertNotAlmostEqual(info["portfolio_return"], returns.iloc[0].mean())

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

    def _auxiliary_features(self, values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {"risk_off_state": values},
            index=self.returns.index,
        )


if __name__ == "__main__":
    unittest.main()
