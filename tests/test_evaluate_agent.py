"""Tests for TD3 agent evaluation utilities."""

import unittest

import numpy as np
import pandas as pd

from src.backtest.evaluate_agent import (
    build_policy_history,
    evaluate_agent,
    run_policy_episode,
    summarize_episode_diagnostics,
)
from src.backtest.evaluate_policy import cumulative_return


class DummyAgent:
    def __init__(self):
        self.calls = 0

    def select_action(self, state):
        self.calls += 1
        if self.calls == 1:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        return np.array([0.0, 1.0, 0.0, 0.0, 0.0])


class EvaluateAgentTests(unittest.TestCase):
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
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [0.1, 0.2, 0.3],
            },
            index=self.returns.index,
        )

    def test_run_policy_episode_returns_expected_keys(self):
        episode = run_policy_episode(DummyAgent(), self.returns, self.features)
        expected_keys = {
            "policy_returns",
            "financial_net_returns",
            "rewards",
            "portfolio_values",
            "turnover",
            "transaction_costs",
            "drawdown",
            "concentration",
            "weights",
            "final_portfolio_value",
            "turnover_reward_info",
        }

        self.assertEqual(set(episode.keys()), expected_keys)

    def test_policy_returns_length_matches_aligned_env_returns_length(self):
        episode = run_policy_episode(DummyAgent(), self.returns, self.features)

        self.assertEqual(len(episode["policy_returns"]), len(self.returns))
        self.assertEqual(len(episode["financial_net_returns"]), len(self.returns))

    def test_weights_dataframe_has_asset_columns(self):
        episode = run_policy_episode(DummyAgent(), self.returns, self.features)

        self.assertEqual(list(episode["weights"].columns), list(self.returns.columns))

    def test_episode_contains_drawdown_and_concentration_series(self):
        episode = run_policy_episode(DummyAgent(), self.returns, self.features)

        self.assertIsInstance(episode["financial_net_returns"], pd.Series)
        self.assertIsInstance(episode["drawdown"], pd.Series)
        self.assertIsInstance(episode["concentration"], pd.Series)
        self.assertEqual(len(episode["drawdown"]), len(self.returns))
        self.assertEqual(len(episode["concentration"]), len(self.returns))

    def test_policy_returns_differ_from_rewards_when_transaction_costs_apply(self):
        episode = run_policy_episode(
            DummyAgent(),
            self.returns,
            self.features,
            transaction_cost=0.01,
        )

        self.assertFalse(episode["policy_returns"].equals(episode["rewards"]))

    def test_evaluate_agent_returns_episode_metrics_and_diagnostics(self):
        result = evaluate_agent(DummyAgent(), self.returns, self.features)

        self.assertEqual(
            set(result.keys()),
            {"episode", "metrics", "diagnostics", "policy_history"},
        )

    def test_evaluate_agent_accepts_reward_config(self):
        result = evaluate_agent(
            DummyAgent(),
            self.returns,
            self.features,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 1.0,
                "lambda_turnover": 0.1,
                "lambda_concentration": 0.1,
                "lambda_drawdown": 0.1,
            },
        )

        self.assertEqual(
            set(result.keys()),
            {"episode", "metrics", "diagnostics", "policy_history"},
        )
        self.assertIn("drawdown", result["episode"])
        self.assertIn("concentration", result["episode"])

    def test_evaluate_agent_policy_history_has_one_row_per_period(self):
        result = evaluate_agent(DummyAgent(), self.returns, self.features)

        self.assertEqual(len(result["policy_history"]), len(self.returns))

    def test_evaluate_agent_policy_history_has_weight_columns_for_each_asset(self):
        result = evaluate_agent(DummyAgent(), self.returns, self.features)
        expected_weight_columns = {f"weight_{asset}" for asset in self.returns.columns}

        self.assertTrue(expected_weight_columns.issubset(result["policy_history"].columns))

    def test_evaluate_agent_policy_history_includes_datetime_date_column(self):
        result = evaluate_agent(DummyAgent(), self.returns, self.features)
        policy_history = result["policy_history"]

        self.assertIn("date", policy_history.columns)
        self.assertEqual(policy_history.loc[self.returns.index[0], "date"], self.returns.index[0])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(policy_history["date"]))

    def test_evaluate_agent_policy_history_includes_mandate_fields_when_available(self):
        result = evaluate_agent(
            DummyAgent(),
            self.returns,
            self.features,
            reward_config={
                "use_mandate_penalty": True,
                "lambda_mandate": 0.1,
                "mandate_profile": "moderate",
            },
        )
        policy_history = result["policy_history"]
        expected_columns = {
            "mandate_penalty",
            "mandate_drawdown_breach",
            "mandate_volatility_breach",
            "mandate_max_weight_breach",
            "mandate_effective_assets_breach",
            "mandate_turnover_breach",
        }

        self.assertTrue(expected_columns.issubset(policy_history.columns))

    def test_evaluate_agent_policy_history_does_not_require_mandate_fields(self):
        result = evaluate_agent(DummyAgent(), self.returns, self.features)
        policy_history = result["policy_history"]

        self.assertNotIn("mandate_penalty", policy_history.columns)
        self.assertNotIn("mandate_max_weight_breach", policy_history.columns)

    def test_evaluate_agent_policy_history_includes_turnover_reward_fields_when_available(self):
        result = evaluate_agent(
            DummyAgent(),
            self.returns,
            self.features,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 0.0,
                "lambda_turnover": 0.5,
                "turnover_penalty_mode": "excess_linear",
                "turnover_free_band": 0.10,
            },
            transaction_cost=0.0,
        )
        policy_history = result["policy_history"]
        expected_columns = {
            "turnover_penalty",
            "turnover_penalty_mode",
            "turnover_free_band",
            "turnover_excess",
        }

        self.assertTrue(expected_columns.issubset(policy_history.columns))
        self.assertEqual(policy_history["turnover_penalty_mode"].iloc[0], "excess_linear")
        self.assertAlmostEqual(policy_history["turnover_free_band"].iloc[0], 0.10)

    def test_build_policy_history_does_not_require_turnover_reward_fields(self):
        episode = run_policy_episode(DummyAgent(), self.returns, self.features)
        episode.pop("turnover_reward_info", None)

        policy_history = build_policy_history(episode)

        self.assertNotIn("turnover_penalty", policy_history.columns)
        self.assertNotIn("turnover_penalty_mode", policy_history.columns)

    def test_evaluate_agent_metrics_use_financial_net_returns(self):
        result = evaluate_agent(
            DummyAgent(),
            self.returns,
            self.features,
            transaction_cost=0.01,
        )
        policy_returns = result["episode"]["policy_returns"]
        financial_net_returns = result["episode"]["financial_net_returns"]

        self.assertFalse(policy_returns.equals(financial_net_returns))
        self.assertAlmostEqual(
            result["metrics"]["cumulative_return"],
            cumulative_return(financial_net_returns),
        )
        self.assertNotAlmostEqual(
            result["metrics"]["cumulative_return"],
            cumulative_return(policy_returns),
        )

    def test_metrics_contain_expected_keys(self):
        result = evaluate_agent(DummyAgent(), self.returns, self.features)
        expected_metric_keys = {
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        }

        self.assertEqual(set(result["metrics"].keys()), expected_metric_keys)

    def test_summarize_episode_diagnostics_returns_expected_keys(self):
        episode = run_policy_episode(DummyAgent(), self.returns, self.features)
        diagnostics = summarize_episode_diagnostics(episode)
        expected_keys = {
            "final_portfolio_value",
            "average_max_weight",
            "final_max_weight",
            "average_cash_weight",
            "final_cash_weight",
            "average_herfindahl_index",
            "final_herfindahl_index",
            "average_effective_number_of_assets",
            "final_effective_number_of_assets",
            "average_entropy",
            "final_entropy",
            "average_turnover",
            "final_turnover",
            "average_transaction_cost",
            "final_transaction_cost",
            "final_weights",
            "max_weight",
            "cash_weight",
        }

        self.assertEqual(set(diagnostics.keys()), expected_keys)

    def test_diagnostics_final_weights_and_cash_weight_are_well_formed(self):
        result = evaluate_agent(DummyAgent(), self.returns, self.features)
        diagnostics = result["diagnostics"]

        self.assertIsInstance(diagnostics["final_weights"], dict)
        self.assertEqual(set(diagnostics["final_weights"].keys()), set(self.returns.columns))
        self.assertGreaterEqual(diagnostics["max_weight"], 0.0)
        self.assertLessEqual(diagnostics["max_weight"], 1.0)
        self.assertGreaterEqual(diagnostics["cash_weight"], 0.0)
        self.assertLessEqual(diagnostics["cash_weight"], 1.0)
        self.assertGreaterEqual(diagnostics["average_turnover"], 0.0)
        self.assertGreaterEqual(diagnostics["average_transaction_cost"], 0.0)

    def test_diagnostics_include_allocation_risk_keys(self):
        result = evaluate_agent(DummyAgent(), self.returns, self.features)
        diagnostics = result["diagnostics"]

        self.assertIn("average_herfindahl_index", diagnostics)
        self.assertIn("final_herfindahl_index", diagnostics)
        self.assertIn("average_effective_number_of_assets", diagnostics)
        self.assertIn("final_effective_number_of_assets", diagnostics)
        self.assertIn("average_entropy", diagnostics)
        self.assertIn("final_entropy", diagnostics)
        self.assertIn("final_turnover", diagnostics)
        self.assertIn("final_transaction_cost", diagnostics)
        self.assertEqual(diagnostics["max_weight"], diagnostics["final_max_weight"])
        self.assertEqual(diagnostics["cash_weight"], diagnostics["final_cash_weight"])


if __name__ == "__main__":
    unittest.main()
