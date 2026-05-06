"""Tests for TD3 agent evaluation utilities."""

import unittest

import numpy as np
import pandas as pd

from src.backtest.evaluate_agent import evaluate_agent, run_policy_episode


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
            "rewards",
            "portfolio_values",
            "turnover",
            "transaction_costs",
            "weights",
            "final_portfolio_value",
        }

        self.assertEqual(set(episode.keys()), expected_keys)

    def test_policy_returns_length_matches_aligned_env_returns_length(self):
        episode = run_policy_episode(DummyAgent(), self.returns, self.features)

        self.assertEqual(len(episode["policy_returns"]), len(self.returns))

    def test_weights_dataframe_has_asset_columns(self):
        episode = run_policy_episode(DummyAgent(), self.returns, self.features)

        self.assertEqual(list(episode["weights"].columns), list(self.returns.columns))

    def test_policy_returns_differ_from_rewards_when_transaction_costs_apply(self):
        episode = run_policy_episode(
            DummyAgent(),
            self.returns,
            self.features,
            transaction_cost=0.01,
        )

        self.assertFalse(episode["policy_returns"].equals(episode["rewards"]))

    def test_evaluate_agent_returns_episode_and_metrics(self):
        result = evaluate_agent(DummyAgent(), self.returns, self.features)

        self.assertEqual(set(result.keys()), {"episode", "metrics"})

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


if __name__ == "__main__":
    unittest.main()
