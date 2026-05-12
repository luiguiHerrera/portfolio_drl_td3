"""Tests for basic policy comparison utilities."""

import unittest

import numpy as np
import pandas as pd

from src.backtest.compare_policies import compare_agent_to_basic_benchmarks
from src.backtest.evaluate_policy import cumulative_return


class EqualWeightDummyAgent:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def select_action(self, state):
        return np.full(self.action_dim, 1.0 / self.action_dim)


class RotatingDummyAgent:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.calls = 0

    def select_action(self, state):
        action = np.zeros(self.action_dim)
        action[self.calls % self.action_dim] = 1.0
        self.calls += 1

        return action


class ComparePoliciesTests(unittest.TestCase):
    def setUp(self):
        self.returns = pd.DataFrame(
            {
                "SPY": [0.01, 0.02, -0.01, 0.00],
                "TLT": [0.00, 0.01, 0.01, -0.01],
                "GLD": [0.02, -0.01, 0.00, 0.01],
                "BTC-USD": [0.03, -0.02, 0.04, -0.01],
                "CASH": [0.00, 0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
        )
        self.features = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0],
                "feature_b": [0.1, 0.2, 0.3, 0.4],
            },
            index=self.returns.index,
        )
        self.agent = EqualWeightDummyAgent(action_dim=len(self.returns.columns))

    def test_compare_returns_expected_top_level_keys(self):
        result = compare_agent_to_basic_benchmarks(self.agent, self.returns, self.features)

        self.assertEqual(set(result.keys()), {"agent", "benchmarks", "metrics_table"})

    def test_compare_returns_basic_benchmark_entries(self):
        result = compare_agent_to_basic_benchmarks(self.agent, self.returns, self.features)

        self.assertEqual(
            set(result["benchmarks"].keys()),
            {"equal_weight_gross", "equal_weight_rebalanced_net", "buy_and_hold"},
        )
        self.assertEqual(
            set(result["benchmarks"]["equal_weight_gross"].keys()),
            {"returns", "metrics"},
        )
        self.assertEqual(
            set(result["benchmarks"]["equal_weight_rebalanced_net"].keys()),
            {"returns", "metrics", "diagnostics"},
        )
        self.assertEqual(
            set(result["benchmarks"]["buy_and_hold"].keys()),
            {"returns", "metrics"},
        )

    def test_equal_weight_rebalanced_net_contains_expected_diagnostics(self):
        result = compare_agent_to_basic_benchmarks(self.agent, self.returns, self.features)
        benchmark = result["benchmarks"]["equal_weight_rebalanced_net"]

        self.assertEqual(
            set(benchmark["diagnostics"].keys()),
            {"turnover", "transaction_costs", "weights"},
        )

    def test_benchmark_returns_preserve_evaluation_index(self):
        result = compare_agent_to_basic_benchmarks(self.agent, self.returns, self.features)
        agent_index = result["agent"]["episode"]["policy_returns"].index

        self.assertTrue(result["benchmarks"]["equal_weight_gross"]["returns"].index.equals(agent_index))
        self.assertTrue(
            result["benchmarks"]["equal_weight_rebalanced_net"]["returns"].index.equals(agent_index)
        )
        self.assertTrue(result["benchmarks"]["buy_and_hold"]["returns"].index.equals(agent_index))

    def test_metrics_table_contains_all_policy_rows(self):
        result = compare_agent_to_basic_benchmarks(self.agent, self.returns, self.features)

        self.assertEqual(
            set(result["metrics_table"].index),
            {"agent", "equal_weight_gross", "equal_weight_rebalanced_net", "buy_and_hold"},
        )

    def test_metrics_table_contains_expected_metric_columns(self):
        result = compare_agent_to_basic_benchmarks(self.agent, self.returns, self.features)
        expected_columns = {
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        }

        self.assertEqual(set(result["metrics_table"].columns), expected_columns)

    def test_agent_metrics_are_based_on_financial_net_returns(self):
        agent = RotatingDummyAgent(action_dim=len(self.returns.columns))
        result = compare_agent_to_basic_benchmarks(
            agent,
            self.returns,
            self.features,
            transaction_cost=0.01,
            reward_config={
                "lambda_return": 1.0,
                "lambda_transaction_cost": 1.0,
                "lambda_turnover": 0.1,
            },
        )
        policy_returns = result["agent"]["episode"]["policy_returns"]
        financial_net_returns = result["agent"]["episode"]["financial_net_returns"]
        rewards = result["agent"]["episode"]["rewards"]

        self.assertFalse(policy_returns.equals(financial_net_returns))
        self.assertFalse(policy_returns.equals(rewards))
        self.assertAlmostEqual(
            result["metrics_table"].loc["agent", "cumulative_return"],
            cumulative_return(financial_net_returns),
        )
        self.assertNotAlmostEqual(
            result["metrics_table"].loc["agent", "cumulative_return"],
            cumulative_return(policy_returns),
        )

    def test_equal_weight_rebalanced_net_applies_transaction_costs_when_drift_occurs(self):
        result = compare_agent_to_basic_benchmarks(
            self.agent,
            self.returns,
            self.features,
            transaction_cost=0.01,
        )
        transaction_costs = result["benchmarks"]["equal_weight_rebalanced_net"][
            "diagnostics"
        ]["transaction_costs"]

        self.assertTrue((transaction_costs >= 0.0).all())
        self.assertGreater(transaction_costs.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
