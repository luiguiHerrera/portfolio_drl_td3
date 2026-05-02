"""Tests for portfolio evaluation metrics."""

import unittest

import pandas as pd

from src.backtest.evaluate_policy import (
    annualized_return,
    annualized_volatility,
    cumulative_return,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    summary_metrics,
)


class EvaluatePolicyTests(unittest.TestCase):
    def setUp(self):
        self.returns = pd.Series(
            [0.01, -0.02, 0.03, 0.00],
            index=pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
            name="portfolio_returns",
        )

    def test_equity_curve_preserves_index_without_extra_initial_row(self):
        result = equity_curve(self.returns, initial_value=1.0)

        self.assertTrue(result.index.equals(self.returns.index))
        self.assertEqual(len(result), len(self.returns))
        self.assertAlmostEqual(result.iloc[0], 1.01)

    def test_cumulative_return_matches_manual_compounded_result(self):
        expected = (1.01 * 0.98 * 1.03 * 1.0) - 1.0

        self.assertAlmostEqual(cumulative_return(self.returns), expected)

    def test_annualized_return_works_on_non_empty_returns(self):
        result = annualized_return(self.returns)

        self.assertIsInstance(result, float)

    def test_annualized_volatility_is_positive_for_variable_returns(self):
        result = annualized_volatility(self.returns)

        self.assertGreater(result, 0.0)

    def test_sharpe_ratio_returns_zero_for_constant_returns(self):
        returns = pd.Series([0.01, 0.01, 0.01])

        self.assertEqual(sharpe_ratio(returns), 0.0)

    def test_sharpe_ratio_returns_zero_for_single_observation(self):
        returns = pd.Series([0.01])

        self.assertEqual(sharpe_ratio(returns), 0.0)

    def test_max_drawdown_is_negative_or_zero(self):
        result = max_drawdown(self.returns)

        self.assertLessEqual(result, 0.0)

    def test_summary_metrics_contains_expected_keys(self):
        result = summary_metrics(self.returns)
        expected_keys = {
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        }

        self.assertEqual(set(result.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
