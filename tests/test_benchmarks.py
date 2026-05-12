"""Tests for basic portfolio benchmark utilities."""

import unittest

import pandas as pd
from pandas.testing import assert_series_equal

from src.backtest.benchmarks import (
    buy_and_hold_returns,
    equal_weight_rebalanced_benchmark,
    equal_weight_returns,
    individual_buy_and_hold_returns,
)


class BenchmarkTests(unittest.TestCase):
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

    def test_equal_weight_returns_are_row_wise_mean(self):
        result = equal_weight_returns(self.returns)
        expected = self.returns.mean(axis=1)

        assert_series_equal(result, expected)

    def test_buy_and_hold_returns_preserve_length_and_index(self):
        result = buy_and_hold_returns(self.returns)

        self.assertEqual(len(result), len(self.returns))
        self.assertTrue(result.index.equals(self.returns.index))

    def test_buy_and_hold_returns_accept_valid_custom_weights(self):
        weights = pd.Series(
            {
                "SPY": 0.30,
                "TLT": 0.20,
                "GLD": 0.20,
                "BTC-USD": 0.10,
                "CASH": 0.20,
            }
        )

        result = buy_and_hold_returns(self.returns, initial_weights=weights)

        self.assertEqual(len(result), len(self.returns))

    def test_buy_and_hold_returns_reject_negative_weights(self):
        weights = pd.Series(
            {
                "SPY": 0.50,
                "TLT": 0.30,
                "GLD": 0.20,
                "BTC-USD": 0.10,
                "CASH": -0.10,
            }
        )

        with self.assertRaises(ValueError):
            buy_and_hold_returns(self.returns, initial_weights=weights)

    def test_buy_and_hold_returns_reject_weights_that_do_not_sum_to_one(self):
        weights = pd.Series(
            {
                "SPY": 0.30,
                "TLT": 0.20,
                "GLD": 0.20,
                "BTC-USD": 0.10,
                "CASH": 0.10,
            }
        )

        with self.assertRaises(ValueError):
            buy_and_hold_returns(self.returns, initial_weights=weights)

    def test_cash_zero_returns_do_not_break_buy_and_hold(self):
        result = buy_and_hold_returns(self.returns)

        self.assertFalse(result.isna().any())

    def test_individual_buy_and_hold_returns_one_entry_per_asset(self):
        result = individual_buy_and_hold_returns(self.returns)

        self.assertEqual(len(result), len(self.returns.columns))

    def test_individual_buy_and_hold_returns_expected_keys(self):
        result = individual_buy_and_hold_returns(self.returns)

        self.assertEqual(
            set(result.keys()),
            {"buy_hold_SPY", "buy_hold_TLT", "buy_hold_GLD", "buy_hold_BTC-USD", "buy_hold_CASH"},
        )

    def test_individual_buy_and_hold_returns_match_asset_columns(self):
        result = individual_buy_and_hold_returns(self.returns)

        for asset in self.returns.columns:
            expected = self.returns[asset].rename(f"buy_hold_{asset}")
            assert_series_equal(result[f"buy_hold_{asset}"], expected)

    def test_individual_buy_and_hold_returns_rejects_empty_returns(self):
        with self.assertRaises(ValueError):
            individual_buy_and_hold_returns(pd.DataFrame())

    def test_equal_weight_rebalanced_benchmark_returns_expected_keys(self):
        result = equal_weight_rebalanced_benchmark(self.returns)

        self.assertEqual(
            set(result.keys()),
            {"gross_returns", "net_returns", "turnover", "transaction_costs", "weights"},
        )

    def test_equal_weight_rebalanced_net_equals_gross_without_transaction_cost(self):
        result = equal_weight_rebalanced_benchmark(self.returns, transaction_cost=0.0)

        assert_series_equal(result["net_returns"], result["gross_returns"], check_names=False)

    def test_equal_weight_rebalanced_net_less_or_equal_gross_when_costs_apply(self):
        result = equal_weight_rebalanced_benchmark(
            self.returns,
            transaction_cost=0.01,
        )

        self.assertTrue((result["net_returns"] <= result["gross_returns"]).all())
        self.assertLess(result["net_returns"].iloc[0], result["gross_returns"].iloc[0])

    def test_equal_weight_rebalanced_turnover_positive_after_return_drift(self):
        result = equal_weight_rebalanced_benchmark(
            self.returns,
            transaction_cost=0.01,
        )

        self.assertGreater(result["turnover"].iloc[0], 0.0)

    def test_equal_weight_rebalanced_weights_have_asset_columns(self):
        result = equal_weight_rebalanced_benchmark(self.returns)

        self.assertEqual(list(result["weights"].columns), list(self.returns.columns))

    def test_equal_weight_rebalanced_weights_sum_to_one(self):
        result = equal_weight_rebalanced_benchmark(self.returns)

        self.assertTrue((result["weights"].sum(axis=1).round(12) == 1.0).all())

    def test_equal_weight_rebalanced_turnover_and_costs_are_non_negative(self):
        result = equal_weight_rebalanced_benchmark(self.returns, transaction_cost=0.01)

        self.assertTrue((result["turnover"] >= 0.0).all())
        self.assertTrue((result["transaction_costs"] >= 0.0).all())

    def test_equal_weight_rebalanced_rejects_negative_transaction_cost(self):
        with self.assertRaises(ValueError):
            equal_weight_rebalanced_benchmark(self.returns, transaction_cost=-0.01)

    def test_equal_weight_rebalanced_rejects_empty_returns(self):
        with self.assertRaises(ValueError):
            equal_weight_rebalanced_benchmark(pd.DataFrame())

    def test_equal_weight_rebalanced_rejects_non_positive_post_return_value(self):
        returns = pd.DataFrame(
            {
                "SPY": [-2.0],
                "TLT": [-2.0],
                "GLD": [-2.0],
                "BTC-USD": [-2.0],
                "CASH": [-2.0],
            },
            index=pd.date_range("2024-01-05", periods=1, freq="W-FRI"),
        )

        with self.assertRaises(ValueError):
            equal_weight_rebalanced_benchmark(returns)


if __name__ == "__main__":
    unittest.main()
