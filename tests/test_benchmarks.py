"""Tests for basic portfolio benchmark utilities."""

import unittest

import pandas as pd
from pandas.testing import assert_series_equal

from src.backtest.benchmarks import buy_and_hold_returns, equal_weight_returns


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


if __name__ == "__main__":
    unittest.main()
