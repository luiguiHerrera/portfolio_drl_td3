"""Tests for minimal data preprocessing behavior."""

import unittest

import pandas as pd

from src.data.preprocess import compute_returns


class DataPreprocessTests(unittest.TestCase):
    def setUp(self):
        self.prices = pd.DataFrame(
            {
                "SPY": [100.0, 101.0, 102.0, 104.0, 105.0, 107.0, 108.0, 110.0],
                "TLT": [90.0, 91.0, 92.0, 91.0, 93.0, 94.0, 95.0, 96.0],
                "GLD": [180.0, 181.0, 179.0, 182.0, 183.0, 184.0, 185.0, 186.0],
                "BTC-USD": [30000.0, 30100.0, 29900.0, 30500.0, 30700.0, 31000.0, 30900.0, 31200.0],
            },
            index=pd.date_range("2024-01-01", periods=8, freq="D"),
        )
        self.assets = ["SPY", "TLT", "GLD", "BTC-USD", "CASH"]

    def test_cash_is_added_as_zero_returns(self):
        returns = compute_returns(self.prices, self.assets, frequency="daily")

        self.assertIn("CASH", returns.columns)
        self.assertTrue((returns["CASH"] == 0.0).all())

    def test_output_columns_preserve_asset_order(self):
        returns = compute_returns(self.prices, self.assets, frequency="daily")

        self.assertEqual(list(returns.columns), self.assets)

    def test_weekly_returns_are_produced(self):
        returns = compute_returns(self.prices, self.assets, frequency="weekly")

        self.assertFalse(returns.empty)
        self.assertEqual(list(returns.columns), self.assets)


if __name__ == "__main__":
    unittest.main()
