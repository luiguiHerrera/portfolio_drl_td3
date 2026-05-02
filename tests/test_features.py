"""Tests for minimal return-based feature engineering."""

import unittest

import pandas as pd

from src.data.features import build_features


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.returns = pd.DataFrame(
            {
                "SPY": [
                    0.01,
                    0.02,
                    -0.01,
                    0.00,
                    0.01,
                    0.02,
                    -0.02,
                    0.01,
                    0.00,
                    0.03,
                    -0.01,
                    0.02,
                    0.01,
                    -0.01,
                    0.02,
                ],
                "TLT": [
                    0.00,
                    0.01,
                    0.01,
                    -0.01,
                    0.02,
                    0.00,
                    0.01,
                    -0.02,
                    0.01,
                    0.00,
                    0.02,
                    -0.01,
                    0.01,
                    0.00,
                    0.01,
                ],
                "GLD": [
                    0.02,
                    -0.01,
                    0.00,
                    0.01,
                    -0.02,
                    0.02,
                    0.01,
                    0.00,
                    -0.01,
                    0.01,
                    0.02,
                    0.00,
                    -0.01,
                    0.02,
                    0.01,
                ],
                "BTC-USD": [
                    0.03,
                    -0.02,
                    0.04,
                    -0.01,
                    0.05,
                    -0.03,
                    0.02,
                    0.01,
                    -0.04,
                    0.03,
                    0.02,
                    -0.01,
                    0.04,
                    -0.02,
                    0.03,
                ],
                "CASH": [0.0] * 15,
            },
            index=pd.date_range("2024-01-05", periods=15, freq="W-FRI"),
        )

    def test_build_features_returns_dataframe(self):
        features = build_features(self.returns)

        self.assertIsInstance(features, pd.DataFrame)

    def test_feature_columns_include_expected_names_for_spy_and_cash(self):
        features = build_features(self.returns)
        expected_columns = {
            "SPY_ret_1w",
            "SPY_mom_4w",
            "SPY_mom_12w",
            "SPY_vol_4w",
            "SPY_vol_12w",
            "CASH_ret_1w",
            "CASH_mom_4w",
            "CASH_mom_12w",
            "CASH_vol_4w",
            "CASH_vol_12w",
        }

        self.assertTrue(expected_columns.issubset(set(features.columns)))

    def test_output_has_no_nan_values(self):
        features = build_features(self.returns)

        self.assertFalse(features.isna().any().any())

    def test_cash_feature_columns_are_zero(self):
        features = build_features(self.returns)
        cash_features = features.filter(like="CASH_")

        self.assertTrue((cash_features == 0.0).all().all())

    def test_output_index_is_shorter_than_input(self):
        features = build_features(self.returns)

        self.assertLess(len(features), len(self.returns))

    def test_chronological_order_is_preserved(self):
        features = build_features(self.returns)

        self.assertTrue(features.index.is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
