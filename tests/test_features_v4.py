"""Tests for Feature Set V4 construction."""

import unittest
from unittest.mock import patch

import pandas as pd

from src.data.feature_factory import build_configured_features
from src.data.features_v2 import build_features_v2
from src.data.features_v4 import build_features_v4


class FeatureSetV4Tests(unittest.TestCase):
    def setUp(self):
        self.returns = self._realistic_returns()

    def test_build_features_v4_returns_non_empty_dataframe(self):
        features = build_features_v4(self.returns)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertFalse(features.empty)

    def test_v4_includes_v2_columns(self):
        v2_features = build_features_v2(self.returns)
        v4_features = build_features_v4(self.returns)

        self.assertTrue(set(v2_features.columns).issubset(v4_features.columns))

    def test_v4_includes_garch_absolute_vol_columns(self):
        features = build_features_v4(self.returns)

        for asset in self.returns.columns:
            self.assertIn(f"garch_vol_{asset}", features.columns)

    def test_v4_includes_garch_relative_columns_when_enabled(self):
        features = build_features_v4(self.returns, garch_include_relative=True)

        self.assertIn("garch_vol_ratio_GLD_vs_SPY", features.columns)
        self.assertIn("garch_vol_rank_BTC-USD", features.columns)

    def test_v4_excludes_garch_features_when_disabled(self):
        features = build_features_v4(self.returns, include_garch_features=False)

        self.assertFalse(any(column.startswith("garch_") for column in features.columns))

    def test_v4_rolling_fitted_mode_excludes_cash_garch_when_requested(self):
        features = build_features_v4(
            self.returns,
            garch_mode="rolling_fitted",
            garch_min_history=8,
            garch_window=12,
            garch_exclude_cash=True,
        )

        self.assertIn("garch_vol_SPY", features.columns)
        self.assertNotIn("garch_vol_CASH", features.columns)
        self.assertFalse(features.empty)

    def test_v4_rolling_fitted_mode_differs_from_deterministic_filter(self):
        deterministic = build_features_v4(self.returns, garch_mode="deterministic_filter")
        fitted = build_features_v4(
            self.returns,
            garch_mode="rolling_fitted",
            garch_min_history=8,
            garch_window=12,
            garch_exclude_cash=True,
        )
        common_index = fitted.index.intersection(deterministic.index)
        max_diff = (
            fitted.loc[common_index, "garch_vol_SPY"]
            - deterministic.loc[common_index, "garch_vol_SPY"]
        ).abs().max()

        self.assertGreater(float(max_diff), 0.0)

    def test_v4_preserves_index_alignment_and_has_no_missing_values(self):
        features = build_features_v4(self.returns)

        self.assertTrue(features.index.isin(self.returns.index).all())
        self.assertEqual(int(features.isna().sum().sum()), 0)

    def test_v4_does_not_mutate_input(self):
        returns = self.returns.copy(deep=True)

        build_features_v4(returns)

        pd.testing.assert_frame_equal(returns, self.returns)

    def test_v4_raises_value_error_on_empty_returns(self):
        returns = pd.DataFrame(index=pd.DatetimeIndex([]))

        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_features_v4(returns)

    def test_feature_factory_dispatches_v4(self):
        expected = pd.DataFrame(
            {"v4_feature": [1.0, 2.0]},
            index=self.returns.index[:2],
        )
        config = {
            "features": {
                "version": "v4",
                "market_asset": "SPY",
                "include_garch_features": True,
            }
        }

        with patch(
            "src.data.feature_factory.build_features_v4",
            return_value=expected,
        ) as build_features_v4_mock:
            result = build_configured_features(self.returns, config)

        build_features_v4_mock.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    def test_invalid_garch_parameters_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "alpha \\+ beta"):
            build_features_v4(
                self.returns,
                garch_alpha=0.50,
                garch_beta=0.50,
            )

    @staticmethod
    def _realistic_returns() -> pd.DataFrame:
        index = pd.date_range("2022-01-07", periods=40, freq="W-FRI")
        values = {
            "SPY": [
                0.010,
                0.015,
                -0.008,
                0.012,
                -0.020,
                0.018,
                0.006,
                -0.011,
            ],
            "GLD": [
                0.004,
                -0.002,
                0.006,
                0.001,
                0.012,
                -0.005,
                0.003,
                0.009,
            ],
            "TLT": [
                -0.003,
                0.004,
                0.005,
                -0.006,
                0.011,
                0.002,
                -0.004,
                0.007,
            ],
            "BTC-USD": [
                0.040,
                -0.030,
                0.050,
                -0.020,
                0.060,
                -0.045,
                0.030,
                -0.015,
            ],
            "CASH": [0.0] * 8,
        }
        repeated_values = {
            asset: (asset_values * 5)[: len(index)]
            for asset, asset_values in values.items()
        }

        return pd.DataFrame(repeated_values, index=index)


if __name__ == "__main__":
    unittest.main()
