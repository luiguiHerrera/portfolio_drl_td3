"""Tests for Feature Set V5 construction."""

import unittest
from unittest.mock import patch

import pandas as pd

from src.data.feature_factory import build_configured_features
from src.data.features_v2 import build_features_v2
from src.data.features_v5 import build_features_v5, build_v5_regime_auxiliary_features


class FeatureSetV5Tests(unittest.TestCase):
    def setUp(self):
        self.returns = self._realistic_returns()

    def test_v5_returns_non_empty_dataframe(self):
        features = build_features_v5(self.returns)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertFalse(features.empty)

    def test_v5_includes_v2_columns(self):
        v2_features = build_features_v2(self.returns)
        v5_features = build_features_v5(self.returns)

        self.assertTrue(set(v2_features.columns).issubset(v5_features.columns))

    def test_v5_includes_market_trend_and_drawdown_regime_columns(self):
        features = build_features_v5(self.returns)
        expected_columns = {
            "regime_market_momentum_12p",
            "regime_market_trend_positive",
            "regime_market_trend_negative",
            "regime_market_rolling_drawdown_12p",
            "regime_market_drawdown_stress",
        }

        self.assertTrue(expected_columns.issubset(features.columns))

    def test_v5_includes_market_volatility_regime_columns(self):
        features = build_features_v5(self.returns)
        expected_columns = {
            "regime_market_vol_4p",
            "regime_market_vol_12p",
            "regime_market_high_vol",
        }

        self.assertTrue(expected_columns.issubset(features.columns))

    def test_v5_includes_asset_vs_market_correlation_columns(self):
        features = build_features_v5(self.returns)

        self.assertIn("corr_TLT_vs_SPY_12p", features.columns)
        self.assertIn("corr_GLD_vs_SPY_12p", features.columns)
        self.assertIn("corr_BTC-USD_vs_SPY_12p", features.columns)

    def test_v5_excludes_cash_from_risky_correlation_features(self):
        features = build_features_v5(self.returns)

        self.assertFalse(any(column.startswith("corr_CASH") for column in features.columns))

    def test_v5_includes_avg_pairwise_corr(self):
        features = build_features_v5(self.returns)

        self.assertIn("avg_pairwise_corr_12p", features.columns)

    def test_v5_includes_diversification_benefit_score(self):
        features = build_features_v5(self.returns)

        self.assertIn("diversification_benefit_score", features.columns)

    def test_v5_includes_risk_off_score_and_state(self):
        features = build_features_v5(self.returns)

        self.assertIn("risk_off_score", features.columns)
        self.assertIn("risk_off_state", features.columns)

    def test_v5_includes_tlt_hedge_signal_when_tlt_exists(self):
        features = build_features_v5(self.returns)

        self.assertIn("tlt_equity_hedge_signal", features.columns)

    def test_v5_includes_gld_hedge_signal_when_gld_exists(self):
        features = build_features_v5(self.returns)

        self.assertIn("gld_equity_hedge_signal", features.columns)

    def test_v5_does_not_mutate_input(self):
        returns = self.returns.copy(deep=True)

        build_features_v5(returns)

        pd.testing.assert_frame_equal(returns, self.returns)

    def test_v5_has_no_missing_values_after_final_dropna(self):
        features = build_features_v5(self.returns)

        self.assertEqual(int(features.isna().sum().sum()), 0)

    def test_build_v5_regime_auxiliary_features_returns_raw_regime_columns(self):
        auxiliary_features = build_v5_regime_auxiliary_features(self.returns)
        expected_columns = [
            "regime_market_trend_positive",
            "regime_market_trend_negative",
            "regime_market_drawdown_stress",
            "regime_market_high_vol",
            "correlation_stress",
            "risk_off_score",
            "risk_off_state",
        ]

        self.assertEqual(list(auxiliary_features.columns), expected_columns)
        self.assertFalse(auxiliary_features.empty)
        for column in expected_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(auxiliary_features[column]))
        self.assertTrue(set(auxiliary_features["risk_off_state"].unique()).issubset({0.0, 1.0}))

    def test_build_v5_regime_auxiliary_features_does_not_mutate_input(self):
        returns = self.returns.copy(deep=True)

        build_v5_regime_auxiliary_features(returns)

        pd.testing.assert_frame_equal(returns, self.returns)

    def test_v5_raises_value_error_when_returns_empty(self):
        returns = pd.DataFrame(index=pd.DatetimeIndex([]))

        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_features_v5(returns)

    def test_v5_raises_value_error_when_market_asset_missing(self):
        with self.assertRaisesRegex(ValueError, "market_asset 'QQQ'"):
            build_features_v5(self.returns, market_asset="QQQ")

    def test_v5_raises_value_error_for_invalid_correlation_window(self):
        with self.assertRaisesRegex(ValueError, "correlation_window"):
            build_features_v5(self.returns, correlation_window=1)

    def test_v5_raises_value_error_for_invalid_drawdown_window(self):
        with self.assertRaisesRegex(ValueError, "drawdown_window"):
            build_features_v5(self.returns, drawdown_window=1)

    def test_v5_raises_value_error_for_bool_window_or_threshold_inputs(self):
        invalid_arguments = (
            {"short_window": True},
            {"long_window": True},
            {"ewma_span": True},
            {"correlation_window": True},
            {"drawdown_window": True},
            {"risk_off_threshold": True},
        )

        for kwargs in invalid_arguments:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    build_features_v5(self.returns, **kwargs)

    def test_feature_factory_dispatches_v5(self):
        expected = pd.DataFrame(
            {"v5_feature": [1.0, 2.0]},
            index=self.returns.index[:2],
        )
        config = {
            "features": {
                "version": "v5",
                "market_asset": "SPY",
                "correlation_window": 9,
                "drawdown_window": 10,
                "risk_off_threshold": 3.0,
            }
        }

        with patch(
            "src.data.feature_factory.build_features_v5",
            return_value=expected,
        ) as build_features_v5_mock:
            result = build_configured_features(self.returns, config)

        build_features_v5_mock.assert_called_once_with(
            self.returns,
            market_asset="SPY",
            short_window=4,
            long_window=12,
            ewma_span=12,
            correlation_window=9,
            drawdown_window=10,
            risk_off_threshold=3.0,
        )
        pd.testing.assert_frame_equal(result, expected)

    @staticmethod
    def _realistic_returns() -> pd.DataFrame:
        index = pd.date_range("2022-01-07", periods=48, freq="W-FRI")
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
            "TLT": [
                -0.006,
                -0.008,
                0.005,
                -0.004,
                0.014,
                -0.007,
                -0.002,
                0.009,
            ],
            "GLD": [
                -0.004,
                -0.002,
                0.006,
                0.001,
                0.012,
                -0.005,
                0.003,
                0.009,
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
            asset: (asset_values * 6)[: len(index)]
            for asset, asset_values in values.items()
        }

        return pd.DataFrame(repeated_values, index=index)


if __name__ == "__main__":
    unittest.main()
