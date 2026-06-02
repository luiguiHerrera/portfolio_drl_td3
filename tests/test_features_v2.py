"""Tests for Feature Set V2 construction."""

import unittest

import pandas as pd

from src.data.features_v2 import (
    build_features_v2,
    ewma_volatility,
    rolling_beta,
    rolling_correlation,
    rolling_drawdown,
)


class FeatureSetV2Tests(unittest.TestCase):
    def setUp(self):
        self.returns = self._realistic_returns()

    def test_build_features_v2_returns_non_empty_dataframe(self):
        features = build_features_v2(self.returns)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertFalse(features.empty)

    def test_output_index_is_datetime_and_monotonic_increasing(self):
        features = build_features_v2(self.returns)

        self.assertIsInstance(features.index, pd.DatetimeIndex)
        self.assertTrue(features.index.is_monotonic_increasing)

    def test_output_contains_expected_asset_columns(self):
        features = build_features_v2(self.returns)
        expected_columns = set()
        for asset in ("SPY", "GLD", "TLT", "BTC-USD", "CASH"):
            expected_columns.update(
                {
                    f"{asset}_ret_1p",
                    f"{asset}_mom_4p",
                    f"{asset}_mom_12p",
                    f"{asset}_vol_4p",
                    f"{asset}_vol_12p",
                    f"{asset}_ewma_vol_12p",
                    f"{asset}_rolling_drawdown_12p",
                }
            )
            if asset != "SPY":
                expected_columns.update(
                    {
                        f"{asset}_beta_vs_SPY_12p",
                        f"{asset}_corr_vs_SPY_12p",
                    }
                )

        self.assertTrue(expected_columns.issubset(set(features.columns)))

    def test_output_excludes_market_asset_self_reference_columns(self):
        features = build_features_v2(self.returns)

        self.assertNotIn("SPY_beta_vs_SPY_12p", features.columns)
        self.assertNotIn("SPY_corr_vs_SPY_12p", features.columns)

    def test_output_does_not_contain_duplicate_distance_from_high_columns(self):
        features = build_features_v2(self.returns)

        self.assertFalse(any("distance_from_high" in column for column in features.columns))

    def test_output_contains_regime_columns(self):
        features = build_features_v2(self.returns)
        expected_columns = {
            "market_high_vol_regime",
            "market_risk_off_regime",
            "market_trend_regime",
            "market_defensive_regime",
        }

        self.assertTrue(expected_columns.issubset(set(features.columns)))

    def test_rolling_beta_is_one_when_asset_equals_market(self):
        series = self.returns["SPY"]
        beta = rolling_beta(series, series, window=12).dropna()

        self.assertAlmostEqual(beta.iloc[-1], 1.0)

    def test_rolling_correlation_is_one_when_asset_equals_market(self):
        series = self.returns["SPY"]
        correlation = rolling_correlation(series, series, window=12).dropna()

        self.assertAlmostEqual(correlation.iloc[-1], 1.0)

    def test_ewma_volatility_is_non_negative_after_enough_observations(self):
        volatility = ewma_volatility(self.returns["SPY"], span=12).dropna()

        self.assertTrue((volatility >= 0.0).all())

    def test_rolling_drawdown_is_non_positive(self):
        drawdown = rolling_drawdown(self.returns["SPY"], window=12).dropna()

        self.assertTrue((drawdown <= 0.0).all())

    def test_empty_returns_raise_value_error(self):
        returns = pd.DataFrame(index=pd.DatetimeIndex([]))

        with self.assertRaisesRegex(ValueError, "non-empty DataFrame"):
            build_features_v2(returns)

    def test_missing_market_asset_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "market_asset 'QQQ'"):
            build_features_v2(self.returns, market_asset="QQQ")

    def test_invalid_windows_and_spans_raise_value_error(self):
        invalid_arguments = (
            {"short_window": 1},
            {"long_window": 1},
            {"ewma_span": 1},
            {"short_window": 12, "long_window": 4},
        )

        for kwargs in invalid_arguments:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    build_features_v2(self.returns, **kwargs)

    def test_too_short_returns_raise_value_error_after_dropna(self):
        short_returns = self.returns.iloc[:5]

        with self.assertRaisesRegex(ValueError, "Feature Set V2 output is empty"):
            build_features_v2(short_returns)

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
