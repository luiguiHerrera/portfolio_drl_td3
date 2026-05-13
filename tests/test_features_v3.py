"""Tests for Feature Set V3 macro feature construction."""

import unittest

import pandas as pd

from src.data.features_v2 import build_features_v2
from src.data.features_v3 import build_features_v3, build_macro_features


class FeatureSetV3Tests(unittest.TestCase):
    def setUp(self):
        self.returns = self._realistic_returns()

    def test_build_features_v3_without_macro_data_returns_v2_columns(self):
        v2_features = build_features_v2(self.returns)
        v3_features = build_features_v3(self.returns)

        self.assertEqual(list(v3_features.columns), list(v2_features.columns))

    def test_build_features_v3_with_macro_data_adds_macro_prefixed_columns(self):
        macro_data = pd.DataFrame(
            {"DGS10": [4.0, 4.1, 4.2]},
            index=pd.to_datetime(["2022-01-07", "2022-04-01", "2022-07-01"]),
        )

        features = build_features_v3(self.returns, macro_data=macro_data)

        self.assertIn("macro_DGS10", features.columns)

    def test_macro_alignment_uses_forward_fill_and_does_not_backfill(self):
        target_index = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
        macro_data = pd.DataFrame(
            {"VIX": [20.0, 25.0]},
            index=[target_index[1], target_index[3]],
        )

        features = build_macro_features(macro_data, target_index)

        self.assertTrue(pd.isna(features.loc[target_index[0], "macro_VIX"]))
        self.assertEqual(features.loc[target_index[1], "macro_VIX"], 20.0)
        self.assertEqual(features.loc[target_index[2], "macro_VIX"], 20.0)
        self.assertEqual(features.loc[target_index[3], "macro_VIX"], 25.0)
        self.assertEqual(features.loc[target_index[4], "macro_VIX"], 25.0)

    def test_rows_before_first_macro_observation_are_dropped_after_alignment(self):
        v2_features = build_features_v2(self.returns)
        first_macro_date = v2_features.index[2]
        macro_data = pd.DataFrame(
            {"DGS10": [4.0]},
            index=[first_macro_date],
        )

        features = build_features_v3(self.returns, macro_data=macro_data)

        self.assertEqual(features.index.min(), first_macro_date)

    def test_yield_curve_features_are_created_when_dgs10_and_dgs2_exist(self):
        target_index = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
        macro_data = pd.DataFrame(
            {"DGS10": [4.0, 3.8], "DGS2": [3.0, 4.0]},
            index=[target_index[0], target_index[1]],
        )

        features = build_macro_features(macro_data, target_index)

        self.assertIn("macro_yield_curve_10y_2y", features.columns)
        self.assertIn("macro_inverted_yield_curve_regime", features.columns)
        self.assertAlmostEqual(
            features.loc[target_index[1], "macro_yield_curve_10y_2y"],
            -0.2,
        )
        self.assertEqual(features.loc[target_index[1], "macro_inverted_yield_curve_regime"], 1.0)

    def test_vix_regime_is_created_when_vix_exists(self):
        target_index = pd.date_range("2024-01-05", periods=14, freq="W-FRI")
        macro_data = pd.DataFrame(
            {"VIX": [float(value) for value in range(14)]},
            index=target_index,
        )

        features = build_macro_features(macro_data, target_index)

        self.assertIn("macro_high_vix_regime", features.columns)
        self.assertTrue(features["macro_high_vix_regime"].iloc[:11].isna().all())
        self.assertEqual(features["macro_high_vix_regime"].iloc[-1], 1.0)

    def test_dollar_regime_is_created_when_dxy_exists(self):
        target_index = pd.date_range("2024-01-05", periods=14, freq="W-FRI")
        macro_data = pd.DataFrame(
            {"DXY": [100.0 + value for value in range(14)]},
            index=target_index,
        )

        features = build_macro_features(macro_data, target_index)

        self.assertIn("macro_dollar_momentum_12p", features.columns)
        self.assertIn("macro_strong_dollar_regime", features.columns)
        self.assertTrue(features["macro_strong_dollar_regime"].iloc[:12].isna().all())
        self.assertGreater(features["macro_dollar_momentum_12p"].iloc[-1], 0.0)
        self.assertEqual(features["macro_strong_dollar_regime"].iloc[-1], 1.0)

    def test_cpi_regime_is_created_when_cpi_exists(self):
        target_index = pd.date_range("2024-01-05", periods=14, freq="W-FRI")
        macro_data = pd.DataFrame(
            {"CPI": [300.0 + value for value in range(14)]},
            index=target_index,
        )

        features = build_macro_features(macro_data, target_index)

        self.assertIn("macro_cpi_momentum_12p", features.columns)
        self.assertIn("macro_inflation_pressure_regime", features.columns)
        self.assertTrue(features["macro_inflation_pressure_regime"].iloc[:12].isna().all())
        self.assertGreater(features["macro_cpi_momentum_12p"].iloc[-1], 0.0)
        self.assertEqual(features["macro_inflation_pressure_regime"].iloc[-1], 1.0)

    def test_empty_macro_data_raises_value_error(self):
        macro_data = pd.DataFrame(index=pd.DatetimeIndex([]))

        with self.assertRaisesRegex(ValueError, "macro_data must be a non-empty DataFrame"):
            build_features_v3(self.returns, macro_data=macro_data)

    def test_invalid_macro_data_index_raises_type_error(self):
        macro_data = pd.DataFrame({"VIX": [20.0]}, index=["2024-01-05"])

        with self.assertRaisesRegex(TypeError, "macro_data index"):
            build_features_v3(self.returns, macro_data=macro_data)

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
