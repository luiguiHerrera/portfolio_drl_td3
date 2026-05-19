"""Tests for deterministic GARCH-style volatility features."""

import math
import unittest

import pandas as pd

from src.data.garch_features import (
    build_garch_feature_set,
    build_garch_relative_features,
    build_garch_volatility_features,
    compute_garch_volatility_series,
    validate_garch_parameters,
)


class GarchFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.returns = self._returns()

    def test_validate_garch_parameters_accepts_valid_params(self):
        validate_garch_parameters(
            omega=1e-6,
            alpha=0.05,
            beta=0.90,
            periods_per_year=52,
        )

    def test_validate_garch_parameters_rejects_non_positive_omega(self):
        with self.assertRaisesRegex(ValueError, "omega"):
            validate_garch_parameters(0.0, 0.05, 0.90, 52)

    def test_validate_garch_parameters_rejects_non_stationary_alpha_beta(self):
        with self.assertRaisesRegex(ValueError, "alpha \\+ beta"):
            validate_garch_parameters(1e-6, 0.50, 0.50, 52)

    def test_validate_garch_parameters_rejects_bool_inputs(self):
        invalid_arguments = (
            {"omega": True, "alpha": 0.05, "beta": 0.90, "periods_per_year": 52},
            {"omega": 1e-6, "alpha": False, "beta": 0.90, "periods_per_year": 52},
            {"omega": 1e-6, "alpha": 0.05, "beta": True, "periods_per_year": 52},
            {"omega": 1e-6, "alpha": 0.05, "beta": 0.90, "periods_per_year": True},
        )

        for kwargs in invalid_arguments:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    validate_garch_parameters(**kwargs)

    def test_compute_garch_volatility_series_returns_same_index_length(self):
        volatility = compute_garch_volatility_series(self.returns["SPY"])

        self.assertEqual(volatility.index.tolist(), self.returns.index.tolist())
        self.assertEqual(len(volatility), len(self.returns))

    def test_compute_garch_volatility_series_uses_lagged_returns(self):
        returns = pd.Series(
            [0.0, 1.0, 0.0],
            index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
            name="SPY",
        )
        volatility = compute_garch_volatility_series(
            returns,
            omega=1e-6,
            alpha=0.05,
            beta=0.90,
            periods_per_year=1,
            annualize=False,
        )
        initial_sigma2 = 1e-6 / (1.0 - 0.05 - 0.90)
        expected_second_sigma2 = 1e-6 + 0.05 * returns.iloc[0] ** 2 + 0.90 * initial_sigma2
        expected_third_sigma2 = 1e-6 + 0.05 * returns.iloc[1] ** 2 + 0.90 * expected_second_sigma2

        self.assertAlmostEqual(volatility.iloc[1], math.sqrt(expected_second_sigma2))
        self.assertAlmostEqual(volatility.iloc[2], math.sqrt(expected_third_sigma2))
        self.assertGreater(volatility.iloc[2], volatility.iloc[1])

    def test_compute_garch_volatility_series_first_value_is_unconditional_volatility(self):
        volatility = compute_garch_volatility_series(
            self.returns["SPY"],
            omega=1e-6,
            alpha=0.05,
            beta=0.90,
            periods_per_year=52,
            annualize=True,
        )
        expected = math.sqrt(1e-6 / (1.0 - 0.05 - 0.90)) * math.sqrt(52)

        self.assertAlmostEqual(volatility.iloc[0], expected)

    def test_compute_garch_volatility_series_raises_on_missing_returns(self):
        returns = self.returns["SPY"].copy()
        returns.iloc[1] = None

        with self.assertRaisesRegex(ValueError, "missing"):
            compute_garch_volatility_series(returns)

    def test_build_garch_volatility_features_creates_one_column_per_asset(self):
        features = build_garch_volatility_features(self.returns)

        self.assertEqual(
            features.columns.tolist(),
            ["garch_vol_SPY", "garch_vol_TLT", "garch_vol_GLD", "garch_vol_CASH"],
        )

    def test_build_garch_volatility_features_respects_selected_assets(self):
        features = build_garch_volatility_features(self.returns, assets=["SPY", "GLD"])

        self.assertEqual(features.columns.tolist(), ["garch_vol_SPY", "garch_vol_GLD"])

    def test_build_garch_volatility_features_raises_on_missing_asset(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            build_garch_volatility_features(self.returns, assets=["SPY", "BTC-USD"])

    def test_build_garch_relative_features_creates_ratio_columns(self):
        absolute = build_garch_volatility_features(self.returns)
        relative = build_garch_relative_features(absolute, market_asset="SPY")

        self.assertIn("garch_vol_ratio_TLT_vs_SPY", relative.columns)
        self.assertIn("garch_vol_ratio_SPY_vs_SPY", relative.columns)
        self.assertTrue((relative["garch_vol_ratio_SPY_vs_SPY"] == 1.0).all())

    def test_build_garch_relative_features_creates_rank_columns(self):
        absolute = pd.DataFrame(
            {
                "garch_vol_SPY": [0.2],
                "garch_vol_TLT": [0.1],
                "garch_vol_GLD": [0.3],
            },
            index=pd.date_range("2024-01-05", periods=1, freq="W-FRI"),
        )

        relative = build_garch_relative_features(absolute, market_asset="SPY")

        self.assertEqual(relative.loc[absolute.index[0], "garch_vol_rank_TLT"], 1.0)
        self.assertEqual(relative.loc[absolute.index[0], "garch_vol_rank_SPY"], 2.0)
        self.assertEqual(relative.loc[absolute.index[0], "garch_vol_rank_GLD"], 3.0)

    def test_build_garch_relative_features_raises_when_market_vol_column_missing(self):
        absolute = build_garch_volatility_features(self.returns, assets=["TLT", "GLD"])

        with self.assertRaisesRegex(ValueError, "Market volatility column"):
            build_garch_relative_features(absolute, market_asset="SPY")

    def test_build_garch_relative_features_raises_when_market_vol_non_positive(self):
        absolute = build_garch_volatility_features(self.returns)
        absolute.loc[absolute.index[0], "garch_vol_SPY"] = 0.0

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            build_garch_relative_features(absolute, market_asset="SPY")

    def test_build_garch_feature_set_returns_absolute_and_relative_features(self):
        features = build_garch_feature_set(self.returns, market_asset="SPY")

        self.assertIn("garch_vol_SPY", features.columns)
        self.assertIn("garch_vol_ratio_TLT_vs_SPY", features.columns)
        self.assertIn("garch_vol_rank_GLD", features.columns)

    def test_input_dataframes_are_not_mutated(self):
        returns = self.returns.copy(deep=True)
        absolute = build_garch_volatility_features(returns)
        absolute_original = absolute.copy(deep=True)

        build_garch_feature_set(returns, market_asset="SPY")
        build_garch_relative_features(absolute, market_asset="SPY")

        pd.testing.assert_frame_equal(returns, self.returns)
        pd.testing.assert_frame_equal(absolute, absolute_original)

    def test_empty_returns_raises_value_error(self):
        empty = pd.DataFrame(index=pd.DatetimeIndex([]))

        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_garch_volatility_features(empty)

    @staticmethod
    def _returns() -> pd.DataFrame:
        index = pd.date_range("2024-01-05", periods=8, freq="W-FRI")

        return pd.DataFrame(
            {
                "SPY": [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.0, 0.01],
                "TLT": [0.002, 0.003, -0.001, 0.004, 0.002, -0.002, 0.001, 0.0],
                "GLD": [0.005, 0.004, 0.006, -0.003, 0.002, 0.001, 0.003, -0.001],
                "CASH": [0.0] * len(index),
            },
            index=index,
        )


if __name__ == "__main__":
    unittest.main()
