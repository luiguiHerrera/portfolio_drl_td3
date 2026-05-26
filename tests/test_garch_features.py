"""Tests for deterministic GARCH-style volatility features."""

import math
import unittest
from unittest.mock import patch

import pandas as pd

from src.data.garch_features import (
    build_garch_feature_set_by_mode,
    build_garch_feature_set,
    build_garch_relative_features,
    build_garch_volatility_features,
    build_rolling_fitted_garch_volatility_features,
    compute_garch_volatility_series,
    GARCH_MODE_DETERMINISTIC,
    GARCH_MODE_ROLLING_FITTED,
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

    def test_rolling_fitted_garch_excludes_cash_by_default(self):
        features, diagnostics = build_rolling_fitted_garch_volatility_features(
            self._long_returns(),
            min_history=5,
            window=8,
            return_diagnostics=True,
        )

        self.assertNotIn("garch_vol_CASH", features.columns)
        self.assertNotIn("CASH", set(diagnostics["asset"]))

    def test_rolling_fitted_garch_no_same_period_shock_leakage(self):
        returns = self._long_returns()
        shocked = returns.copy()
        shock_date = returns.index[10]
        next_date = returns.index[11]
        shocked.loc[shock_date, "SPY"] = 0.25

        base = build_rolling_fitted_garch_volatility_features(
            returns,
            assets=["SPY"],
            min_history=5,
            window=8,
        )
        with_shock = build_rolling_fitted_garch_volatility_features(
            shocked,
            assets=["SPY"],
            min_history=5,
            window=8,
        )

        self.assertAlmostEqual(
            base.loc[shock_date, "garch_vol_SPY"],
            with_shock.loc[shock_date, "garch_vol_SPY"],
        )
        self.assertNotAlmostEqual(
            base.loc[next_date, "garch_vol_SPY"],
            with_shock.loc[next_date, "garch_vol_SPY"],
        )

    def test_rolling_fitted_garch_uses_realized_fallback_when_fitters_unavailable(self):
        with (
            patch("src.data.garch_features._arch_model", None),
            patch("src.data.garch_features.minimize", None),
        ):
            features, diagnostics = build_rolling_fitted_garch_volatility_features(
                self._long_returns(),
                assets=["SPY"],
                min_history=5,
                window=8,
                return_diagnostics=True,
            )

        self.assertIn("garch_vol_SPY", features.columns)
        self.assertTrue(
            diagnostics["fallback_reason"].astype(str).str.contains("scipy_unavailable").any()
        )

    def test_rolling_fitted_garch_uses_arch_backend_when_available(self):
        features, diagnostics = build_rolling_fitted_garch_volatility_features(
            self._long_returns(),
            assets=["SPY"],
            min_history=5,
            window=8,
            return_diagnostics=True,
        )

        fitted = diagnostics[diagnostics["status"] == "fitted"]
        if bool(diagnostics["arch_available"].any()):
            self.assertFalse(fitted.empty)
            self.assertEqual({"arch_model"}, set(fitted["backend"]))
        self.assertIn("garch_vol_SPY", features.columns)

    def test_rolling_fitted_garch_is_deterministic_for_same_input(self):
        first = build_rolling_fitted_garch_volatility_features(
            self._long_returns(),
            assets=["SPY", "TLT"],
            min_history=5,
            window=8,
        )
        second = build_rolling_fitted_garch_volatility_features(
            self._long_returns(),
            assets=["SPY", "TLT"],
            min_history=5,
            window=8,
        )

        pd.testing.assert_frame_equal(first, second)

    def test_rolling_fitted_mode_differs_from_deterministic_filter(self):
        returns = self._long_returns()
        deterministic = build_garch_feature_set_by_mode(
            returns,
            mode=GARCH_MODE_DETERMINISTIC,
            include_relative=False,
            assets=["SPY"],
        )
        fitted = build_garch_feature_set_by_mode(
            returns,
            mode=GARCH_MODE_ROLLING_FITTED,
            include_relative=False,
            assets=["SPY"],
            min_history=5,
            window=8,
        )
        common_index = fitted.dropna().index.intersection(deterministic.index)
        max_diff = (
            fitted.loc[common_index, "garch_vol_SPY"]
            - deterministic.loc[common_index, "garch_vol_SPY"]
        ).abs().max()

        self.assertGreater(float(max_diff), 0.0)

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

    @staticmethod
    def _long_returns() -> pd.DataFrame:
        index = pd.date_range("2024-01-05", periods=24, freq="W-FRI")
        return pd.DataFrame(
            {
                "SPY": [0.01, -0.02, 0.015, -0.005, 0.012, -0.011] * 4,
                "TLT": [0.002, 0.003, -0.001, 0.004, 0.002, -0.002] * 4,
                "GLD": [0.005, 0.004, 0.006, -0.003, 0.002, 0.001] * 4,
                "CASH": [0.0] * len(index),
            },
            index=index,
        )


if __name__ == "__main__":
    unittest.main()
