"""Tests for Feature Set V8 EWMA plus real GARCH construction."""

import unittest

import numpy as np
import pandas as pd

from src.data.features_v4 import build_features_v4
from src.data.features_v8 import build_features_v8


class FeatureSetV8Tests(unittest.TestCase):
    def setUp(self):
        self.returns = self._returns()

    def test_v8_includes_garch_ewma_and_comparison_columns(self):
        features, diagnostics = build_features_v8(
            self.returns,
            garch_min_history=8,
            garch_window=12,
            return_diagnostics=True,
        )

        self.assertIn("garch_vol_SPY", features.columns)
        self.assertIn("ewma_vol_SPY", features.columns)
        self.assertIn("garch_minus_ewma_vol_SPY", features.columns)
        self.assertIn("garch_to_ewma_vol_ratio_SPY", features.columns)
        self.assertIn("garch", diagnostics)
        self.assertIn("ewma", diagnostics)

    def test_v8_excludes_cash_volatility_columns(self):
        features = build_features_v8(
            self.returns,
            garch_min_history=8,
            garch_window=12,
            garch_exclude_cash=True,
        )

        volatility_columns = [
            column
            for column in features.columns
            if column.startswith("garch_")
            or column.startswith("ewma_vol_")
            or column.startswith("garch_minus_ewma_")
            or column.startswith("garch_to_ewma_")
        ]
        self.assertFalse(any("CASH" in column for column in volatility_columns))

    def test_v8_has_no_missing_or_infinite_aligned_features(self):
        features = build_features_v8(
            self.returns,
            garch_min_history=8,
            garch_window=12,
        )
        shifted = features.shift(1).dropna()
        aligned = shifted.loc[shifted.index.intersection(self.returns.index)]

        self.assertFalse(aligned.empty)
        self.assertEqual(int(aligned.isna().sum().sum()), 0)
        self.assertTrue(np.isfinite(aligned.to_numpy(dtype=float)).all())

    def test_synthetic_shock_does_not_change_same_period_forecast(self):
        shocked = self.returns.copy()
        shock_date = self.returns.index[20]
        future_date = self.returns.index[21]
        shocked.loc[shock_date, "SPY"] = 0.50

        base = build_features_v8(self.returns, garch_min_history=8, garch_window=12)
        changed = build_features_v8(shocked, garch_min_history=8, garch_window=12)
        columns = [
            "garch_vol_SPY",
            "ewma_vol_SPY",
            "garch_minus_ewma_vol_SPY",
            "garch_to_ewma_vol_ratio_SPY",
        ]

        self.assertTrue(np.allclose(base.loc[shock_date, columns], changed.loc[shock_date, columns]))
        self.assertFalse(np.allclose(base.loc[future_date, columns], changed.loc[future_date, columns]))

    def test_v8_differs_from_v4_feature_set(self):
        v4 = build_features_v4(
            self.returns,
            garch_mode="rolling_fitted",
            garch_min_history=8,
            garch_window=12,
            garch_exclude_cash=True,
        )
        v8 = build_features_v8(
            self.returns,
            garch_min_history=8,
            garch_window=12,
        )

        self.assertNotEqual(set(v4.columns), set(v8.columns))
        self.assertTrue(any(column.startswith("ewma_vol_") for column in v8.columns))

    def test_v8_uses_arch_model_for_fitted_garch_when_available(self):
        _, diagnostics = build_features_v8(
            self.returns,
            garch_min_history=8,
            garch_window=12,
            return_diagnostics=True,
        )
        fitted = diagnostics["garch"][diagnostics["garch"]["status"] == "fitted"]

        self.assertFalse(fitted.empty)
        self.assertEqual({"arch_model"}, set(fitted["backend"]))

    @staticmethod
    def _returns() -> pd.DataFrame:
        index = pd.date_range("2022-01-07", periods=40, freq="W-FRI")
        return pd.DataFrame(
            {
                "SPY": ([0.010, 0.015, -0.008, 0.012, -0.020] * 8)[: len(index)],
                "TLT": ([-0.003, 0.004, 0.005, -0.006, 0.011] * 8)[: len(index)],
                "GLD": ([0.004, -0.002, 0.006, 0.001, 0.012] * 8)[: len(index)],
                "BTC-USD": ([0.040, -0.030, 0.050, -0.020, 0.060] * 8)[: len(index)],
                "CASH": [0.0] * len(index),
            },
            index=index,
        )


if __name__ == "__main__":
    unittest.main()
