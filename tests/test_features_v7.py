"""Tests for Feature Set V7 macro plus real GARCH construction."""

import unittest

import pandas as pd

from src.data.features_v7 import build_features_v7


class FeatureSetV7Tests(unittest.TestCase):
    def setUp(self):
        self.returns = self._returns()
        self.macro = self._macro(self.returns.index)

    def test_v7_includes_macro_and_fitted_garch_columns(self):
        features, diagnostics = build_features_v7(
            self.returns,
            macro_data=self.macro,
            garch_min_history=8,
            garch_window=12,
            return_garch_diagnostics=True,
        )

        self.assertTrue(any(column.startswith("macro_") for column in features.columns))
        self.assertIn("garch_vol_SPY", features.columns)
        self.assertTrue(bool(diagnostics["arch_available"].any()))
        fitted = diagnostics[diagnostics["status"] == "fitted"]
        self.assertEqual({"arch_model"}, set(fitted["backend"]))

    def test_v7_excludes_cash_fitted_garch_columns(self):
        features = build_features_v7(
            self.returns,
            macro_data=self.macro,
            garch_min_history=8,
            garch_window=12,
            garch_exclude_cash=True,
        )

        self.assertNotIn("garch_vol_CASH", features.columns)
        self.assertFalse(
            any(
                "CASH" in column
                for column in features.columns
                if isinstance(column, str) and column.startswith("garch_")
            )
        )

    def test_v7_has_no_missing_aligned_features(self):
        features = build_features_v7(
            self.returns,
            macro_data=self.macro,
            garch_min_history=8,
            garch_window=12,
        )
        shifted = features.shift(1).dropna()
        aligned = shifted.loc[shifted.index.intersection(self.returns.index)]

        self.assertFalse(aligned.empty)
        self.assertEqual(int(aligned.isna().sum().sum()), 0)

    def test_v7_requires_macro_data(self):
        with self.assertRaisesRegex(ValueError, "macro_data"):
            build_features_v7(
                self.returns,
                macro_data=None,
                garch_min_history=8,
                garch_window=12,
            )

    @staticmethod
    def _returns() -> pd.DataFrame:
        index = pd.date_range("2022-01-07", periods=40, freq="W-FRI")
        return pd.DataFrame(
            {
                "SPY": ([0.010, 0.015, -0.008, 0.012, -0.020] * 8)[: len(index)],
                "GLD": ([0.004, -0.002, 0.006, 0.001, 0.012] * 8)[: len(index)],
                "TLT": ([-0.003, 0.004, 0.005, -0.006, 0.011] * 8)[: len(index)],
                "BTC-USD": ([0.040, -0.030, 0.050, -0.020, 0.060] * 8)[: len(index)],
                "CASH": [0.0] * len(index),
            },
            index=index,
        )

    @staticmethod
    def _macro(index: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "DGS10": [4.0 + i * 0.001 for i in range(len(index))],
                "DGS2": [3.0 + i * 0.001 for i in range(len(index))],
                "VIX": [20.0 + (i % 5) for i in range(len(index))],
                "DXY": [100.0 + i * 0.01 for i in range(len(index))],
                "CPI": [300.0 + i * 0.02 for i in range(len(index))],
            },
            index=index,
        )


if __name__ == "__main__":
    unittest.main()
