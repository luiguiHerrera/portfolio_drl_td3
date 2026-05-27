"""Tests for lagged EWMA volatility features."""

import unittest

import pandas as pd

from src.data.ewma_features import (
    build_ewma_volatility_features,
    compute_lagged_ewma_volatility_series,
)


class EWMAFeaturesTests(unittest.TestCase):
    def test_ewma_forecast_uses_lagged_returns_only(self):
        index = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
        returns = pd.Series(
            [0.01, 0.01, 0.01, 0.01, 0.50, 0.01, 0.01, 0.01],
            index=index,
            name="SPY",
        )
        shocked = returns.copy()
        shocked.iloc[4] = 0.90

        base = compute_lagged_ewma_volatility_series(returns, ewma_lambda=0.94)
        changed = compute_lagged_ewma_volatility_series(shocked, ewma_lambda=0.94)

        self.assertAlmostEqual(base.iloc[4], changed.iloc[4])
        self.assertNotAlmostEqual(base.iloc[5], changed.iloc[5])

    def test_ewma_features_exclude_cash_by_default(self):
        features = build_ewma_volatility_features(self._returns())

        self.assertIn("ewma_vol_SPY", features.columns)
        self.assertNotIn("ewma_vol_CASH", features.columns)

    def test_ewma_diagnostics_are_returned(self):
        features, diagnostics = build_ewma_volatility_features(
            self._returns(),
            return_diagnostics=True,
        )

        self.assertFalse(features.empty)
        self.assertEqual({"SPY", "TLT"}, set(diagnostics["asset"]))
        self.assertEqual({"weekly"}, set(diagnostics["volatility_unit"]))

    def test_invalid_lambda_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ewma_lambda"):
            compute_lagged_ewma_volatility_series(
                self._returns()["SPY"],
                ewma_lambda=1.0,
            )

    @staticmethod
    def _returns() -> pd.DataFrame:
        index = pd.date_range("2024-01-05", periods=10, freq="W-FRI")
        return pd.DataFrame(
            {
                "SPY": [0.01, -0.01, 0.02, 0.00, 0.015] * 2,
                "TLT": [0.00, 0.01, -0.01, 0.01, 0.005] * 2,
                "CASH": [0.0] * len(index),
            },
            index=index,
        )


if __name__ == "__main__":
    unittest.main()
