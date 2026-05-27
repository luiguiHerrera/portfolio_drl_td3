"""Tests for V8 EWMA/GARCH validation reporting."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.validate_v8_ewma_garch_current import (
    build_alignment_checks,
    synthetic_shock_timing_check,
    validate_v8_ewma_garch_current,
)
from src.data.features_v8 import build_features_v8


class ValidateV8EWMAGARCHCurrentTests(unittest.TestCase):
    def test_validation_smoke_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = Path(temp_dir) / "returns.csv"
            self._returns().reset_index(names="date").to_csv(returns_path, index=False)
            output_dir = Path(temp_dir) / "out"

            result = validate_v8_ewma_garch_current(
                returns_path=str(returns_path),
                output_dir=str(output_dir),
                ewma_lambda=0.94,
                garch_min_history=8,
                garch_window=12,
                exclude_cash=True,
            )

            self.assertTrue(Path(result["paths"]["coverage"]).exists())
            self.assertTrue(Path(result["paths"]["feature_summary"]).exists())
            self.assertTrue(Path(result["paths"]["alignment_checks"]).exists())
            self.assertTrue(Path(result["paths"]["diagnostics"]).exists())
            self.assertTrue(Path(result["paths"]["summary"]).exists())

    def test_alignment_checks_pass_for_valid_v8_features(self):
        returns = self._returns()
        features, diagnostics = build_features_v8(
            returns,
            garch_min_history=8,
            garch_window=12,
            return_diagnostics=True,
        )
        aligned = features.shift(1).dropna()
        aligned = aligned.loc[aligned.index.intersection(returns.index)]

        checks = build_alignment_checks(
            returns=returns,
            raw_features=features,
            aligned_features=aligned,
            diagnostics=diagnostics,
            exclude_cash=True,
        )

        self.assertEqual({"pass"}, set(checks["status"]))

    def test_synthetic_shock_check_detects_lagged_timing(self):
        check = synthetic_shock_timing_check()

        self.assertTrue(check["same_period_unchanged"])
        self.assertTrue(check["future_changed"])

    def test_cash_columns_fail_when_excluded(self):
        returns = self._returns()
        features, diagnostics = build_features_v8(
            returns,
            garch_min_history=8,
            garch_window=12,
            return_diagnostics=True,
        )
        features = features.copy()
        features["ewma_vol_CASH"] = 0.0
        aligned = features.shift(1).dropna()
        aligned = aligned.loc[aligned.index.intersection(returns.index)]

        checks = build_alignment_checks(
            returns=returns,
            raw_features=features,
            aligned_features=aligned,
            diagnostics=diagnostics,
            exclude_cash=True,
        )

        row = checks.set_index("check_name").loc["cash_volatility_excluded"]
        self.assertEqual(row["status"], "fail")

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
