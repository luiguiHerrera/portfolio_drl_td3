"""Tests for V4 rolling-fitted GARCH validation smoke."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.validate_v4_garch_current import (
    synthetic_shock_timing_checks,
    validate_v4_garch_current,
)


class ValidateV4GarchCurrentTests(unittest.TestCase):
    def test_synthetic_shock_timing_checks_pass(self):
        checks = synthetic_shock_timing_checks(min_history=5, window=8)

        self.assertTrue(checks["same_period_unchanged"])
        self.assertTrue(checks["future_changed"])

    def test_validation_smoke_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = Path(temp_dir) / "returns.csv"
            output_dir = Path(temp_dir) / "out"
            self._returns().to_csv(returns_path, index=False)

            result = validate_v4_garch_current(
                returns_path=str(returns_path),
                output_dir=str(output_dir),
                min_history=8,
                window=12,
                exclude_cash=True,
            )

            paths = result["paths"]
            path_exists = {key: Path(path).exists() for key, path in paths.items()}

        self.assertTrue(path_exists["coverage"])
        self.assertTrue(path_exists["feature_summary"])
        self.assertTrue(path_exists["alignment_checks"])
        self.assertTrue(path_exists["fit_diagnostics"])
        self.assertTrue(path_exists["summary"])
        self.assertIn("Fit successes", result["summary"])

    def test_validation_reports_cash_exclusion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = Path(temp_dir) / "returns.csv"
            output_dir = Path(temp_dir) / "out"
            self._returns().to_csv(returns_path, index=False)

            result = validate_v4_garch_current(
                returns_path=str(returns_path),
                output_dir=str(output_dir),
                min_history=8,
                window=12,
                exclude_cash=True,
            )

        self.assertEqual(result["coverage"].loc[0, "cash_handling"], "excluded")
        self.assertEqual(
            int(result["feature_summary"].loc[0, "cash_garch_column_count"]),
            0,
        )

    @staticmethod
    def _returns() -> pd.DataFrame:
        index = pd.date_range("2021-01-01", periods=40, freq="W-FRI")
        rows = []
        for i, date in enumerate(index):
            rows.append(
                {
                    "date": date,
                    "SPY": 0.01 if i % 2 == 0 else -0.008,
                    "TLT": 0.002 if i % 3 else -0.001,
                    "GLD": 0.004 if i % 4 else -0.003,
                    "BTC-USD": 0.03 if i % 2 == 0 else -0.025,
                    "CASH": 0.0,
                }
            )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
