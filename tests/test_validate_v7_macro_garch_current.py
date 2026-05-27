"""Tests for V7 macro plus GARCH validation."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.validate_v7_macro_garch_current import (
    validate_alignment_checks,
    validate_v7_macro_garch_current,
)


class ValidateV7MacroGarchCurrentTests(unittest.TestCase):
    def test_validation_smoke_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = Path(temp_dir) / "returns.csv"
            macro_path = Path(temp_dir) / "macro.csv"
            output_dir = Path(temp_dir) / "out"
            returns = self._returns()
            self._write_returns(returns, returns_path)
            self._write_macro(returns.index, macro_path)

            result = validate_v7_macro_garch_current(
                returns_path=str(returns_path),
                macro_path=str(macro_path),
                output_dir=str(output_dir),
            )

            self.assertTrue((output_dir / "v7_macro_garch_current_coverage.csv").exists())
            self.assertTrue((output_dir / "v7_macro_garch_current_feature_summary.csv").exists())
            self.assertTrue((output_dir / "v7_macro_garch_current_alignment_checks.csv").exists())
            self.assertTrue((output_dir / "v7_macro_garch_current_summary.md").exists())
            self.assertTrue((result["alignment_checks"]["status"] == "pass").all())

    def test_validation_reports_macro_and_garch_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = Path(temp_dir) / "returns.csv"
            macro_path = Path(temp_dir) / "macro.csv"
            returns = self._returns()
            self._write_returns(returns, returns_path)
            self._write_macro(returns.index, macro_path)

            result = validate_v7_macro_garch_current(
                returns_path=str(returns_path),
                macro_path=str(macro_path),
                output_dir=str(Path(temp_dir) / "out"),
            )

            summary = result["feature_summary"].iloc[0]
            self.assertGreater(int(summary["n_macro_features"]), 0)
            self.assertGreater(int(summary["n_garch_features"]), 0)
            self.assertEqual(int(summary["cash_garch_column_count"]), 0)

    def test_stale_macro_coverage_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = Path(temp_dir) / "returns.csv"
            macro_path = Path(temp_dir) / "macro.csv"
            returns = self._returns()
            self._write_returns(returns, returns_path)
            self._write_macro(returns.index[:-5], macro_path)

            with self.assertRaisesRegex(ValueError, "stale"):
                validate_v7_macro_garch_current(
                    returns_path=str(returns_path),
                    macro_path=str(macro_path),
                    output_dir=str(Path(temp_dir) / "out"),
                )

    def test_failed_alignment_checks_raise(self):
        checks = pd.DataFrame(
            [{"check_name": "bad", "status": "fail", "detail": "nope"}]
        )

        with self.assertRaisesRegex(ValueError, "bad"):
            validate_alignment_checks(checks)

    @staticmethod
    def _returns() -> pd.DataFrame:
        index = pd.date_range("2021-01-01", periods=160, freq="W-FRI")
        return pd.DataFrame(
            {
                "SPY": ([0.010, 0.015, -0.008, 0.012, -0.020] * 32)[: len(index)],
                "GLD": ([0.004, -0.002, 0.006, 0.001, 0.012] * 32)[: len(index)],
                "TLT": ([-0.003, 0.004, 0.005, -0.006, 0.011] * 32)[: len(index)],
                "BTC-USD": ([0.040, -0.030, 0.050, -0.020, 0.060] * 32)[: len(index)],
                "CASH": [0.0] * len(index),
            },
            index=index,
        )

    @staticmethod
    def _write_returns(returns: pd.DataFrame, path: Path) -> None:
        frame = returns.reset_index().rename(columns={"index": "date"})
        frame.to_csv(path, index=False)

    @staticmethod
    def _write_macro(index: pd.DatetimeIndex, path: Path) -> None:
        pd.DataFrame(
            {
                "date": index,
                "DGS10": [4.0 + i * 0.001 for i in range(len(index))],
                "DGS2": [3.0 + i * 0.001 for i in range(len(index))],
                "VIX": [20.0 + (i % 5) for i in range(len(index))],
                "DXY": [100.0 + i * 0.01 for i in range(len(index))],
                "CPI": [300.0 + i * 0.02 for i in range(len(index))],
            }
        ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
