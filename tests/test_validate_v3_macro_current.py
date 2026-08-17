"""Tests for current-window V3 macro validation smoke."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.validate_v3_macro_current import (
    build_alignment_checks,
    build_macro_coverage_table,
    validate_macro_coverage,
    validate_v3_macro_current,
)


LOCAL_RETURNS_PATH = Path("data/processed/returns_weekly_latest.csv")
LOCAL_MACRO_PATH = Path("data/processed/macro_weekly_latest.csv")
LOCAL_PROCESSED_DATA_AVAILABLE = (
    LOCAL_RETURNS_PATH.exists() and LOCAL_MACRO_PATH.exists()
)

requires_local_processed_data = unittest.skipUnless(
    LOCAL_PROCESSED_DATA_AVAILABLE,
    "requires local processed returns and macro datasets",
)


class ValidateV3MacroCurrentTests(unittest.TestCase):
    @requires_local_processed_data
    def test_latest_macro_file_covers_latest_returns_end_date(self):
        returns = self._read_dated_csv("data/processed/returns_weekly_latest.csv")
        macro = self._read_dated_csv("data/processed/macro_weekly_latest.csv")

        coverage = build_macro_coverage_table(
            returns,
            macro,
            "data/processed/returns_weekly_latest.csv",
            "data/processed/macro_weekly_latest.csv",
        )

        self.assertTrue(bool(coverage.loc[0, "macro_covers_returns_end"]))
        self.assertTrue(bool(coverage.loc[0, "is_current_window_covered"]))

    def test_stale_macro_coverage_fails_fast(self):
        returns = pd.DataFrame(
            {"SPY": [0.01, 0.02]},
            index=pd.to_datetime(["2026-05-08", "2026-05-15"]),
        )
        macro = pd.DataFrame(
            {"DGS10": [4.0]},
            index=pd.to_datetime(["2026-05-08"]),
        )
        coverage = build_macro_coverage_table(returns, macro, "returns.csv", "macro.csv")

        with self.assertRaisesRegex(ValueError, "stale"):
            validate_macro_coverage(coverage)

    def test_macro_start_after_returns_start_fails_no_backfill_gate(self):
        returns = pd.DataFrame(
            {"SPY": [0.01, 0.02]},
            index=pd.to_datetime(["2026-05-08", "2026-05-15"]),
        )
        macro = pd.DataFrame(
            {"DGS10": [4.0]},
            index=pd.to_datetime(["2026-05-15"]),
        )
        coverage = build_macro_coverage_table(returns, macro, "returns.csv", "macro.csv")

        with self.assertRaisesRegex(ValueError, "starts after returns_start"):
            validate_macro_coverage(coverage)

    @requires_local_processed_data
    def test_validation_smoke_writes_outputs_and_has_no_missing_macro_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_v3_macro_current(
                returns_path="data/processed/returns_weekly_latest.csv",
                macro_path="data/processed/macro_weekly_latest.csv",
                output_dir=temp_dir,
            )

            for path in result["paths"].values():
                self.assertTrue(Path(path).exists())

        summary = result["feature_summary"].iloc[0]
        self.assertEqual(int(summary["missing_aligned_macro_features"]), 0)
        self.assertGreater(int(summary["n_macro_features"]), 0)

    @requires_local_processed_data
    def test_feature_dates_do_not_overrun_returns_dates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_v3_macro_current(
                returns_path="data/processed/returns_weekly_latest.csv",
                macro_path="data/processed/macro_weekly_latest.csv",
                output_dir=temp_dir,
            )

        checks = result["alignment_checks"].set_index("check_name")
        self.assertEqual(checks.loc["features_do_not_overrun_returns", "status"], "pass")

    @requires_local_processed_data
    def test_protocol_validation_and_test_windows_match_when_possible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_v3_macro_current(
                returns_path="data/processed/returns_weekly_latest.csv",
                macro_path="data/processed/macro_weekly_latest.csv",
                output_dir=temp_dir,
            )

        checks = result["alignment_checks"]
        validation_test = checks[
            checks["check_name"].str.contains("_validation_|_test_", regex=True)
        ]
        self.assertFalse(validation_test.empty)
        self.assertTrue((validation_test["status"] == "pass").all())

    def test_alignment_check_detects_macro_backfill_need(self):
        returns = pd.DataFrame(
            {"SPY": [0.01, 0.02]},
            index=pd.to_datetime(["2026-05-08", "2026-05-15"]),
        )
        macro = pd.DataFrame(
            {"DGS10": [4.0]},
            index=pd.to_datetime(["2026-05-15"]),
        )
        aligned_features = pd.DataFrame(
            {"macro_DGS10": [4.0]},
            index=pd.to_datetime(["2026-05-15"]),
        )

        checks = build_alignment_checks(returns, macro, aligned_features)

        status = checks.set_index("check_name").loc[
            "macro_no_backfill_required",
            "status",
        ]
        self.assertEqual(status, "fail")

    @staticmethod
    def _read_dated_csv(path: str) -> pd.DataFrame:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index("date").sort_index()


if __name__ == "__main__":
    unittest.main()
