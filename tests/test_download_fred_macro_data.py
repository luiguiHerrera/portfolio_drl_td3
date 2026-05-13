"""Tests for the standalone FRED macro acquisition script helpers."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.download_fred_macro_data import (
    build_fred_csv_url,
    build_processed_macro_after_download,
    normalize_fred_csv,
)


class DownloadFredMacroDataTests(unittest.TestCase):
    def test_build_fred_csv_url_contains_series_id_and_dates(self):
        url = build_fred_csv_url("DGS10", "2015-01-01", "2024-12-31")

        self.assertIn("id=DGS10", url)
        self.assertIn("observation_start=2015-01-01", url)
        self.assertIn("observation_end=2024-12-31", url)
        self.assertTrue(url.startswith("https://fred.stlouisfed.org/graph/fredgraph.csv"))

    def test_normalize_fred_csv_handles_standard_payload_and_drops_missing(self):
        raw_csv = (
            "observation_date,DGS10\n"
            "2015-01-02,2.12\n"
            "2015-01-05,.\n"
            "2015-01-06,2.10\n"
        )

        result = normalize_fred_csv(raw_csv, "DGS10")

        self.assertEqual(list(result.columns), ["date", "value"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["date"], pd.Timestamp("2015-01-02"))
        self.assertEqual(result.iloc[0]["value"], 2.12)
        self.assertEqual(result.iloc[1]["date"], pd.Timestamp("2015-01-06"))
        self.assertEqual(result.iloc[1]["value"], 2.10)

    def test_normalize_fred_csv_raises_key_error_for_missing_series_column(self):
        raw_csv = "observation_date,DGS2\n2015-01-02,0.67\n"

        with self.assertRaises(KeyError):
            normalize_fred_csv(raw_csv, "DGS10")

    def test_normalize_fred_csv_raises_value_error_when_no_usable_rows_remain(self):
        raw_csv = "observation_date,DGS10\n2015-01-02,.\nnot-a-date,2.10\n"

        with self.assertRaises(ValueError):
            normalize_fred_csv(raw_csv, "DGS10")

    def test_build_processed_macro_after_download_uses_local_raw_fixture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "macro_weekly.csv"

            result = build_processed_macro_after_download(
                raw_macro_dir="tests/fixtures/raw_macro",
                output_path=str(output_path),
                start_date="2020-01-01",
                end_date="2024-12-31",
                cpi_lag_weeks=4,
            )

            saved = pd.read_csv(output_path)

        self.assertFalse(result.empty)
        self.assertEqual(output_path.name, "macro_weekly.csv")
        self.assertIn("date", saved.columns)
        self.assertEqual(
            list(result.columns),
            ["DGS10", "DGS2", "VIX", "DXY", "CPI"],
        )
        self.assertEqual(int(result.isna().sum().sum()), 0)


if __name__ == "__main__":
    unittest.main()
