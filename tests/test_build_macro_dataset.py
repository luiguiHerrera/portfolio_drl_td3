"""Tests for local weekly macro dataset construction."""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.build_macro_dataset import (
    align_cpi_to_weekly_with_lag,
    build_weekly_macro_dataset,
)


FIXTURE_DIR = Path("tests/fixtures/raw_macro")


class BuildMacroDatasetTests(unittest.TestCase):
    def test_build_weekly_macro_dataset_returns_non_empty_dataframe(self):
        result = build_weekly_macro_dataset(str(FIXTURE_DIR))

        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    def test_output_columns_are_expected_macro_names(self):
        result = build_weekly_macro_dataset(str(FIXTURE_DIR))

        self.assertEqual(
            list(result.columns),
            ["DGS10", "DGS2", "VIX", "DXY", "CPI"],
        )

    def test_output_index_is_datetime_index_and_weekly_friday(self):
        result = build_weekly_macro_dataset(str(FIXTURE_DIR))

        self.assertIsInstance(result.index, pd.DatetimeIndex)
        self.assertTrue(all(timestamp.weekday() == 4 for timestamp in result.index))

    def test_output_has_no_missing_values_after_final_dropna(self):
        result = build_weekly_macro_dataset(str(FIXTURE_DIR))

        self.assertEqual(int(result.isna().sum().sum()), 0)

    def test_start_date_and_end_date_clipping_are_respected(self):
        result = build_weekly_macro_dataset(
            str(FIXTURE_DIR),
            start_date="2021-01-01",
            end_date="2021-03-31",
        )

        self.assertGreaterEqual(result.index.min(), pd.Timestamp("2021-01-01"))
        self.assertLessEqual(result.index.max(), pd.Timestamp("2021-03-31"))

    def test_output_path_writes_csv_with_date_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "macro" / "weekly.csv"

            build_weekly_macro_dataset(str(FIXTURE_DIR), output_path=str(output_path))
            saved = pd.read_csv(output_path)

        self.assertEqual(output_path.name, "weekly.csv")
        self.assertIn("date", saved.columns)
        self.assertEqual(
            list(saved.columns),
            ["date", "DGS10", "DGS2", "VIX", "DXY", "CPI"],
        )

    def test_cpi_lag_is_applied_before_weekly_alignment(self):
        cpi = pd.Series(
            [100.0],
            index=pd.to_datetime(["2020-01-01"]),
            name="CPIAUCSL",
        )

        result = align_cpi_to_weekly_with_lag(
            cpi,
            weekly_frequency="W-FRI",
            cpi_lag_weeks=4,
        )

        self.assertEqual(result.index.min(), pd.Timestamp("2020-01-31"))
        self.assertEqual(result.iloc[0], 100.0)
        self.assertFalse((result.index < pd.Timestamp("2020-01-29")).any())

    def test_missing_raw_file_raises_file_not_found_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                build_weekly_macro_dataset(temp_dir)

    def test_invalid_cpi_lag_weeks_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_weekly_macro_dataset(str(FIXTURE_DIR), cpi_lag_weeks=-1)

        with self.assertRaises(ValueError):
            build_weekly_macro_dataset(str(FIXTURE_DIR), cpi_lag_weeks=1.5)

    def test_missing_date_or_value_column_raises_key_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            for fixture_path in FIXTURE_DIR.glob("*.csv"):
                shutil.copy(fixture_path, raw_dir / fixture_path.name)

            (raw_dir / "DGS10.csv").write_text(
                "timestamp,value\n2020-01-03,1.8\n",
                encoding="utf-8",
            )

            with self.assertRaises(KeyError):
                build_weekly_macro_dataset(str(raw_dir))


if __name__ == "__main__":
    unittest.main()
