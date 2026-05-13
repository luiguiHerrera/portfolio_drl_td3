"""Tests for local macro CSV loading."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.macro_loader import load_macro_data_from_csv


class MacroLoaderTests(unittest.TestCase):
    def test_loads_valid_csv_and_returns_datetime_index(self):
        with self._temporary_csv("date,VIX\n2024-01-05,20\n") as path:
            result = load_macro_data_from_csv(path)

        self.assertIsInstance(result.index, pd.DatetimeIndex)
        self.assertEqual(result.loc[pd.Timestamp("2024-01-05"), "VIX"], 20)

    def test_sorts_dates_ascending(self):
        with self._temporary_csv("date,VIX\n2024-01-12,25\n2024-01-05,20\n") as path:
            result = load_macro_data_from_csv(path)

        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertEqual(list(result["VIX"]), [20, 25])

    def test_drops_duplicate_dates_keeping_last(self):
        with self._temporary_csv("date,VIX\n2024-01-05,20\n2024-01-05,22\n") as path:
            result = load_macro_data_from_csv(path)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[pd.Timestamp("2024-01-05"), "VIX"], 22)

    def test_converts_numeric_columns_correctly(self):
        with self._temporary_csv("date,VIX,DGS10\n2024-01-05,20.5,4.1\n") as path:
            result = load_macro_data_from_csv(path)

        self.assertEqual(result.loc[pd.Timestamp("2024-01-05"), "VIX"], 20.5)
        self.assertEqual(result.loc[pd.Timestamp("2024-01-05"), "DGS10"], 4.1)

    def test_drops_fully_non_numeric_macro_columns(self):
        with self._temporary_csv(
            "date,VIX,label\n2024-01-05,20,risk_on\n2024-01-12,25,risk_off\n"
        ) as path:
            result = load_macro_data_from_csv(path)

        self.assertIn("VIX", result.columns)
        self.assertNotIn("label", result.columns)

    def test_missing_path_raises_file_not_found_error(self):
        missing_path = "/tmp/portfolio_drl_td3_missing_macro.csv"

        with self.assertRaises(FileNotFoundError):
            load_macro_data_from_csv(missing_path)

    def test_missing_date_column_raises_key_error(self):
        with self._temporary_csv("timestamp,VIX\n2024-01-05,20\n") as path:
            with self.assertRaises(KeyError):
                load_macro_data_from_csv(path)

    def test_no_usable_macro_columns_raise_value_error(self):
        with self._temporary_csv("date,label\n2024-01-05,risk_on\n") as path:
            with self.assertRaisesRegex(ValueError, "no usable macro columns"):
                load_macro_data_from_csv(path)

    def test_does_not_forward_fill_missing_macro_values(self):
        with self._temporary_csv("date,VIX\n2024-01-05,20\n2024-01-12,\n") as path:
            result = load_macro_data_from_csv(path)

        self.assertTrue(pd.isna(result.loc[pd.Timestamp("2024-01-12"), "VIX"]))

    def _temporary_csv(self, content: str):
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "macro.csv"
        path.write_text(content, encoding="utf-8")

        class TemporaryCsv:
            def __enter__(self_inner):
                return str(path)

            def __exit__(self_inner, exc_type, exc_value, traceback):
                temp_dir.cleanup()
                return False

        return TemporaryCsv()


if __name__ == "__main__":
    unittest.main()
