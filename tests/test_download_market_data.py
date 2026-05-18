"""Tests for the standalone local market data acquisition script helpers."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.download_market_data import (
    compute_weekly_returns_from_prices,
    normalize_price_data,
    write_market_data_outputs,
)


class DownloadMarketDataTests(unittest.TestCase):
    def test_normalize_price_data_accepts_date_close_and_sorts(self):
        raw = pd.DataFrame(
            {
                "date": ["2024-01-08", "2024-01-05"],
                "close": [101.0, 100.0],
            }
        )

        result = normalize_price_data(raw)

        self.assertIsInstance(result.index, pd.DatetimeIndex)
        self.assertEqual(result.index.tolist(), [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")])
        self.assertEqual(result["close"].tolist(), [100.0, 101.0])

    def test_normalize_price_data_drops_invalid_dates_and_close_values(self):
        raw = pd.DataFrame(
            {
                "date": ["2024-01-05", "bad-date", "2024-01-08"],
                "close": ["100.0", "101.0", "bad-close"],
            }
        )

        result = normalize_price_data(raw)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.index[0], pd.Timestamp("2024-01-05"))
        self.assertEqual(result.iloc[0]["close"], 100.0)

    def test_normalize_price_data_deduplicates_dates_keeping_last(self):
        raw = pd.DataFrame(
            {
                "date": ["2024-01-05", "2024-01-05", "2024-01-08"],
                "close": [100.0, 101.0, 102.0],
            }
        )

        result = normalize_price_data(raw)

        self.assertEqual(result.loc[pd.Timestamp("2024-01-05"), "close"], 101.0)

    def test_compute_weekly_returns_from_prices_returns_weekly_friday_index(self):
        result = compute_weekly_returns_from_prices(
            self._price_data_by_asset(),
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        self.assertTrue(all(date.weekday() == 4 for date in result.index))

    def test_compute_weekly_returns_from_prices_computes_pct_change(self):
        result = compute_weekly_returns_from_prices(
            self._price_data_by_asset(),
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        self.assertAlmostEqual(result.iloc[0]["SPY"], 0.10)
        self.assertAlmostEqual(result.iloc[1]["GLD"], 0.05)

    def test_compute_weekly_returns_from_prices_includes_cash_as_zero(self):
        result = compute_weekly_returns_from_prices(
            self._price_data_by_asset(),
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        self.assertIn("CASH", result.columns)
        self.assertTrue((result["CASH"] == 0.0).all())

    def test_compute_weekly_returns_from_prices_clips_rows_after_end_date(self):
        result = compute_weekly_returns_from_prices(
            self._price_data_by_asset(),
            start_date="2024-01-01",
            end_date="2024-01-19",
        )

        self.assertLessEqual(result.index.max(), pd.Timestamp("2024-01-19"))

    def test_compute_weekly_returns_from_prices_raises_for_empty_asset_data(self):
        data = self._price_data_by_asset()
        data["BTC-USD"] = pd.DataFrame()

        with self.assertRaisesRegex(ValueError, "BTC-USD"):
            compute_weekly_returns_from_prices(
                data,
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

    def test_compute_weekly_returns_from_prices_raises_when_processed_returns_empty(self):
        with self.assertRaises(ValueError):
            compute_weekly_returns_from_prices(
                self._price_data_by_asset(periods=1),
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

    def test_mocked_download_path_writes_raw_and_processed_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            output_path = Path(temp_dir) / "returns.csv"

            with patch(
                "scripts.download_market_data.download_asset_price_data",
                side_effect=lambda asset, start_date, end_date: self._asset_prices(asset),
            ):
                result = write_market_data_outputs(
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    raw_dir=str(raw_dir),
                    output_path=str(output_path),
                )

            self.assertTrue(output_path.exists())
            self.assertTrue((raw_dir / "SPY.csv").exists())
            self.assertTrue((raw_dir / "BTC-USD.csv").exists())
            saved = pd.read_csv(output_path)

        self.assertIn("date", saved.columns)
        self.assertIn("BTC-USD", saved.columns)
        self.assertFalse(result["returns"].empty)

    def test_btc_download_failure_raises_clear_value_error(self):
        def fake_download(asset, start_date, end_date):
            if asset == "BTC-USD":
                return pd.DataFrame()
            return self._asset_prices(asset)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "scripts.download_market_data.download_asset_price_data",
                side_effect=fake_download,
            ):
                with self.assertRaisesRegex(ValueError, "BTC-USD"):
                    write_market_data_outputs(
                        start_date="2024-01-01",
                        end_date="2024-01-31",
                        raw_dir=str(Path(temp_dir) / "raw"),
                        output_path=str(Path(temp_dir) / "returns.csv"),
                    )

    def _price_data_by_asset(self, periods=4):
        return {
            "SPY": self._asset_prices("SPY", periods=periods),
            "TLT": self._asset_prices("TLT", periods=periods),
            "GLD": self._asset_prices("GLD", periods=periods),
            "BTC-USD": self._asset_prices("BTC-USD", periods=periods),
            "CASH": pd.DataFrame(),
        }

    def _asset_prices(self, asset, periods=4):
        date_index = pd.date_range("2024-01-05", periods=periods, freq="W-FRI")
        base_prices = {
            "SPY": 100.0,
            "TLT": 80.0,
            "GLD": 200.0,
            "BTC-USD": 40000.0,
        }
        base = base_prices.get(asset, 100.0)
        values = [base * (1.0 + 0.10 * step) for step in range(periods)]
        if asset == "GLD":
            values = [200.0, 210.0, 220.5, 231.525][:periods]

        return pd.DataFrame({"date": date_index, "close": values})


if __name__ == "__main__":
    unittest.main()
