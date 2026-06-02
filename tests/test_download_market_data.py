"""Tests for standalone market-data processing helpers."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.download_market_data import (
    build_cash_proxy_metadata,
    compute_weekly_returns_from_prices,
    write_market_data_outputs,
)


class DownloadMarketDataTests(unittest.TestCase):
    def test_zero_cash_model_produces_zero_cash_returns(self):
        price_data = {
            "SPY": self._price_frame([100.0, 101.0, 102.0]),
            "BIL": self._price_frame([90.0, 90.1, 90.2]),
            "CASH": pd.DataFrame(),
        }

        returns = compute_weekly_returns_from_prices(
            price_data,
            start_date="2024-01-01",
            end_date="2024-01-31",
            cash_return_model="zero",
        )

        self.assertIn("CASH", returns.columns)
        self.assertTrue((returns["CASH"] == 0.0).all())

    def test_bil_proxy_model_maps_bil_to_cash_without_output_bil_column(self):
        price_data = {
            "SPY": self._price_frame([100.0, 101.0, 102.0]),
            "BIL": self._price_frame([90.0, 90.1, 90.3]),
            "CASH": pd.DataFrame(),
        }

        returns = compute_weekly_returns_from_prices(
            price_data,
            start_date="2024-01-01",
            end_date="2024-01-31",
            cash_return_model="bil_proxy",
            cash_proxy_asset="BIL",
        )

        self.assertIn("CASH", returns.columns)
        self.assertNotIn("BIL", returns.columns)
        self.assertGreater((returns["CASH"] != 0.0).sum(), 0)
        self.assertTrue(returns.index.equals(returns.dropna().index))

    def test_write_market_data_outputs_records_cash_proxy_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "returns.csv"
            metadata_path = Path(temp_dir) / "metadata.json"
            with patch(
                "scripts.download_market_data.download_asset_price_data",
                side_effect=lambda asset, start_date, end_date: self._price_frame(
                    [90.0, 90.1, 90.3]
                    if asset == "BIL"
                    else [100.0, 101.0, 102.0]
                ).reset_index(names="date"),
            ):
                result = write_market_data_outputs(
                    assets=("SPY", "CASH"),
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    raw_dir=str(Path(temp_dir) / "raw"),
                    output_path=str(output_path),
                    cash_return_model="bil_proxy",
                    cash_proxy_asset="BIL",
                    metadata_output_path=str(metadata_path),
                )

            self.assertTrue(output_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(result["metadata"]["cash_return_model"], "bil_proxy")
            self.assertEqual(result["metadata"]["cash_proxy_asset"], "BIL")
            self.assertEqual(result["metadata"]["cash_transaction_cost_bps_default"], 2.0)
            self.assertEqual(
                result["metadata"]["recommended_asset_transaction_cost_bps"]["CASH"],
                2.0,
            )
            self.assertEqual(
                result["metadata"]["recommended_asset_transaction_cost_bps"]["BTC-USD"],
                10.0,
            )
            self.assertEqual(
                result["metadata"]["recommended_asset_transaction_cost_bps"]["SPY"],
                2.0,
            )
            self.assertIn("BIL", result["raw_paths"])
            self.assertNotIn("BIL", result["returns"].columns)
            self.assertIn("CASH", result["returns"].columns)

    def test_zero_cash_metadata_uses_frictionless_cash_cost(self):
        returns = pd.DataFrame(
            {"SPY": [0.01], "CASH": [0.0]},
            index=pd.to_datetime(["2024-01-05"]),
        )

        metadata = build_cash_proxy_metadata(
            returns=returns,
            assets=("SPY", "CASH"),
            output_path="returns.csv",
            cash_return_model="zero",
            cash_proxy_asset="BIL",
            raw_paths={},
        )

        self.assertEqual(metadata["cash_return_model"], "zero")
        self.assertIsNone(metadata["cash_proxy_asset"])
        self.assertEqual(metadata["cash_transaction_cost_bps_default"], 0.0)
        self.assertEqual(
            metadata["recommended_asset_transaction_cost_bps"]["CASH"],
            0.0,
        )

    def test_build_cash_proxy_metadata_documents_zero_and_robustness_protocols(self):
        returns = pd.DataFrame(
            {"SPY": [0.01], "CASH": [0.001]},
            index=pd.to_datetime(["2024-01-05"]),
        )

        metadata = build_cash_proxy_metadata(
            returns=returns,
            assets=("SPY", "CASH"),
            output_path="returns.csv",
            cash_return_model="bil_proxy",
            cash_proxy_asset="BIL",
            raw_paths={"BIL": "raw/BIL.csv"},
        )

        self.assertEqual(metadata["cash_return_model"], "bil_proxy")
        self.assertEqual(metadata["cash_proxy_asset"], "BIL")
        self.assertIn("zero-return synthetic", metadata["main_protocol_cash_note"])
        self.assertIn("BIL short-term Treasury ETF proxy", metadata["robustness_protocol_cash_note"])
        self.assertIn("ETF-like", metadata["robustness_protocol_cash_note"])

    @staticmethod
    def _price_frame(values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {"close": values},
            index=pd.date_range("2024-01-05", periods=len(values), freq="W-FRI"),
        )


if __name__ == "__main__":
    unittest.main()
