import tempfile
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from src.data.build_realtime_macro_dataset import (
    SERIES_CONFIGS,
    build_realtime_macro_dataset,
    fetch_current_vintage_observation_asof_fallback,
    fetch_fred_vintage_observations,
    fetch_fred_weekly_asof_values,
    select_asof_weekly_values,
)


class BuildRealtimeMacroDatasetTests(unittest.TestCase):
    def test_select_asof_weekly_values_uses_only_available_vintage(self):
        records = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(
                    ["2015-01-02", "2015-01-09", "2015-01-09"]
                ),
                "value": [1.0, 2.0, 3.0],
                "realtime_start": pd.to_datetime(
                    ["2015-01-02", "2015-01-10", "2015-01-20"]
                ),
                "realtime_end": pd.to_datetime(
                    ["2262-04-11", "2015-01-19", "2262-04-11"]
                ),
                "realtime_end_raw": ["9999-12-31", "2015-01-19", "9999-12-31"],
            }
        )
        dates = pd.to_datetime(["2015-01-09", "2015-01-16", "2015-01-23"])

        selected = select_asof_weekly_values(
            records,
            pd.DatetimeIndex(dates),
            SERIES_CONFIGS[0],
            "local_raw_vintage",
        )

        self.assertEqual(selected.loc[0, "value"], 1.0)
        self.assertEqual(selected.loc[1, "value"], 2.0)
        self.assertEqual(selected.loc[2, "value"], 3.0)
        self.assertTrue((selected["observation_date_used"] <= selected["date"]).all())
        self.assertTrue((selected["realtime_start_used"] <= selected["date"]).all())

    def test_build_from_local_raw_vintage_writes_macro_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            returns_path = root / "returns.csv"
            raw_dir = root / "raw"
            output_path = root / "macro.csv"
            metadata_path = root / "metadata.csv"
            raw_dir.mkdir()
            _write_returns(returns_path)
            for config in SERIES_CONFIGS:
                _write_vintage_file(raw_dir / f"{config.series_id}.csv")

            result = build_realtime_macro_dataset(
                returns_path=str(returns_path),
                output_path=str(output_path),
                metadata_output=str(metadata_path),
                raw_vintage_dir=str(raw_dir),
                api_key=None,
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(list(result["macro"].columns), ["DGS10", "DGS2", "VIX", "DXY", "CPI"])
            self.assertEqual(int(result["macro"].isna().sum().sum()), 0)
            self.assertEqual(set(result["metadata"]["source"]), {"local_raw_vintage"})

    def test_missing_api_key_without_raw_vintage_fails_clearly(self):
        with patch.dict(os.environ, {"FRED_API_KEY": ""}, clear=False):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                returns_path = root / "returns.csv"
                _write_returns(returns_path)

                with self.assertRaisesRegex(RuntimeError, "FRED_API_KEY"):
                    build_realtime_macro_dataset(
                        returns_path=str(returns_path),
                        output_path=str(root / "macro.csv"),
                        metadata_output=str(root / "metadata.csv"),
                        raw_vintage_dir=str(root / "missing_raw"),
                        api_key=None,
                    )

    def test_fetch_fred_vintage_observations_uses_output_type_two(self):
        vintage_response = Mock()
        vintage_response.json.return_value = {
            "count": 1,
            "vintage_dates": ["2015-01-09"],
        }
        vintage_response.raise_for_status.return_value = None
        observation_response = Mock()
        observation_response.json.return_value = {
            "count": 1,
            "observations": [
                {
                    "realtime_start": "2015-01-02",
                    "realtime_end": "9999-12-31",
                    "date": "2015-01-02",
                    "value": "1.0",
                }
            ],
        }
        observation_response.raise_for_status.return_value = None
        session = Mock()
        session.get.side_effect = [vintage_response, observation_response]

        result = fetch_fred_vintage_observations(
            series_id="DGS10",
            api_key="dummy",
            observation_start="2015-01-01",
            observation_end="2015-01-31",
            session=session,
        )

        params = session.get.call_args_list[-1].kwargs["params"]
        self.assertEqual(params["output_type"], 2)
        self.assertEqual(params["vintage_dates"], "2015-01-09")
        self.assertIn("realtime_start", result.columns)
        self.assertEqual(float(result.iloc[0]["value"]), 1.0)

    def test_fetch_error_omits_api_key_from_message(self):
        session = Mock()
        response = Mock()
        response.status_code = 400
        response.text = "bad request super-secret-key"
        session.get.side_effect = requests.HTTPError("network failed", response=response)

        with self.assertRaisesRegex(RuntimeError, "API key are intentionally omitted") as context:
            fetch_fred_vintage_observations(
                series_id="DGS10",
                api_key="super-secret-key",
                observation_start="2015-01-01",
                observation_end="2015-01-31",
                session=session,
            )

        self.assertNotIn("super-secret-key", str(context.exception))
        self.assertIn("[redacted]", str(context.exception))

    def test_fetch_weekly_asof_values_uses_realtime_date_and_latest_numeric(self):
        response = Mock()
        response.json.return_value = {
            "count": 2,
            "observations": [
                {
                    "date": "2015-01-15",
                    "DGS10_20150116": "1.7",
                    "DGS10_20150123": "1.8",
                },
                {
                    "date": "2015-01-22",
                    "DGS10_20150116": ".",
                    "DGS10_20150123": "1.9",
                },
            ]
        }
        response.raise_for_status.return_value = None
        session = Mock()
        session.get.return_value = response

        result = fetch_fred_weekly_asof_values(
            config=SERIES_CONFIGS[0],
            api_key="dummy",
            weekly_dates=pd.DatetimeIndex(
                [pd.Timestamp("2015-01-16"), pd.Timestamp("2015-01-23")]
            ),
            observation_start="2013-01-16",
            session=session,
        )

        params = session.get.call_args.kwargs["params"]
        self.assertEqual(params["vintage_dates"], "2015-01-16,2015-01-23")
        self.assertEqual(params["output_type"], 2)
        self.assertEqual(result.loc[0, "value"], 1.7)
        self.assertEqual(result.loc[1, "value"], 1.9)
        self.assertEqual(result.loc[0, "vintage_method"], "fred_api_asof")

    def test_current_vintage_fallback_is_explicitly_flagged(self):
        response = Mock()
        response.json.return_value = {
            "observations": [
                {"date": "2015-01-02", "value": "100.0"},
                {"date": "2015-01-16", "value": "101.0"},
            ]
        }
        response.raise_for_status.return_value = None
        session = Mock()
        session.get.return_value = response

        result = fetch_current_vintage_observation_asof_fallback(
            config=SERIES_CONFIGS[3],
            api_key="dummy",
            weekly_dates=pd.DatetimeIndex(
                [pd.Timestamp("2015-01-09"), pd.Timestamp("2015-01-16")]
            ),
            observation_start="2013-01-09",
            fallback_reason="not in ALFRED",
            session=session,
        )

        self.assertEqual(result.loc[0, "value"], 100.0)
        self.assertEqual(result.loc[1, "value"], 101.0)
        self.assertFalse(bool(result.loc[0, "true_vintage_data_available"]))
        self.assertTrue(bool(result.loc[0, "fallback_used"]))
        self.assertEqual(result.loc[0, "source"], "fred_current_vintage_fallback")


def _write_returns(path: Path) -> None:
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-01-09", "2015-01-16", "2015-01-23"]),
            "SPY": [0.01, 0.02, -0.01],
            "TLT": [0.0, 0.01, 0.0],
            "GLD": [0.0, 0.0, 0.01],
            "BTC-USD": [0.03, -0.02, 0.01],
            "CASH": [0.0, 0.0, 0.0],
        }
    ).to_csv(path, index=False)


def _write_vintage_file(path: Path) -> None:
    pd.DataFrame(
        {
            "observation_date": ["2015-01-02", "2015-01-09"],
            "value": [1.0, 2.0],
            "realtime_start": ["2015-01-02", "2015-01-10"],
            "realtime_end": ["9999-12-31", "9999-12-31"],
        }
    ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
