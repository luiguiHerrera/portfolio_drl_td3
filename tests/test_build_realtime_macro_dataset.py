import tempfile
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from src.data.build_realtime_macro_dataset import (
    SERIES_CONFIGS,
    build_cpi_yoy_asof_metadata,
    build_realtime_macro_dataset,
    fetch_fred_vintage_observations,
    fetch_fred_weekly_asof_values,
    resolve_series_configs,
    select_asof_weekly_values,
    validate_asof_endpoint_usability,
    validate_vintage_coverage,
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
        self.assertEqual(selected.loc[0, "feature_name"], "DGS10")
        self.assertEqual(selected.loc[0, "source"], "FRED/Federal Reserve H.15")
        self.assertFalse(bool(selected.loc[0, "fallback_used"]))

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
            self.assertEqual(
                list(result["macro"].columns),
                ["DGS10", "DGS2", "VIX", "DXY", "CPI", "cpi_yoy_asof"],
            )
            self.assertEqual(int(result["macro"].isna().sum().sum()), 0)
            self.assertEqual(set(result["metadata"]["vintage_method"]), {"local_raw_vintage"})
            self.assertTrue(result["metadata"]["true_vintage_data_available"].astype(bool).all())
            self.assertFalse(result["metadata"]["fallback_used"].astype(bool).any())
            self.assertIn("transformation_applied", result["metadata"].columns)
            self.assertIn("cpi_yoy_asof", set(result["metadata"]["output_name"]))

    def test_build_can_exclude_dollar_series_for_clean_no_dxy_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            returns_path = root / "returns.csv"
            raw_dir = root / "raw"
            raw_dir.mkdir()
            _write_returns(returns_path)
            for config in SERIES_CONFIGS:
                if config.output_name != "DXY":
                    _write_vintage_file(raw_dir / f"{config.series_id}.csv")

            result = build_realtime_macro_dataset(
                returns_path=str(returns_path),
                output_path=str(root / "macro.csv"),
                metadata_output=str(root / "metadata.csv"),
                raw_vintage_dir=str(raw_dir),
                api_key=None,
                exclude_series=("DXY",),
                require_no_fallback=True,
            )

            self.assertEqual(
                list(result["macro"].columns),
                ["DGS10", "DGS2", "VIX", "CPI", "cpi_yoy_asof"],
            )
            self.assertNotIn("DXY", set(result["metadata"]["output_name"]))
            self.assertIn("cpi_yoy_asof", set(result["metadata"]["output_name"]))

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
        vintage_response = Mock()
        vintage_response.json.return_value = {
            "count": 2,
            "vintage_dates": ["2015-01-16", "2015-01-23"],
        }
        vintage_response.raise_for_status.return_value = None
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
        session.get.side_effect = [vintage_response, response]

        result = fetch_fred_weekly_asof_values(
            config=SERIES_CONFIGS[0],
            api_key="dummy",
            weekly_dates=pd.DatetimeIndex(
                [pd.Timestamp("2015-01-16"), pd.Timestamp("2015-01-23")]
            ),
            observation_start="2013-01-16",
            session=session,
        )

        params = session.get.call_args_list[-1].kwargs["params"]
        self.assertEqual(params["vintage_dates"], "2015-01-16,2015-01-23")
        self.assertEqual(params["output_type"], 2)
        self.assertEqual(result.loc[0, "value"], 1.7)
        self.assertEqual(result.loc[1, "value"], 1.9)
        self.assertEqual(result.loc[0, "vintage_method"], "fred_api_asof")
        self.assertEqual(result.loc[0, "source"], "FRED/Federal Reserve H.15")
        self.assertTrue(bool(result.loc[0, "true_vintage_data_available"]))
        self.assertFalse(bool(result.loc[0, "fallback_used"]))

    def test_cpi_yoy_is_computed_from_monthly_same_vintage_pair(self):
        dates = pd.DatetimeIndex([pd.Timestamp("2015-01-16")])
        records = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2014-01-01", "2015-01-01"]),
                "value": [100.0, 110.0],
                "realtime_start": pd.to_datetime(["2015-01-16", "2015-01-16"]),
                "realtime_end": pd.to_datetime(["2262-04-11", "2262-04-11"]),
                "realtime_end_raw": ["9999-12-31", "9999-12-31"],
            }
        )
        selected = select_asof_weekly_values(
            records,
            dates,
            next(config for config in SERIES_CONFIGS if config.output_name == "CPI"),
            "local_raw_vintage",
        )

        yoy = build_cpi_yoy_asof_metadata(
            selected=selected,
            records=records,
            config=next(config for config in SERIES_CONFIGS if config.output_name == "CPI"),
            source="local_raw_vintage",
        )

        self.assertEqual(yoy.loc[0, "output_name"], "cpi_yoy_asof")
        self.assertAlmostEqual(yoy.loc[0, "value"], 0.10)
        self.assertEqual(
            yoy.loc[0, "transformation_applied"],
            "monthly_yoy_before_weekly_alignment",
        )
        self.assertEqual(
            pd.Timestamp(yoy.loc[0, "lagged_12m_cpi_observation_date_used"]),
            pd.Timestamp("2014-01-01"),
        )

    def test_dxy_uses_dtwexbgs_true_vintage_proxy_metadata(self):
        dxy = next(config for config in SERIES_CONFIGS if config.output_name == "DXY")

        self.assertEqual(dxy.series_id, "DTWEXBGS")
        self.assertEqual(dxy.title, "Nominal Broad U.S. Dollar Index")
        self.assertEqual(dxy.data_source, "FRED/Federal Reserve H.10")
        self.assertEqual(dxy.conceptual_role, "dollar_strength_proxy")
        self.assertIn("not ICE DXY/USDX", dxy.note)

    def test_resolve_series_configs_requires_dtwexbgs_dxy_pair(self):
        configs = resolve_series_configs(
            dollar_series_id="DTWEXBGS",
            dollar_column_name="DXY",
        )
        dxy = next(config for config in configs if config.output_name == "DXY")

        self.assertEqual(dxy.series_id, "DTWEXBGS")
        self.assertEqual(dxy.data_source, "FRED/Federal Reserve H.10")
        with self.assertRaisesRegex(ValueError, "Only DTWEXBGS"):
            resolve_series_configs(dollar_series_id="UNSUPPORTED")
        with self.assertRaisesRegex(ValueError, "must remain DXY"):
            resolve_series_configs(dollar_column_name="USD")
        without_dxy = resolve_series_configs(exclude_series=("DXY",))
        self.assertNotIn("DXY", {config.output_name for config in without_dxy})

    def test_validate_vintage_coverage_fails_when_series_starts_too_late(self):
        with self.assertRaisesRegex(ValueError, "after required start"):
            validate_vintage_coverage(
                series_id="DTWEXBGS",
                vintage_dates=["2019-02-04", "2019-02-11"],
                required_start=pd.Timestamp("2015-01-09"),
                required_end=pd.Timestamp("2019-02-11"),
            )

    def test_validate_vintage_coverage_passes_when_window_is_covered(self):
        validate_vintage_coverage(
            series_id="DGS10",
            vintage_dates=["2015-01-02", "2026-05-15"],
            required_start=pd.Timestamp("2015-01-09"),
            required_end=pd.Timestamp("2026-05-15"),
        )

    def test_endpoint_usability_fails_when_endpoint_observation_is_stale(self):
        selected = pd.DataFrame(
            {
                "date": pd.to_datetime(["2015-01-09", "2026-05-15"]),
                "value": [100.0, 105.0],
                "observation_date_used": pd.to_datetime(["2015-01-09", "2024-01-01"]),
                "true_vintage_data_available": [True, True],
            }
        )
        dxy = next(config for config in SERIES_CONFIGS if config.output_name == "DXY")

        with self.assertRaisesRegex(ValueError, "Discontinued/stale series"):
            validate_asof_endpoint_usability(
                selected=selected,
                config=dxy,
                required_start=pd.Timestamp("2015-01-09"),
                required_end=pd.Timestamp("2026-05-15"),
            )


def _write_returns(path: Path) -> None:
    dates = pd.date_range("2015-01-09", periods=60, freq="W-FRI")
    pd.DataFrame(
        {
            "date": dates,
            "SPY": ([0.01, 0.02, -0.01] * 20)[: len(dates)],
            "TLT": ([0.0, 0.01, 0.0] * 20)[: len(dates)],
            "GLD": ([0.0, 0.0, 0.01] * 20)[: len(dates)],
            "BTC-USD": ([0.03, -0.02, 0.01] * 20)[: len(dates)],
            "CASH": [0.0] * len(dates),
        }
    ).to_csv(path, index=False)


def _write_vintage_file(path: Path) -> None:
    observation_dates = pd.date_range("2013-01-01", "2016-03-01", freq="MS")
    pd.DataFrame(
        {
            "observation_date": observation_dates,
            "value": [100.0 + i for i in range(len(observation_dates))],
            "realtime_start": observation_dates,
            "realtime_end": ["9999-12-31"] * len(observation_dates),
        }
    ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
