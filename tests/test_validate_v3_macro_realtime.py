import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.validate_v3_macro_realtime import (
    build_freshness_checks,
    build_leakage_checks,
    compare_current_vs_realtime,
    validate_v3_macro_realtime,
)


class ValidateV3MacroRealtimeTests(unittest.TestCase):
    def test_validation_writes_outputs_and_passes_leakage_checks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            returns_path = root / "returns.csv"
            realtime_path = root / "macro_realtime.csv"
            current_path = root / "macro_current.csv"
            output_dir = root / "validation"
            metadata_path = output_dir / "v3_macro_realtime_series_metadata.csv"
            output_dir.mkdir()
            dates = _protocol_dates()
            _write_returns(returns_path, dates)
            _write_macro(realtime_path, dates, offset=0.0)
            _write_macro(current_path, dates, offset=0.1)
            _write_metadata(metadata_path, dates)

            result = validate_v3_macro_realtime(
                returns_path=str(returns_path),
                macro_realtime_path=str(realtime_path),
                macro_current_path=str(current_path),
                output_dir=str(output_dir),
            )

            for path in result["paths"].values():
                self.assertTrue(Path(path).exists())
            self.assertTrue((result["leakage_checks"]["status"] == "pass").all())
            self.assertGreater(
                result["current_vs_realtime"]["mean_absolute_difference"].max(),
                0.0,
            )

    def test_leakage_checks_detect_observation_after_weekly_date(self):
        metadata = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-10"]),
                "series_id": ["CPIAUCSL"],
                "feature_name": ["CPI"],
                "output_name": ["CPI"],
                "title": [
                    "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average"
                ],
                "source": ["FRED/Bureau of Labor Statistics"],
                "conceptual_role": ["inflation_proxy"],
                "frequency": ["monthly"],
                "note": [""],
                "observation_date_used": pd.to_datetime(["2020-01-17"]),
                "as_of_date": pd.to_datetime(["2020-01-10"]),
                "realtime_start_used": pd.to_datetime(["2020-01-10"]),
                "realtime_end_used": ["9999-12-31"],
                "realtime_end_parsed": pd.to_datetime(["2262-04-11"]),
                "vintage_method": ["local_raw_vintage"],
                "true_vintage_data_available": [True],
                "transformation_applied": ["asof_level"],
                "fallback_used": [False],
            }
        )

        checks = build_leakage_checks(metadata).set_index("check_name")

        self.assertEqual(
            checks.loc["observation_date_not_after_weekly_date", "status"],
            "fail",
        )
        self.assertEqual(checks.loc["cpi_release_timing_respected", "status"], "fail")

    def test_current_vs_realtime_difference_summary(self):
        index = pd.to_datetime(["2020-01-03", "2020-01-10"])
        current = pd.DataFrame({"DGS10": [1.0, 2.0]}, index=index)
        realtime = pd.DataFrame({"DGS10": [1.0, 1.5]}, index=index)

        comparison = compare_current_vs_realtime(current, realtime)

        self.assertEqual(comparison.loc[0, "series"], "DGS10")
        self.assertAlmostEqual(comparison.loc[0, "mean_absolute_difference"], 0.25)
        self.assertAlmostEqual(comparison.loc[0, "max_absolute_difference"], 0.5)

    def test_freshness_checks_detect_discontinued_daily_series(self):
        metadata = pd.DataFrame(
            {
                "date": pd.to_datetime(["2015-01-09", "2026-05-15"]),
                "series_id": ["DTWEXB", "DTWEXB"],
                "feature_name": ["DXY", "DXY"],
                "output_name": ["DXY", "DXY"],
                "title": ["Discontinued dollar series", "Discontinued dollar series"],
                "source": ["FRED/Federal Reserve H.10", "FRED/Federal Reserve H.10"],
                "conceptual_role": ["dollar_strength_proxy", "dollar_strength_proxy"],
                "frequency": ["daily", "daily"],
                "note": ["", ""],
                "value": [100.0, 105.0],
                "observation_date_used": pd.to_datetime(["2015-01-09", "2024-01-01"]),
                "as_of_date": pd.to_datetime(["2015-01-09", "2026-05-15"]),
                "realtime_start_used": pd.to_datetime(["2015-01-09", "2026-05-15"]),
                "realtime_end_used": ["9999-12-31", "9999-12-31"],
                "vintage_method": ["fred_api_asof", "fred_api_asof"],
                "true_vintage_data_available": [True, True],
                "transformation_applied": ["asof_level", "asof_level"],
                "fallback_method": ["", ""],
                "fallback_used": [False, False],
            }
        )

        checks = build_freshness_checks(metadata)

        self.assertEqual(checks.loc[0, "status"], "fail")
        self.assertGreater(checks.loc[0, "latest_observation_age_days"], 30)


def _protocol_dates() -> pd.DatetimeIndex:
    return pd.date_range("2015-01-02", "2026-05-15", freq="W-FRI")


def _write_returns(path: Path, dates: pd.DatetimeIndex) -> None:
    n = len(dates)
    pd.DataFrame(
        {
            "date": dates,
            "SPY": [0.001 + (i % 5) * 0.0001 for i in range(n)],
            "TLT": [0.0005 + (i % 3) * 0.0001 for i in range(n)],
            "GLD": [0.0002 + (i % 4) * 0.0001 for i in range(n)],
            "BTC-USD": [0.002 + (i % 7) * 0.0001 for i in range(n)],
            "CASH": [0.0 for _ in range(n)],
        }
    ).to_csv(path, index=False)


def _write_macro(path: Path, dates: pd.DatetimeIndex, offset: float) -> None:
    n = len(dates)
    pd.DataFrame(
        {
            "date": dates,
            "DGS10": [4.0 + offset + i * 0.001 for i in range(n)],
            "DGS2": [3.0 + offset + i * 0.001 for i in range(n)],
            "VIX": [20.0 + offset + (i % 5) for i in range(n)],
            "DXY": [100.0 + offset + i * 0.01 for i in range(n)],
            "CPI": [300.0 + offset + i * 0.02 for i in range(n)],
            "cpi_yoy_asof": [0.02 + offset * 0.001 + i * 0.0001 for i in range(n)],
        }
    ).to_csv(path, index=False)


def _write_metadata(path: Path, dates: pd.DatetimeIndex) -> None:
    rows = []
    series = [
        (
            "DGS10",
            "DGS10",
            "10-Year Treasury Constant Maturity Rate",
            "FRED/Federal Reserve H.15",
            "long_rate",
            "",
        ),
        (
            "DGS2",
            "DGS2",
            "2-Year Treasury Constant Maturity Rate",
            "FRED/Federal Reserve H.15",
            "short_rate",
            "",
        ),
        (
            "VIX",
            "VIXCLS",
            "CBOE Volatility Index: VIX",
            "FRED/CBOE",
            "equity_volatility_proxy",
            "",
        ),
        (
            "DXY",
            "DTWEXBGS",
            "Nominal Broad U.S. Dollar Index",
            "FRED/Federal Reserve H.10",
            "dollar_strength_proxy",
            "This is not ICE DXY/USDX; it is the Fed nominal broad trade-weighted U.S. dollar index.",
        ),
        (
            "cpi_yoy_asof",
            "CPIAUCSL",
            "CPI year-over-year change computed from as-of monthly CPI observations",
            "FRED/Bureau of Labor Statistics",
            "inflation_yoy_asof",
            "Computed before weekly alignment.",
        ),
        (
            "CPI",
            "CPIAUCSL",
            "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
            "FRED/Bureau of Labor Statistics",
            "inflation_proxy",
            "",
        ),
    ]
    for date in dates:
        for output_name, series_id, title, source, role, note in series:
            rows.append(
                {
                    "date": date,
                    "series_id": series_id,
                    "feature_name": output_name,
                    "output_name": output_name,
                    "title": title,
                    "source": source,
                    "conceptual_role": role,
                    "frequency": "monthly" if output_name in {"CPI", "cpi_yoy_asof"} else "daily",
                    "note": note,
                    "value": 1.0,
                    "observation_date_used": date,
                    "as_of_date": date,
                    "realtime_start_used": date,
                    "realtime_end_used": "9999-12-31",
                    "vintage_method": "local_raw_vintage",
                    "true_vintage_data_available": True,
                    "transformation_applied": (
                        "monthly_yoy_before_weekly_alignment"
                        if output_name == "cpi_yoy_asof"
                        else "asof_level"
                    ),
                    "fallback_method": "",
                    "fallback_used": False,
                    "current_cpi_observation_date_used": date if output_name == "cpi_yoy_asof" else pd.NaT,
                    "current_cpi_as_of_date": date if output_name == "cpi_yoy_asof" else pd.NaT,
                    "current_cpi_realtime_start_used": date if output_name == "cpi_yoy_asof" else pd.NaT,
                    "lagged_12m_cpi_observation_date_used": (
                        date - pd.DateOffset(years=1) if output_name == "cpi_yoy_asof" else pd.NaT
                    ),
                    "lagged_12m_cpi_as_of_date": date if output_name == "cpi_yoy_asof" else pd.NaT,
                    "lagged_12m_cpi_realtime_start_used": date if output_name == "cpi_yoy_asof" else pd.NaT,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
