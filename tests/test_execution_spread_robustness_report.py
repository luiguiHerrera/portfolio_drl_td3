import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.execution_spread_robustness_report import (
    SPREAD_SCENARIOS,
    build_execution_spread_robustness_report,
)


class ExecutionSpreadRobustnessReportTest(unittest.TestCase):
    def test_report_outputs_and_degradation_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zero_td3 = root / "zero_td3"
            bil_td3 = root / "bil_td3"
            zero_bench = root / "zero_bench" / "benchmarks"
            bil_bench = root / "bil_bench" / "benchmarks"
            output = root / "spread_report"

            _write_td3_histories(zero_td3, "V3_real_macro_vintage_clean_no_dxy", "0p70", value=0.01)
            _write_td3_histories(bil_td3, "V7_real_macro_vintage_clean_no_dxy_garch", "0p80", value=0.012)
            for benchmark_dir in [zero_bench, bil_bench]:
                _write_benchmark_history(benchmark_dir, "trend_spy_cash_12p", value=0.009)
                _write_benchmark_history(benchmark_dir, "Equal_Weight", value=0.008)
                _write_benchmark_history(benchmark_dir, "BuyHold_GLD", value=0.007)

            result = build_execution_spread_robustness_report(
                zero_td3_dir=str(zero_td3),
                zero_benchmark_dir=str(zero_bench),
                bil_td3_dir=str(bil_td3),
                bil_benchmark_dir=str(bil_bench),
                output_dir=str(output),
            )

            for path in result["paths"].values():
                self.assertTrue(Path(path).exists())

            metrics = pd.read_csv(result["paths"]["metrics"])
            self.assertEqual(set(metrics["scenario"]), {scenario.name for scenario in SPREAD_SCENARIOS})
            stress = metrics[metrics["scenario"] == "stress_spread"]
            base = metrics[metrics["scenario"] == "base_no_extra_spread"]
            self.assertGreater(stress["total_spread_cost"].max(), base["total_spread_cost"].max())
            self.assertTrue((stress["delta_return_vs_base"] <= 0.0).all())

    def test_missing_asset_turnover_records_warning_not_silent_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zero_td3 = root / "zero_td3"
            bil_td3 = root / "bil_td3"
            zero_bench = root / "zero_bench" / "benchmarks"
            bil_bench = root / "bil_bench" / "benchmarks"
            output = root / "spread_report"

            _write_td3_histories(zero_td3, "V3_real_macro_vintage_clean_no_dxy", "0p70", value=0.01)
            _write_td3_histories(bil_td3, "V7_real_macro_vintage_clean_no_dxy_garch", "0p80", value=0.012)
            for benchmark_dir in [zero_bench, bil_bench]:
                _write_benchmark_history(benchmark_dir, "trend_spy_cash_12p", value=0.009)
                _write_benchmark_history(benchmark_dir, "Equal_Weight", value=0.008)
                _write_benchmark_history(benchmark_dir, "BuyHold_GLD", value=0.007)

            history = zero_bench / "histories" / "trend_spy_cash_12p_history.csv"
            frame = pd.read_csv(history).drop(columns=["asset_turnover_BTC-USD"])
            frame.to_csv(history, index=False)

            result = build_execution_spread_robustness_report(
                zero_td3_dir=str(zero_td3),
                zero_benchmark_dir=str(zero_bench),
                bil_td3_dir=str(bil_td3),
                bil_benchmark_dir=str(bil_bench),
                output_dir=str(output),
                include_references=False,
            )
            metadata = json.loads(Path(result["paths"]["metadata"]).read_text())
            self.assertTrue(any("asset_turnover_BTC-USD" in warning for warning in metadata["warnings"]))

    def test_report_is_reporting_only_and_does_not_create_new_winners(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zero_td3 = root / "zero_td3"
            bil_td3 = root / "bil_td3"
            zero_bench = root / "zero_bench" / "benchmarks"
            bil_bench = root / "bil_bench" / "benchmarks"
            output = root / "spread_report"

            _write_td3_histories(zero_td3, "V3_real_macro_vintage_clean_no_dxy", "0p70", value=0.01)
            _write_td3_histories(bil_td3, "V7_real_macro_vintage_clean_no_dxy_garch", "0p80", value=0.012)
            for benchmark_dir in [zero_bench, bil_bench]:
                _write_benchmark_history(benchmark_dir, "trend_spy_cash_12p", value=0.009)

            result = build_execution_spread_robustness_report(
                zero_td3_dir=str(zero_td3),
                zero_benchmark_dir=str(zero_bench),
                bil_td3_dir=str(bil_td3),
                bil_benchmark_dir=str(bil_bench),
                output_dir=str(output),
                include_references=False,
            )
            metadata = json.loads(Path(result["paths"]["metadata"]).read_text())
            self.assertTrue(metadata["reporting_only"])
            self.assertFalse(metadata["retrained"])
            self.assertFalse(metadata["creates_new_final_winners"])


def _write_td3_histories(root: Path, candidate: str, cap: str, value: float) -> None:
    for fold in ["F1", "F2"]:
        run_dir = root / "per_candidate" / candidate / f"{fold}_{candidate}_cap_{cap}_seed_7"
        run_dir.mkdir(parents=True, exist_ok=True)
        _history(value=value).to_csv(run_dir / "test_policy_history.csv", index=False)


def _write_benchmark_history(root: Path, name: str, value: float) -> None:
    path = root / "histories" / f"{name}_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    _history(value=value).to_csv(path, index=False)


def _history(value: float) -> pd.DataFrame:
    dates = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    return pd.DataFrame(
        {
            "date": dates,
            "financial_net_return": [value, value, -value / 2, value, 0.0, value / 2],
            "portfolio_return": [value + 0.001] * len(dates),
            "turnover": [0.5, 0.2, 0.3, 0.1, 0.0, 0.4],
            "transaction_cost": [0.001, 0.0004, 0.0006, 0.0002, 0.0, 0.0008],
            "asset_turnover_SPY": [0.1, 0.1, 0.1, 0.0, 0.0, 0.2],
            "asset_turnover_TLT": [0.1, 0.0, 0.1, 0.0, 0.0, 0.1],
            "asset_turnover_GLD": [0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
            "asset_turnover_BTC-USD": [0.1, 0.1, 0.1, 0.0, 0.0, 0.1],
            "asset_turnover_CASH": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            "weight_SPY": [0.3, 0.35, 0.3, 0.4, 0.4, 0.3],
            "weight_TLT": [0.2, 0.2, 0.25, 0.2, 0.2, 0.2],
            "weight_GLD": [0.2, 0.2, 0.2, 0.15, 0.15, 0.2],
            "weight_BTC-USD": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "weight_CASH": [0.2, 0.15, 0.15, 0.15, 0.15, 0.2],
        }
    )


if __name__ == "__main__":
    unittest.main()
