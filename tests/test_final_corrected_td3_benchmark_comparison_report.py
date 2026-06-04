"""Tests for final corrected TD3-vs-benchmark comparison reporting."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.final_corrected_td3_benchmark_comparison_report import (
    EXPECTED_BENCHMARK_COUNT,
    EXPECTED_TD3_COUNT,
    build_single_report,
    validate_benchmark_cost_model,
)


class FinalCorrectedTd3BenchmarkComparisonReportTests(unittest.TestCase):
    def test_report_writes_outputs_and_combines_selected_td3_with_benchmarks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            td3_dir, benchmark_dir, output_dir = self._write_inputs(temp_dir, cash_bps=0.0)

            report = build_single_report(
                cash_label="zero_cash",
                td3_dir=str(td3_dir),
                benchmark_dir=str(benchmark_dir),
                output_dir=str(output_dir),
                expected_cash_bps=0.0,
                expected_returns_path_contains="returns_weekly_latest.csv",
            )

            for path in report["paths"].values():
                self.assertTrue(Path(path).exists())
            ranking = report["combined_ranking"]
            self.assertEqual(EXPECTED_TD3_COUNT, (ranking["strategy_type"] == "TD3").sum())
            self.assertEqual(
                EXPECTED_BENCHMARK_COUNT,
                (ranking["strategy_type"] == "benchmark").sum(),
            )
            self.assertIn("V5_no_volatility_block_cap_0p50", set(ranking["strategy_name"]))
            self.assertIn("benchmark_00", set(ranking["strategy_name"]))
            self.assertIn("mandate_aware_score", ranking.columns)
            self.assertIn("robust_score", ranking.columns)

    def test_bil_cash_cost_mapping_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            td3_dir, benchmark_dir, output_dir = self._write_inputs(
                temp_dir,
                cash_bps=2.0,
                returns_path="data/processed/returns_weekly_latest_cash_bil_proxy.csv",
            )

            report = build_single_report(
                cash_label="bil_cash",
                td3_dir=str(td3_dir),
                benchmark_dir=str(benchmark_dir),
                output_dir=str(output_dir),
                expected_cash_bps=2.0,
                expected_returns_path_contains="returns_weekly_latest_cash_bil_proxy.csv",
            )

            metadata = report["metadata"]
            self.assertEqual(2.0, metadata["asset_transaction_cost_bps"]["CASH"])
            self.assertIn("BIL-CASH", report["summary"])

    def test_cash_cost_mismatch_fails(self):
        metadata = {
            "transaction_cost_mode": "asset_specific",
            "returns_path": "data/processed/returns_weekly_latest.csv",
            "asset_transaction_cost_bps": {
                "SPY": 2.0,
                "TLT": 2.0,
                "GLD": 2.0,
                "BTC-USD": 10.0,
                "CASH": 2.0,
            },
        }

        with self.assertRaisesRegex(ValueError, "cost map mismatch"):
            validate_benchmark_cost_model(
                benchmark_metadata=metadata,
                expected_cash_bps=0.0,
                expected_returns_path_contains="returns_weekly_latest.csv",
            )

    def test_cross_cash_returns_path_mismatch_fails(self):
        metadata = {
            "transaction_cost_mode": "asset_specific",
            "returns_path": "data/processed/returns_weekly_latest.csv",
            "asset_transaction_cost_bps": {
                "SPY": 2.0,
                "TLT": 2.0,
                "GLD": 2.0,
                "BTC-USD": 10.0,
                "CASH": 0.0,
            },
        }

        with self.assertRaisesRegex(ValueError, "cash assumption"):
            validate_benchmark_cost_model(
                benchmark_metadata=metadata,
                expected_cash_bps=0.0,
                expected_returns_path_contains="cash_bil_proxy",
            )

    def test_missing_asset_specific_history_columns_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            td3_dir, benchmark_dir, output_dir = self._write_inputs(temp_dir, cash_bps=0.0)
            history_path = benchmark_dir / "histories" / "benchmark_00_history.csv"
            history = pd.read_csv(history_path)
            history.drop(columns=["asset_transaction_cost_contribution_CASH"]).to_csv(
                history_path,
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                build_single_report(
                    cash_label="zero_cash",
                    td3_dir=str(td3_dir),
                    benchmark_dir=str(benchmark_dir),
                    output_dir=str(output_dir),
                    expected_cash_bps=0.0,
                    expected_returns_path_contains="returns_weekly_latest.csv",
                )

    def _write_inputs(
        self,
        temp_dir: str,
        cash_bps: float,
        returns_path: str = "data/processed/returns_weekly_latest.csv",
    ) -> tuple[Path, Path, Path]:
        base = Path(temp_dir)
        td3_dir = base / "td3"
        benchmark_dir = base / "benchmarks"
        output_dir = base / "out"
        td3_dir.mkdir()
        (benchmark_dir / "histories").mkdir(parents=True)

        base_candidates = [
            "V3_real_macro_vintage_clean_no_dxy",
            "V4_real_garch_current",
            "V5_no_volatility_block",
            "V7_real_macro_vintage_clean_no_dxy_garch",
            "V8_ewma_garch_vol_current",
        ]
        selected_caps = ["0.70", "0.50", "0.50", "0.80", "0.70"]
        best_rows = []
        all_rows = []
        for idx, (candidate, cap_label) in enumerate(zip(base_candidates, selected_caps)):
            cap_value = float(cap_label)
            score = 0.50 + idx * 0.02
            name = f"{candidate}_cap_{cap_label.replace('.', 'p')}"
            best_rows.append(
                {
                    "base_candidate": candidate,
                    "best_by_mandate_aware_score": cap_value,
                    "best_mandate_aware_score": score,
                    "best_by_robust_score": cap_value,
                    "best_robust_score": score + 0.05,
                    "best_by_max_drawdown": cap_value,
                    "best_max_drawdown": -0.10,
                    "best_by_turnover": cap_value,
                    "best_turnover": 0.10,
                    "best_by_effective_assets": cap_value,
                    "best_effective_assets": 2.5,
                }
            )
            all_rows.append(
                {
                    "candidate_name": name,
                    "base_candidate": candidate,
                    "max_weight_cap": cap_value,
                    "cap_label": cap_label,
                    "split": "test",
                    "n_folds": 4,
                    "n_seeds": 10,
                    "episodes": 60,
                    "cumulative_return": 0.08 + idx * 0.01,
                    "annualized_return": 0.06 + idx * 0.01,
                    "annualized_volatility": 0.10,
                    "sharpe": 0.60 + idx * 0.05,
                    "sortino": 1.0 + idx * 0.1,
                    "calmar": 1.2 + idx * 0.1,
                    "robust_score": score + 0.05,
                    "mandate_aware_score": score,
                    "max_drawdown": -0.09 - idx * 0.005,
                    "worst_max_drawdown": -0.20,
                    "average_turnover": 0.10,
                    "mean_transaction_cost": 0.00002,
                    "average_effective_number_of_assets": 2.5,
                    "average_max_weight": 0.50,
                    "mean_cash_weight": 0.20,
                    "cash_above_10_rate": 0.3,
                }
            )
        pd.DataFrame(best_rows).to_csv(td3_dir / "cap_sensitivity_best_caps.csv", index=False)
        pd.DataFrame(all_rows).to_csv(td3_dir / "cap_sensitivity_all_results.csv", index=False)

        asset_bps = {
            "SPY": 2.0,
            "TLT": 2.0,
            "GLD": 2.0,
            "BTC-USD": 10.0,
            "CASH": cash_bps,
        }
        benchmark_names = [f"benchmark_{i:02d}" for i in range(EXPECTED_BENCHMARK_COUNT)]
        (benchmark_dir / "benchmark_metadata.json").write_text(
            json.dumps(
                {
                    "transaction_cost_mode": "asset_specific",
                    "returns_path": returns_path,
                    "asset_transaction_cost_bps": asset_bps,
                    "benchmark_names": benchmark_names,
                },
            ),
            encoding="utf-8",
        )
        benchmark_rows = []
        diagnostic_rows = []
        for idx, name in enumerate(benchmark_names):
            benchmark_rows.append(
                {
                    "benchmark_name": name,
                    "cumulative_return": 0.05 + idx * 0.001,
                    "annualized_return": 0.04 + idx * 0.001,
                    "annualized_volatility": 0.08,
                    "sharpe": 0.50 + idx * 0.01,
                    "sortino": 0.9,
                    "calmar": 0.8,
                    "max_drawdown": -0.08,
                    "average_turnover": 0.02,
                    "total_transaction_cost": 0.001,
                    "average_transaction_cost": 0.00001,
                    "average_max_weight": 1.0,
                    "average_effective_number_of_assets": 1.0,
                    "mean_cash_weight": 0.1,
                    "cash_above_10pct": 0.1,
                }
            )
            diagnostic_rows.append(
                {
                    "benchmark_name": name,
                    "transaction_cost_mode": "asset_specific",
                }
            )
            self._write_history(benchmark_dir / "histories" / f"{name}_history.csv")
        pd.DataFrame(benchmark_rows).to_csv(
            benchmark_dir / "benchmark_metrics_table.csv",
            index=False,
        )
        pd.DataFrame(diagnostic_rows).to_csv(
            benchmark_dir / "benchmark_diagnostics.csv",
            index=False,
        )
        return td3_dir, benchmark_dir, output_dir

    def _write_history(self, path: Path) -> None:
        dates = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
        frame = pd.DataFrame(
            {
                "date": dates,
                "portfolio_return": [0.002] * 40,
                "financial_net_return": [0.0018] * 40,
                "gross_return": [0.002] * 40,
                "net_return": [0.0018] * 40,
                "portfolio_value": [100000.0] * 40,
                "drawdown": [0.0] * 40,
                "turnover": [0.1] * 40,
                "transaction_cost": [0.0002] * 40,
                "transaction_cost_mode": ["asset_specific"] * 40,
                "weight_SPY": [0.2] * 40,
                "weight_TLT": [0.2] * 40,
                "weight_GLD": [0.2] * 40,
                "weight_BTC-USD": [0.2] * 40,
                "weight_CASH": [0.2] * 40,
            },
        )
        for asset in ["SPY", "TLT", "GLD", "BTC-USD", "CASH"]:
            frame[f"asset_turnover_{asset}"] = 0.02
            frame[f"asset_transaction_cost_contribution_{asset}"] = 0.00001
        frame.to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
