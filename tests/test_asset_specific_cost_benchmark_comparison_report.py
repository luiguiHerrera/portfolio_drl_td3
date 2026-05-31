"""Tests for asset-specific-cost TD3 vs benchmark comparison reporting."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.asset_specific_cost_benchmark_comparison_report import (
    build_asset_specific_cost_benchmark_comparison_report,
    validate_cost_model_match,
)


class AssetSpecificCostBenchmarkComparisonReportTests(unittest.TestCase):
    def test_report_writes_outputs_and_combines_td3_with_benchmarks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            td3_dir, benchmark_dir, output_dir = self._write_inputs(temp_dir)

            report = build_asset_specific_cost_benchmark_comparison_report(
                td3_report_dir=str(td3_dir),
                benchmark_dir=str(benchmark_dir),
                output_dir=str(output_dir),
            )

            for path in report["paths"].values():
                self.assertTrue(Path(path).exists())
            ranking = report["combined_ranking"]
            self.assertIn("V3_real_macro_vintage_clean_no_dxy_cap_0p70", set(ranking["strategy_name"]))
            self.assertIn("BuyHold_GLD", set(ranking["strategy_name"]))
            self.assertIn("mandate_aware_score", ranking.columns)
            self.assertIn("robust_score", ranking.columns)

    def test_cost_model_mismatch_fails(self):
        td3_metadata = {
            "score_scope": "combined_asset_specific_full_universe",
            "cost_model": {
                "transaction_cost_mode": "asset_specific",
                "asset_transaction_cost_bps": {"SPY": 2.0, "CASH": 0.0},
            },
        }
        benchmark_metadata = {
            "transaction_cost_mode": "scalar",
            "asset_transaction_cost_bps": {"SPY": 2.0, "CASH": 0.0},
        }

        with self.assertRaisesRegex(ValueError, "Benchmark report"):
            validate_cost_model_match(td3_metadata, benchmark_metadata)

    def test_missing_benchmark_asset_specific_columns_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            td3_dir, benchmark_dir, output_dir = self._write_inputs(temp_dir)
            history_path = benchmark_dir / "histories" / "BuyHold_GLD_history.csv"
            history = pd.read_csv(history_path)
            history = history.drop(columns=["asset_turnover_SPY"])
            history.to_csv(history_path, index=False)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                build_asset_specific_cost_benchmark_comparison_report(
                    td3_report_dir=str(td3_dir),
                    benchmark_dir=str(benchmark_dir),
                    output_dir=str(output_dir),
                )

    def test_summary_mentions_no_statistical_superiority_claim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            td3_dir, benchmark_dir, output_dir = self._write_inputs(temp_dir)

            report = build_asset_specific_cost_benchmark_comparison_report(
                td3_report_dir=str(td3_dir),
                benchmark_dir=str(benchmark_dir),
                output_dir=str(output_dir),
            )

            self.assertIn("does not establish statistical superiority", report["summary"])
            self.assertIn("asset-specific transaction-cost", report["summary"])

    def _write_inputs(self, temp_dir: str) -> tuple[Path, Path, Path]:
        base = Path(temp_dir)
        td3_dir = base / "td3"
        benchmark_dir = base / "benchmarks"
        output_dir = base / "out"
        td3_dir.mkdir()
        (benchmark_dir / "histories").mkdir(parents=True)

        cost_model = {
            "transaction_cost_mode": "asset_specific",
            "asset_transaction_cost_bps": {
                "SPY": 2.0,
                "TLT": 2.0,
                "GLD": 2.0,
                "BTC-USD": 10.0,
                "CASH": 0.0,
            },
        }
        (td3_dir / "asset_specific_cost_metadata.json").write_text(
            json.dumps(
                {
                    "score_scope": "combined_asset_specific_full_universe",
                    "cost_model": cost_model,
                },
            ),
            encoding="utf-8",
        )
        (benchmark_dir / "benchmark_metadata.json").write_text(
            json.dumps(
                {
                    "transaction_cost_mode": "asset_specific",
                    "asset_transaction_cost_bps": cost_model[
                        "asset_transaction_cost_bps"
                    ],
                    "benchmark_names": ["BuyHold_GLD", "trend_spy_cash_12p"],
                },
            ),
            encoding="utf-8",
        )

        pd.DataFrame(
            [
                {
                    "candidate_name": "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
                    "base_candidate": "V3_real_macro_vintage_clean_no_dxy",
                    "cap_label": "0.70",
                    "cumulative_return": 0.10,
                    "annualized_return": 0.09,
                    "annualized_volatility": 0.09,
                    "sharpe": 0.95,
                    "sortino": 2.5,
                    "calmar": 2.8,
                    "robust_score": 0.61,
                    "mandate_aware_score": 0.55,
                    "max_drawdown": -0.08,
                    "worst_max_drawdown": -0.30,
                    "average_turnover": 0.10,
                    "mean_transaction_cost": 0.00002,
                    "average_effective_number_of_assets": 2.0,
                    "average_max_weight": 0.70,
                    "mean_cash_weight": 0.40,
                    "mean_btc_weight": 0.08,
                    "mean_btc_transaction_cost_contribution": 0.00001,
                    "cash_above_10_rate": 0.5,
                    "dsr_score": 0.40,
                    "dsr_method": "median_run",
                }
            ],
        ).to_csv(td3_dir / "asset_specific_cost_selected_candidates.csv", index=False)

        benchmark_rows = [
            {
                "benchmark_name": "BuyHold_GLD",
                "cumulative_return": 0.08,
                "annualized_return": 0.07,
                "annualized_volatility": 0.10,
                "sharpe": 0.70,
                "sortino": 1.3,
                "calmar": 1.0,
                "max_drawdown": -0.07,
                "average_turnover": 0.01,
                "total_transaction_cost": 0.001,
                "average_transaction_cost": 0.00001,
                "average_max_weight": 1.0,
                "average_effective_number_of_assets": 1.0,
                "mean_cash_weight": 0.0,
                "cash_above_10pct": 0.0,
            },
            {
                "benchmark_name": "trend_spy_cash_12p",
                "cumulative_return": 0.06,
                "annualized_return": 0.05,
                "annualized_volatility": 0.08,
                "sharpe": 0.60,
                "sortino": 1.1,
                "calmar": 0.9,
                "max_drawdown": -0.06,
                "average_turnover": 0.05,
                "total_transaction_cost": 0.002,
                "average_transaction_cost": 0.00002,
                "average_max_weight": 1.0,
                "average_effective_number_of_assets": 1.0,
                "mean_cash_weight": 0.25,
                "cash_above_10pct": 0.25,
            },
        ]
        pd.DataFrame(benchmark_rows).to_csv(
            benchmark_dir / "benchmark_metrics_table.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "benchmark_name": row["benchmark_name"],
                    "transaction_cost_mode": "asset_specific",
                    "average_transaction_cost": row["average_transaction_cost"],
                }
                for row in benchmark_rows
            ],
        ).to_csv(benchmark_dir / "benchmark_diagnostics.csv", index=False)
        for row in benchmark_rows:
            self._write_history(
                benchmark_dir / "histories" / f"{row['benchmark_name']}_history.csv",
            )
        return td3_dir, benchmark_dir, output_dir

    def _write_history(self, path: Path) -> None:
        dates = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
        frame = pd.DataFrame(
            {
                "date": dates,
                "portfolio_return": [0.002] * 40,
                "financial_net_return": [0.0015] * 40,
                "transaction_cost_mode": ["asset_specific"] * 40,
                "transaction_cost": [0.0005] * 40,
                "turnover": [0.1] * 40,
                "drawdown": [0.0] * 40,
                "weight_SPY": [0.0] * 40,
                "weight_TLT": [0.0] * 40,
                "weight_GLD": [1.0] * 40,
                "weight_BTC-USD": [0.0] * 40,
                "weight_CASH": [0.0] * 40,
            },
        )
        for asset in ["SPY", "TLT", "GLD", "BTC-USD", "CASH"]:
            frame[f"asset_turnover_{asset}"] = 0.0
            frame[f"asset_transaction_cost_contribution_{asset}"] = 0.0
        frame.to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
