"""Tests for capped-vs-uncapped TD3 protocol comparison reporting."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.experiments.run_capped_td3_protocol_comparison import (
    build_combined_comparison_table,
    build_metadata,
    build_pairwise_cap_deltas,
    label_cap_pairwise_decision,
    load_capped_td3_rows,
    normalize_benchmark_rows,
    run_capped_td3_protocol_comparison,
)


class CappedTD3ProtocolComparisonTests(unittest.TestCase):
    def setUp(self):
        self.benchmark_rows = pd.DataFrame(
            [
                {
                    "strategy_name": "Equal_Weight",
                    "strategy_type": "benchmark",
                    "cumulative_return": 0.10,
                    "annualized_return": 0.08,
                    "annualized_volatility": 0.10,
                    "sharpe": 0.80,
                    "sortino": 1.20,
                    "calmar": 0.70,
                    "robust_score": 0.62,
                    "max_drawdown": -0.12,
                    "average_turnover": 0.05,
                    "total_transaction_cost": 0.01,
                    "average_max_weight": 0.25,
                    "average_effective_number_of_assets": 4.0,
                    "mean_cash_weight": 0.20,
                    "cash_above_10pct": 1.0,
                    "pooled_dsr_n25": 0.30,
                    "date_averaged_dsr_n25": 0.30,
                    "dsr_method": "date_averaged",
                },
                {
                    "strategy_name": "rolling_risk_parity_inverse_vol_12p",
                    "strategy_type": "benchmark",
                    "cumulative_return": 0.08,
                    "annualized_return": 0.06,
                    "annualized_volatility": 0.09,
                    "sharpe": 0.67,
                    "sortino": 1.00,
                    "calmar": 0.55,
                    "robust_score": 0.50,
                    "max_drawdown": -0.13,
                    "average_turnover": 0.10,
                    "total_transaction_cost": 0.02,
                    "average_max_weight": 0.35,
                    "average_effective_number_of_assets": 3.0,
                    "mean_cash_weight": 0.0,
                    "cash_above_10pct": 0.0,
                    "pooled_dsr_n25": 0.20,
                    "date_averaged_dsr_n25": 0.20,
                    "dsr_method": "date_averaged",
                },
            ]
        )

    def test_loads_capped_experiment_rows_correctly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_cap_experiment(temp_dir, "V2_reference_full")

            rows = load_capped_td3_rows({"V2_reference_full": str(path)})

        self.assertEqual(set(rows["strategy_name"]), {"V2_uncapped", "V2_cap_0.60"})
        self.assertEqual(set(rows["base_candidate"]), {"V2_reference_full"})
        self.assertIn("median_run_dsr_n25", rows.columns)

    def test_distinguishes_capped_vs_uncapped_td3(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_cap_experiment(temp_dir, "V5_no_volatility_block")

            rows = load_capped_td3_rows({"V5_no_volatility_block": str(path)})

        by_name = rows.set_index("strategy_name")
        self.assertEqual(by_name.loc["V5_uncapped", "strategy_type"], "td3_uncapped")
        self.assertEqual(by_name.loc["V5_cap_0.60", "strategy_type"], "td3_capped")
        self.assertEqual(by_name.loc["V5_cap_0.60", "constraint_status"], "cap_0.60")

    def test_combines_benchmarks_and_td3_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_cap_experiment(temp_dir, "V6_financial_state")
            td3 = load_capped_td3_rows({"V6_financial_state": str(path)})
            benchmarks = normalize_benchmark_rows(self.benchmark_rows)

            combined = build_combined_comparison_table(td3, benchmarks)

        self.assertIn("V6_cap_0.60", set(combined["strategy_name"]))
        self.assertIn("Equal_Weight", set(combined["strategy_name"]))
        self.assertIn("mandate_aware_score", combined.columns)

    def test_computes_pairwise_deltas_correctly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_cap_experiment(temp_dir, "V2_reference_full")
            td3 = load_capped_td3_rows({"V2_reference_full": str(path)})
            combined = build_combined_comparison_table(
                td3,
                normalize_benchmark_rows(self.benchmark_rows),
            )

            deltas = build_pairwise_cap_deltas(combined)

        self.assertEqual(len(deltas), 1)
        row = deltas.iloc[0]
        self.assertAlmostEqual(float(row["delta_robust_score"]), 0.30)
        self.assertAlmostEqual(
            float(row["delta_average_effective_number_of_assets"]),
            1.30,
        )

    def test_mandate_aware_ranking_works(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_cap_experiment(temp_dir, "V2_reference_full")
            td3 = load_capped_td3_rows({"V2_reference_full": str(path)})

            combined = build_combined_comparison_table(
                td3,
                normalize_benchmark_rows(self.benchmark_rows),
            )

        top = combined.sort_values("mandate_aware_rank").iloc[0]
        self.assertEqual(top["strategy_name"], "V2_cap_0.60")
        self.assertEqual(int(top["mandate_aware_rank"]), 1)

    def test_benchmark_rows_are_not_treated_as_capped_or_uncapped(self):
        benchmarks = normalize_benchmark_rows(self.benchmark_rows)

        self.assertTrue((benchmarks["strategy_type"] == "benchmark").all())
        self.assertTrue((benchmarks["constraint_status"] == "benchmark").all())
        self.assertTrue(benchmarks["max_weight_cap"].isna().all())

    def test_metadata_includes_input_folders_and_cap_value(self):
        metadata = build_metadata(
            returns_path="returns.csv",
            output_dir="out",
            input_paths={"V2_reference_full": "v2_path"},
            cap_value=0.60,
            transaction_cost=0.001,
            benchmark_output_dir="benchmarks",
            benchmark_info={"benchmark_robust_score_computed": True},
        )

        self.assertEqual(metadata["cap_value"], 0.60)
        self.assertEqual(metadata["input_experiment_folders"]["V2_reference_full"], "v2_path")
        self.assertTrue(metadata["benchmark_robust_score_computed"])

    def test_decision_labels_cover_cap_cases(self):
        dominates = {
            "delta_average_effective_number_of_assets": 0.30,
            "delta_average_max_weight": -0.20,
            "delta_robust_score": 0.02,
            "delta_mandate_aware_score": 0.03,
            "delta_max_drawdown": 0.04,
            "delta_annualized_return": 0.01,
        }
        hurts_return = {
            **dominates,
            "delta_annualized_return": -0.02,
        }

        self.assertEqual(label_cap_pairwise_decision(dominates), "cap_dominates_uncapped")
        self.assertEqual(
            label_cap_pairwise_decision(hurts_return),
            "cap_improves_mandate_but_hurts_return",
        )

    def test_runner_writes_expected_outputs_with_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            v2_path = self._write_cap_experiment(temp_dir, "V2_reference_full")
            v5_path = self._write_cap_experiment(temp_dir, "V5_no_volatility_block")
            v6_path = self._write_cap_experiment(temp_dir, "V6_financial_state")
            output_dir = Path(temp_dir) / "comparison"

            result = run_capped_td3_protocol_comparison(
                returns_path=str(returns_path),
                v2_path=str(v2_path),
                v5_path=str(v5_path),
                v6_path=str(v6_path),
                output_dir=str(output_dir),
                transaction_cost=0.01,
            )

            paths = result["paths"]
            for key in [
                "summary",
                "pairwise_deltas",
                "mandate_ranking",
                "performance_ranking",
                "metadata",
            ]:
                self.assertTrue(Path(paths[key]).exists())
            metadata = json.loads(Path(paths["metadata"]).read_text())

        self.assertEqual(metadata["cap_value"], 0.60)
        self.assertIn("V6_financial_state", metadata["input_experiment_folders"])

    def _write_cap_experiment(self, temp_dir: str, base_candidate: str) -> Path:
        path = Path(temp_dir) / base_candidate
        path.mkdir()
        prefix = {
            "V2_reference_full": "V2_reference_full",
            "V5_no_volatility_block": "V5_no_volatility_block",
            "V6_financial_state": "V6_financial_state",
        }[base_candidate]
        rows = pd.DataFrame(
            [
                {
                    "candidate_name": f"{prefix}_cap_uncapped",
                    "base_candidate": base_candidate,
                    "max_weight_cap": pd.NA,
                    "split": "test",
                    "n_folds": 1,
                    "n_seeds": 1,
                    "episodes": 1,
                    "cumulative_return": 0.03,
                    "annualized_return": 0.02,
                    "annualized_volatility": 0.12,
                    "sharpe": 0.20,
                    "sortino": 0.30,
                    "calmar": 0.20,
                    "robust_score": 0.40,
                    "mandate_aware_score": 0.30,
                    "max_drawdown": -0.20,
                    "worst_max_drawdown": -0.25,
                    "average_turnover": 0.60,
                    "mean_transaction_cost": 0.0006,
                    "average_effective_number_of_assets": 1.10,
                    "average_max_weight": 0.95,
                    "mean_cash_weight": 0.01,
                    "cash_above_10_rate": 0.0,
                    "concentration_classification": "learned_extreme_concentration",
                    "suspicious_or_lazy_concentration_candidate": True,
                    "justified_concentration_candidate": False,
                },
                {
                    "candidate_name": f"{prefix}_cap_0p60",
                    "base_candidate": base_candidate,
                    "max_weight_cap": 0.60,
                    "split": "test",
                    "n_folds": 1,
                    "n_seeds": 1,
                    "episodes": 1,
                    "cumulative_return": 0.07,
                    "annualized_return": 0.06,
                    "annualized_volatility": 0.10,
                    "sharpe": 0.70,
                    "sortino": 0.90,
                    "calmar": 0.60,
                    "robust_score": 0.70,
                    "mandate_aware_score": 0.55,
                    "max_drawdown": -0.16,
                    "worst_max_drawdown": -0.18,
                    "average_turnover": 0.35,
                    "mean_transaction_cost": 0.0003,
                    "average_effective_number_of_assets": 2.40,
                    "average_max_weight": 0.60,
                    "mean_cash_weight": 0.02,
                    "cash_above_10_rate": 0.0,
                    "concentration_classification": "not_concentrated",
                    "suspicious_or_lazy_concentration_candidate": False,
                    "justified_concentration_candidate": False,
                },
            ]
        )
        rows.to_csv(path / "max_weight_cap_rankings.csv", index=False)
        robust = pd.DataFrame(
            [
                {
                    "strategy": f"{prefix}_cap_uncapped",
                    "robust_score": 0.40,
                    "pooled_dsr_n25": 0.20,
                    "median_run_dsr_n25": 0.10,
                    "date_averaged_dsr_n25": 0.15,
                    "dsr_method": "median_run",
                },
                {
                    "strategy": f"{prefix}_cap_0p60",
                    "robust_score": 0.70,
                    "pooled_dsr_n25": 0.40,
                    "median_run_dsr_n25": 0.30,
                    "date_averaged_dsr_n25": 0.35,
                    "dsr_method": "median_run",
                },
            ]
        )
        robust.to_csv(path / "robust_score_ranking.csv", index=False)
        return path

    def _write_returns(self, temp_dir: str) -> Path:
        dates = pd.date_range("2023-01-06", periods=60, freq="W-FRI")
        returns = pd.DataFrame(
            {
                "date": dates,
                "SPY": [0.01 + 0.001 * ((i % 4) - 1.5) for i in range(60)],
                "TLT": [0.003 + 0.001 * ((i % 3) - 1.0) for i in range(60)],
                "GLD": [0.004 + 0.002 * ((i % 5) - 2.0) for i in range(60)],
                "BTC-USD": [0.012 + 0.010 * ((i % 6) - 2.5) for i in range(60)],
                "CASH": [0.0] * 60,
            }
        )
        path = Path(temp_dir) / "returns.csv"
        returns.to_csv(path, index=False)
        return path


if __name__ == "__main__":
    unittest.main()
