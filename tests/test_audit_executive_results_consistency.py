"""Tests for executive results consistency audit."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.audit_executive_results_consistency import (
    audit_executive_results_consistency,
    build_consistency_summary,
    check_capped_td3_above_buyhold_gld,
    check_duplicate_strategy_rows,
    check_eligible_scores,
    check_executive_has_no_train_validation_rows,
    check_not_eligible_scores,
    load_audit_inputs,
)


class ExecutiveResultsConsistencyAuditTests(unittest.TestCase):
    def test_duplicate_strategy_detection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir, executive_dir = self._write_inputs(temp_dir)
            duplicate = pd.read_csv(comparison_dir / "capped_td3_vs_benchmarks_summary.csv")
            duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
            duplicate.to_csv(
                comparison_dir / "capped_td3_vs_benchmarks_summary.csv",
                index=False,
            )
            inputs = load_audit_inputs(comparison_dir, executive_dir)

            check = check_duplicate_strategy_rows(inputs)

        self.assertEqual(check["status"], "fail")
        self.assertIn("V5_cap_0.60", check["details"])

    def test_non_test_split_detection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir, executive_dir = self._write_inputs(temp_dir)
            source_path = Path(temp_dir) / "v5_source"
            ranking = pd.read_csv(source_path / "max_weight_cap_rankings.csv")
            ranking.loc[
                ranking["candidate_name"] == "V5_no_volatility_block_cap_0p60",
                "split",
            ] = "validation"
            ranking.to_csv(source_path / "max_weight_cap_rankings.csv", index=False)
            inputs = load_audit_inputs(comparison_dir, executive_dir)

            check = check_executive_has_no_train_validation_rows(inputs)

        self.assertEqual(check["status"], "fail")
        self.assertIn("non-test", check["details"])

    def test_non_eligible_benchmark_with_positive_mandate_score_is_flagged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir, executive_dir = self._write_inputs(temp_dir)
            summary = pd.read_csv(comparison_dir / "capped_td3_vs_benchmarks_summary.csv")
            summary.loc[
                summary["strategy_name"] == "momentum_winner_12p",
                "mandate_aware_score",
            ] = 0.10
            summary.to_csv(
                comparison_dir / "capped_td3_vs_benchmarks_summary.csv",
                index=False,
            )
            inputs = load_audit_inputs(comparison_dir, executive_dir)

            check = check_not_eligible_scores(inputs)

        self.assertEqual(check["status"], "fail")
        self.assertIn("momentum_winner_12p", check["details"])

    def test_eligible_strategy_with_valid_score_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir, executive_dir = self._write_inputs(temp_dir)
            inputs = load_audit_inputs(comparison_dir, executive_dir)

            check = check_eligible_scores(inputs)

        self.assertEqual(check["status"], "pass")

    def test_v5_v2_above_buyhold_gld_check_works(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir, executive_dir = self._write_inputs(temp_dir)
            inputs = load_audit_inputs(comparison_dir, executive_dir)

            check = check_capped_td3_above_buyhold_gld(inputs)

        self.assertEqual(check["status"], "pass")

    def test_markdown_summary_is_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir, executive_dir = self._write_inputs(temp_dir)
            output_dir = Path(temp_dir) / "audit"

            result = audit_executive_results_consistency(
                comparison_dir=str(comparison_dir),
                executive_dir=str(executive_dir),
                output_dir=str(output_dir),
            )

            summary_path = Path(result["paths"]["summary"])
            summary_exists = summary_path.exists()

        self.assertTrue(summary_exists)
        self.assertIn("Final verdict", result["summary_markdown"])

    def test_summary_marks_fail_as_not_reliable(self):
        checks = pd.DataFrame(
            [
                {
                    "check_id": "bad",
                    "description": "bad",
                    "status": "fail",
                    "details": "broken",
                }
            ]
        )

        markdown = build_consistency_summary(checks, checks)

        self.assertIn("not yet reliable", markdown)

    def _write_inputs(self, temp_dir: str) -> tuple[Path, Path]:
        root = Path(temp_dir)
        comparison_dir = root / "comparison"
        executive_dir = root / "executive"
        benchmark_dir = comparison_dir / "benchmarks"
        comparison_dir.mkdir()
        executive_dir.mkdir()
        benchmark_dir.mkdir()
        for filename in [
            "benchmark_metrics_table.csv",
            "benchmark_comparison_summary.csv",
            "benchmark_diagnostics.csv",
        ]:
            pd.DataFrame({"benchmark_name": ["BuyHold_GLD"]}).to_csv(
                benchmark_dir / filename,
                index=False,
            )

        v2_source = root / "v2_source"
        v5_source = root / "v5_source"
        v6_source = root / "v6_source"
        self._write_source_ranking(v2_source, "V2_reference_full_cap_0p60")
        self._write_source_ranking(v5_source, "V5_no_volatility_block_cap_0p60")
        self._write_source_ranking(v6_source, "V6_financial_state_cap_0p60")

        summary = self._summary_frame(v2_source, v5_source, v6_source)
        summary.to_csv(comparison_dir / "capped_td3_vs_benchmarks_summary.csv", index=False)
        self._pairwise_frame().to_csv(
            comparison_dir / "capped_td3_pairwise_deltas.csv",
            index=False,
        )
        (comparison_dir / "capped_td3_protocol_metadata.json").write_text(
            (
                "{"
                f"\"returns_path\":\"data/processed/returns_weekly_latest.csv\","
                f"\"benchmark_output_dir\":\"{benchmark_dir}\","
                "\"timing_convention\":\"information through t-1, weights for t\","
                "\"turnover_convention\":\"sum(abs(w_t - w_{t-1}))\","
                "\"benchmark_robust_score_note\":\"Benchmark DSR caveat.\","
                "\"input_experiment_folders\":{"
                f"\"V2_reference_full\":\"{v2_source}\","
                f"\"V5_no_volatility_block\":\"{v5_source}\","
                f"\"V6_financial_state\":\"{v6_source}\""
                "}}"
            ),
            encoding="utf-8",
        )

        executive_main = summary[
            [
                "strategy_name",
                "strategy_type",
                "constraint_status",
                "robust_score",
                "mandate_aware_score",
                "mandate_bucket",
                "annualized_return",
                "max_drawdown",
                "sharpe",
                "average_turnover",
                "average_effective_number_of_assets",
                "average_max_weight",
            ]
        ].copy()
        executive_main["rank_robust"] = range(1, len(executive_main) + 1)
        executive_main.to_csv(executive_dir / "executive_main_ranking.csv", index=False)
        executive_main[executive_main["mandate_bucket"] != "not_eligible"].to_csv(
            executive_dir / "executive_mandate_eligible_ranking.csv",
            index=False,
        )
        executive_main[executive_main["mandate_bucket"] == "not_eligible"].to_csv(
            executive_dir / "executive_non_eligible_strategies.csv",
            index=False,
        )
        return comparison_dir, executive_dir

    def _write_source_ranking(self, path: Path, candidate_name: str) -> None:
        path.mkdir()
        pd.DataFrame(
            [
                {
                    "candidate_name": candidate_name,
                    "split": "test",
                }
            ]
        ).to_csv(path / "max_weight_cap_rankings.csv", index=False)

    def _summary_frame(self, v2_source: Path, v5_source: Path, v6_source: Path) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strategy_name": "V5_cap_0.60",
                    "strategy_type": "td3_capped",
                    "candidate_name": "V5_no_volatility_block_cap_0p60",
                    "source_path": str(v5_source),
                    "robust_score": 0.70,
                    "mandate_aware_score": 0.56,
                    "mandate_bucket": "clean_mandate",
                    "max_drawdown": -0.16,
                    "annualized_return": 0.08,
                    "sharpe": 0.70,
                    "average_turnover": 0.35,
                    "average_effective_number_of_assets": 2.4,
                    "average_max_weight": 0.60,
                    "dsr_method": "median_run",
                    "constraint_status": "cap_0.60",
                },
                {
                    "strategy_name": "V2_cap_0.60",
                    "strategy_type": "td3_capped",
                    "candidate_name": "V2_reference_full_cap_0p60",
                    "source_path": str(v2_source),
                    "robust_score": 0.68,
                    "mandate_aware_score": 0.54,
                    "mandate_bucket": "clean_mandate",
                    "max_drawdown": -0.17,
                    "annualized_return": 0.09,
                    "sharpe": 0.75,
                    "average_turnover": 0.36,
                    "average_effective_number_of_assets": 2.4,
                    "average_max_weight": 0.60,
                    "dsr_method": "median_run",
                    "constraint_status": "cap_0.60",
                },
                {
                    "strategy_name": "V6_cap_0.60",
                    "strategy_type": "td3_capped",
                    "candidate_name": "V6_financial_state_cap_0p60",
                    "source_path": str(v6_source),
                    "robust_score": 0.65,
                    "mandate_aware_score": 0.50,
                    "mandate_bucket": "clean_mandate",
                    "max_drawdown": -0.18,
                    "annualized_return": 0.12,
                    "sharpe": 0.50,
                    "average_turnover": 0.30,
                    "average_effective_number_of_assets": 2.4,
                    "average_max_weight": 0.60,
                    "dsr_method": "median_run",
                    "constraint_status": "cap_0.60",
                },
                {
                    "strategy_name": "BuyHold_GLD",
                    "strategy_type": "benchmark",
                    "candidate_name": pd.NA,
                    "source_path": "protocol_benchmark_runner",
                    "robust_score": 0.69,
                    "mandate_aware_score": 0.52,
                    "mandate_bucket": "clean_mandate",
                    "max_drawdown": -0.198,
                    "annualized_return": 0.11,
                    "sharpe": 0.80,
                    "average_turnover": 0.01,
                    "average_effective_number_of_assets": 1.0,
                    "average_max_weight": 1.0,
                    "dsr_method": "date_averaged",
                    "constraint_status": "benchmark",
                },
                {
                    "strategy_name": "momentum_winner_12p",
                    "strategy_type": "benchmark",
                    "candidate_name": pd.NA,
                    "source_path": "protocol_benchmark_runner",
                    "robust_score": 0.90,
                    "mandate_aware_score": 0.0,
                    "mandate_bucket": "not_eligible",
                    "max_drawdown": -0.45,
                    "annualized_return": 0.40,
                    "sharpe": 1.30,
                    "average_turnover": 0.40,
                    "average_effective_number_of_assets": 1.0,
                    "average_max_weight": 1.0,
                    "dsr_method": "date_averaged",
                    "constraint_status": "benchmark",
                },
            ]
        )

    def _pairwise_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "base_candidate": "V5_no_volatility_block",
                    "summary_decision": "cap_dominates_uncapped",
                }
            ]
        )


if __name__ == "__main__":
    unittest.main()
