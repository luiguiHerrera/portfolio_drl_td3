"""Tests for executive capped TD3 results reporting."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.build_executive_results_report import (
    build_executive_results_report,
    build_mandate_eligible_ranking,
    build_non_eligible_strategies,
    build_strategy_groups_summary,
    build_td3_cap_impact,
    classify_strategy_group,
)


class ExecutiveResultsReportTests(unittest.TestCase):
    def test_eligible_ranking_excludes_not_eligible_strategies(self):
        summary = self._summary_frame()

        eligible = build_mandate_eligible_ranking(summary)

        self.assertNotIn("momentum_winner_12p", set(eligible["strategy_name"]))
        self.assertIn("V5_cap_0.60", set(eligible["strategy_name"]))

    def test_non_eligible_file_includes_high_drawdown_strategies(self):
        summary = self._summary_frame()

        non_eligible = build_non_eligible_strategies(summary)

        self.assertIn("momentum_winner_12p", set(non_eligible["strategy_name"]))
        reason = non_eligible.set_index("strategy_name").loc[
            "momentum_winner_12p",
            "reason_not_eligible",
        ]
        self.assertIn("-30%", reason)

    def test_td3_cap_impact_deltas_are_computed_correctly(self):
        impact = build_td3_cap_impact(self._pairwise_frame())

        v5 = impact.set_index("base_candidate").loc["V5_no_volatility_block"]

        self.assertAlmostEqual(float(v5["delta_mandate_aware_score"]), 0.45)
        self.assertAlmostEqual(float(v5["delta_effective_assets"]), 1.30)
        self.assertEqual(v5["pairwise_decision"], "cap_dominates_uncapped")

    def test_group_summary_works(self):
        groups = build_strategy_groups_summary(self._summary_frame())

        names = set(groups["group"])

        self.assertIn("benchmark_eligible", names)
        self.assertIn("benchmark_not_eligible", names)
        self.assertIn("td3_capped", names)
        self.assertIn("td3_uncapped", names)

    def test_markdown_summary_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = self._write_input_dir(temp_dir)
            output_dir = Path(temp_dir) / "executive"

            report = build_executive_results_report(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
            )

            markdown_path = Path(report["paths"]["markdown_summary"])
            markdown_exists = markdown_path.exists()

        self.assertTrue(markdown_exists)
        self.assertIn("Suggested Claim", report["markdown_summary"])

    def test_benchmark_and_td3_strategy_types_are_classified_correctly(self):
        self.assertEqual(
            classify_strategy_group(
                pd.Series(
                    {
                        "strategy_type": "benchmark",
                        "is_mandate_eligible": True,
                    }
                )
            ),
            "benchmark_eligible",
        )
        self.assertEqual(
            classify_strategy_group(pd.Series({"strategy_type": "td3_capped"})),
            "td3_capped",
        )
        self.assertEqual(
            classify_strategy_group(pd.Series({"strategy_type": "td3_uncapped"})),
            "td3_uncapped",
        )

    def test_full_report_writes_all_expected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = self._write_input_dir(temp_dir)
            output_dir = Path(temp_dir) / "executive"

            report = build_executive_results_report(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
            )

            for path in report["paths"].values():
                self.assertTrue(Path(path).exists())

    def _write_input_dir(self, temp_dir: str) -> Path:
        input_dir = Path(temp_dir) / "comparison"
        input_dir.mkdir()
        self._summary_frame().to_csv(
            input_dir / "capped_td3_vs_benchmarks_summary.csv",
            index=False,
        )
        self._pairwise_frame().to_csv(
            input_dir / "capped_td3_pairwise_deltas.csv",
            index=False,
        )
        return input_dir

    def _summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strategy_name": "momentum_winner_12p",
                    "strategy_type": "benchmark",
                    "constraint_status": "benchmark",
                    "robust_score": 0.90,
                    "mandate_aware_score": 0.00,
                    "mandate_bucket": "not_eligible",
                    "annualized_return": 0.40,
                    "annualized_volatility": 0.30,
                    "sharpe": 1.30,
                    "sortino": 2.00,
                    "calmar": 1.00,
                    "max_drawdown": -0.45,
                    "recovery_required": 0.82,
                    "average_turnover": 0.40,
                    "average_effective_number_of_assets": 1.0,
                    "average_max_weight": 1.0,
                    "is_mandate_eligible": False,
                },
                {
                    "strategy_name": "BuyHold_GLD",
                    "strategy_type": "benchmark",
                    "constraint_status": "benchmark",
                    "robust_score": 0.65,
                    "mandate_aware_score": 0.50,
                    "mandate_bucket": "clean_mandate",
                    "annualized_return": 0.10,
                    "annualized_volatility": 0.14,
                    "sharpe": 0.80,
                    "sortino": 1.10,
                    "calmar": 0.60,
                    "max_drawdown": -0.18,
                    "recovery_required": 0.22,
                    "average_turnover": 0.01,
                    "average_effective_number_of_assets": 1.0,
                    "average_max_weight": 1.0,
                    "is_mandate_eligible": True,
                },
                {
                    "strategy_name": "V5_cap_0.60",
                    "strategy_type": "td3_capped",
                    "constraint_status": "cap_0.60",
                    "robust_score": 0.70,
                    "mandate_aware_score": 0.56,
                    "mandate_bucket": "clean_mandate",
                    "annualized_return": 0.08,
                    "annualized_volatility": 0.17,
                    "sharpe": 0.70,
                    "sortino": 2.00,
                    "calmar": 2.50,
                    "max_drawdown": -0.16,
                    "recovery_required": 0.19,
                    "average_turnover": 0.35,
                    "average_effective_number_of_assets": 2.4,
                    "average_max_weight": 0.60,
                    "is_mandate_eligible": True,
                },
                {
                    "strategy_name": "V5_uncapped",
                    "strategy_type": "td3_uncapped",
                    "constraint_status": "uncapped",
                    "robust_score": 0.15,
                    "mandate_aware_score": 0.10,
                    "mandate_bucket": "eligible_yellow",
                    "annualized_return": 0.04,
                    "annualized_volatility": 0.25,
                    "sharpe": 0.35,
                    "sortino": 0.80,
                    "calmar": 0.50,
                    "max_drawdown": -0.23,
                    "recovery_required": 0.30,
                    "average_turnover": 0.60,
                    "average_effective_number_of_assets": 1.1,
                    "average_max_weight": 0.96,
                    "is_mandate_eligible": True,
                },
            ]
        )

    def _pairwise_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "base_candidate": "V5_no_volatility_block",
                    "uncapped_mandate_aware_score": 0.10,
                    "capped_mandate_aware_score": 0.55,
                    "delta_mandate_aware_score": 0.45,
                    "uncapped_robust_score": 0.15,
                    "capped_robust_score": 0.70,
                    "delta_robust_score": 0.55,
                    "uncapped_annualized_return": 0.04,
                    "capped_annualized_return": 0.08,
                    "delta_annualized_return": 0.04,
                    "uncapped_sharpe": 0.35,
                    "capped_sharpe": 0.70,
                    "delta_sharpe": 0.35,
                    "uncapped_max_drawdown": -0.23,
                    "capped_max_drawdown": -0.16,
                    "delta_max_drawdown": 0.07,
                    "uncapped_average_turnover": 0.60,
                    "capped_average_turnover": 0.35,
                    "delta_average_turnover": -0.25,
                    "uncapped_average_effective_number_of_assets": 1.10,
                    "capped_average_effective_number_of_assets": 2.40,
                    "delta_average_effective_number_of_assets": 1.30,
                    "uncapped_average_max_weight": 0.96,
                    "capped_average_max_weight": 0.60,
                    "delta_average_max_weight": -0.36,
                    "summary_decision": "cap_dominates_uncapped",
                }
            ]
        )


if __name__ == "__main__":
    unittest.main()
