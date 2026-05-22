"""Tests for final constrained TD3 report builder."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.build_final_constrained_td3_report import (
    build_final_constrained_td3_report,
    build_final_combined_table,
    build_selected_td3_rows,
)


class FinalConstrainedTD3ReportTests(unittest.TestCase):
    def test_selected_best_caps_are_identified(self):
        selected = build_selected_td3_rows(
            self._cap_results(),
            self._best_caps(),
        )

        best = selected[selected["strategy_group"] == "td3_best_constrained"]

        self.assertIn("V2_cap_0.50", set(best["strategy_name"]))
        self.assertIn("V5_cap_0.70", set(best["strategy_name"]))

    def test_benchmarks_are_included_and_classified(self):
        selected = build_selected_td3_rows(self._cap_results(), self._best_caps())
        combined = build_final_combined_table(selected, self._benchmark_rows())

        by_name = combined.set_index("strategy_name")

        self.assertEqual(by_name.loc["BuyHold_GLD", "strategy_group"], "benchmark_eligible")
        self.assertEqual(
            by_name.loc["momentum_winner_12p", "strategy_group"],
            "benchmark_not_eligible",
        )

    def test_uncapped_td3_rows_are_preserved(self):
        selected = build_selected_td3_rows(self._cap_results(), self._best_caps())

        self.assertIn("V2_uncapped", set(selected["strategy_name"]))
        self.assertIn("td3_uncapped", set(selected["strategy_group"]))

    def test_cap_060_reference_rows_are_marked_separately(self):
        selected = build_selected_td3_rows(self._cap_results(), self._best_caps())
        reference = selected[selected["strategy_group"] == "td3_cap_0.60_reference"]

        self.assertIn("V2_cap_0.60", set(reference["strategy_name"]))
        self.assertNotIn(
            "td3_best_constrained",
            set(reference["strategy_group"]),
        )

    def test_mandate_ranking_works(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cap_dir, benchmark_dir = self._write_inputs(temp_dir)

            report = build_final_constrained_td3_report(
                cap_sensitivity_dir=str(cap_dir),
                benchmark_comparison_dir=str(benchmark_dir),
                output_dir=str(Path(temp_dir) / "out"),
            )

            top = report["mandate_ranking"].iloc[0]

        self.assertEqual(top["strategy_name"], "V5_cap_0.70")

    def test_interpretation_flags_work(self):
        selected = build_selected_td3_rows(self._cap_results(), self._best_caps())
        combined = build_final_combined_table(selected, self._benchmark_rows())
        v5 = combined.set_index("strategy_name").loc["V5_cap_0.70"]

        self.assertTrue(bool(v5["beats_best_clean_benchmark_by_mandate"]))
        self.assertTrue(bool(v5["beats_uncapped_by_mandate"]))
        self.assertTrue(bool(v5["concentration_controlled"]))

    def test_summary_markdown_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cap_dir, benchmark_dir = self._write_inputs(temp_dir)

            report = build_final_constrained_td3_report(
                cap_sensitivity_dir=str(cap_dir),
                benchmark_comparison_dir=str(benchmark_dir),
                output_dir=str(Path(temp_dir) / "out"),
            )

        self.assertIn("Final Defensible Claim", report["markdown_summary"])

    def test_metadata_includes_source_dirs_and_selected_caps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cap_dir, benchmark_dir = self._write_inputs(temp_dir)
            output_dir = Path(temp_dir) / "out"

            report = build_final_constrained_td3_report(
                cap_sensitivity_dir=str(cap_dir),
                benchmark_comparison_dir=str(benchmark_dir),
                output_dir=str(output_dir),
            )
            metadata = json.loads(Path(report["paths"]["metadata"]).read_text())

        self.assertEqual(metadata["cap_sensitivity_dir"], str(cap_dir))
        self.assertEqual(metadata["benchmark_comparison_dir"], str(benchmark_dir))
        self.assertEqual(metadata["selected_caps"]["V2_reference_full"], 0.50)

    def _write_inputs(self, temp_dir: str) -> tuple[Path, Path]:
        cap_dir = Path(temp_dir) / "cap"
        benchmark_dir = Path(temp_dir) / "bench"
        cap_dir.mkdir()
        benchmark_dir.mkdir()
        self._cap_results().to_csv(cap_dir / "cap_sensitivity_all_results.csv", index=False)
        self._best_caps().to_csv(cap_dir / "cap_sensitivity_best_caps.csv", index=False)
        self._benchmark_rows().to_csv(
            benchmark_dir / "capped_td3_vs_benchmarks_summary.csv",
            index=False,
        )
        return cap_dir, benchmark_dir

    def _cap_results(self) -> pd.DataFrame:
        rows = []
        for base, best_cap in [
            ("V2_reference_full", 0.50),
            ("V5_no_volatility_block", 0.70),
        ]:
            prefix = "V2" if base == "V2_reference_full" else "V5"
            rows.extend(
                [
                    {
                        "candidate_name": f"{base}_cap_uncapped",
                        "base_candidate": base,
                        "max_weight_cap": pd.NA,
                        "cap_label": "uncapped",
                        "robust_score": 0.20,
                        "mandate_aware_score": 0.10,
                        "annualized_return": 0.04,
                        "annualized_volatility": 0.20,
                        "sharpe": 0.30,
                        "sortino": 0.50,
                        "calmar": 0.40,
                        "max_drawdown": -0.24,
                        "average_turnover": 0.60,
                        "average_effective_number_of_assets": 1.10,
                        "average_max_weight": 0.96,
                        "decision_label": "uncapped_baseline",
                    },
                    {
                        "candidate_name": f"{base}_cap_0p60",
                        "base_candidate": base,
                        "max_weight_cap": 0.60,
                        "cap_label": "0.60",
                        "robust_score": 0.50,
                        "mandate_aware_score": 0.40,
                        "annualized_return": 0.06,
                        "annualized_volatility": 0.15,
                        "sharpe": 0.60,
                        "sortino": 1.00,
                        "calmar": 1.00,
                        "max_drawdown": -0.18,
                        "average_turnover": 0.40,
                        "average_effective_number_of_assets": 2.40,
                        "average_max_weight": 0.60,
                        "decision_label": "cap_dominates_uncapped",
                    },
                    {
                        "candidate_name": f"{base}_cap_{str(best_cap).replace('.', 'p')}",
                        "base_candidate": base,
                        "max_weight_cap": best_cap,
                        "cap_label": f"{best_cap:.2f}",
                        "robust_score": 0.70 if prefix == "V5" else 0.65,
                        "mandate_aware_score": 0.58 if prefix == "V5" else 0.54,
                        "annualized_return": 0.08,
                        "annualized_volatility": 0.14,
                        "sharpe": 0.80,
                        "sortino": 1.20,
                        "calmar": 1.40,
                        "max_drawdown": -0.16,
                        "average_turnover": 0.35,
                        "average_effective_number_of_assets": 2.70,
                        "average_max_weight": 0.55,
                        "decision_label": "cap_dominates_uncapped",
                    },
                ]
            )
        return pd.DataFrame(rows)

    def _best_caps(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "base_candidate": "V2_reference_full",
                    "best_by_mandate_aware_score": 0.50,
                    "best_mandate_aware_score": 0.54,
                    "best_by_robust_score": 0.50,
                    "best_robust_score": 0.65,
                },
                {
                    "base_candidate": "V5_no_volatility_block",
                    "best_by_mandate_aware_score": 0.70,
                    "best_mandate_aware_score": 0.58,
                    "best_by_robust_score": 0.70,
                    "best_robust_score": 0.70,
                },
            ]
        )

    def _benchmark_rows(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strategy_name": "BuyHold_GLD",
                    "strategy_type": "benchmark",
                    "robust_score": 0.60,
                    "mandate_aware_score": 0.50,
                    "mandate_bucket": "clean_mandate",
                    "annualized_return": 0.10,
                    "annualized_volatility": 0.12,
                    "sharpe": 0.75,
                    "sortino": 1.10,
                    "calmar": 0.80,
                    "max_drawdown": -0.18,
                    "average_turnover": 0.01,
                    "average_effective_number_of_assets": 1.0,
                    "average_max_weight": 1.0,
                },
                {
                    "strategy_name": "momentum_winner_12p",
                    "strategy_type": "benchmark",
                    "robust_score": 0.90,
                    "mandate_aware_score": 0.00,
                    "mandate_bucket": "not_eligible",
                    "annualized_return": 0.40,
                    "annualized_volatility": 0.35,
                    "sharpe": 1.20,
                    "sortino": 2.00,
                    "calmar": 1.00,
                    "max_drawdown": -0.45,
                    "average_turnover": 0.40,
                    "average_effective_number_of_assets": 1.0,
                    "average_max_weight": 1.0,
                },
                {
                    "strategy_name": "trend_spy_cash_12p",
                    "strategy_type": "benchmark",
                    "robust_score": 0.55,
                    "mandate_aware_score": 0.42,
                    "mandate_bucket": "clean_mandate",
                    "annualized_return": 0.08,
                    "annualized_volatility": 0.10,
                    "sharpe": 0.70,
                    "sortino": 1.00,
                    "calmar": 0.70,
                    "max_drawdown": -0.19,
                    "average_turnover": 0.20,
                    "average_effective_number_of_assets": 1.0,
                    "average_max_weight": 1.0,
                },
            ]
        )


if __name__ == "__main__":
    unittest.main()
