"""Tests for the full cap sensitivity experiment wrapper/reporting."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.experiments.run_cap_sensitivity_experiment import (
    build_cap_sensitivity_all_results,
    build_cap_sensitivity_best_caps,
    build_cap_sensitivity_markdown,
    build_cap_sensitivity_metadata,
    build_cap_sensitivity_pairwise_deltas,
    build_cap_sensitivity_summary,
    format_cap_label,
    label_cap_sensitivity_decision,
    parse_candidate_list,
)
from src.experiments.run_max_weight_cap_experiment import parse_max_weight_grid


class CapSensitivityExperimentTests(unittest.TestCase):
    def test_cap_labels_are_parsed_correctly(self):
        caps = parse_max_weight_grid("uncapped,0.50,0.60,0.70,0.80")

        self.assertEqual(caps, [None, 0.50, 0.60, 0.70, 0.80])
        self.assertEqual([format_cap_label(cap) for cap in caps], ["uncapped", "0.50", "0.60", "0.70", "0.80"])

    def test_candidate_list_parses(self):
        self.assertEqual(
            parse_candidate_list("V2_reference_full,V5_no_volatility_block"),
            ["V2_reference_full", "V5_no_volatility_block"],
        )

    def test_deltas_versus_uncapped_are_computed_per_candidate(self):
        all_results = build_cap_sensitivity_all_results([self._rankings_frame()])

        deltas = build_cap_sensitivity_pairwise_deltas(all_results)
        cap_060 = deltas[
            (deltas["base_candidate"] == "V5_no_volatility_block")
            & (deltas["cap_label"] == "0.60")
        ].iloc[0]

        self.assertAlmostEqual(float(cap_060["delta_robust_score"]), 0.30)
        self.assertAlmostEqual(
            float(cap_060["delta_average_effective_number_of_assets"]),
            1.30,
        )

    def test_best_cap_per_candidate_is_selected_correctly(self):
        all_results = build_cap_sensitivity_all_results([self._rankings_frame()])

        best = build_cap_sensitivity_best_caps(all_results).set_index("base_candidate")

        self.assertEqual(
            best.loc["V5_no_volatility_block", "best_by_mandate_aware_score"],
            "0.60",
        )
        self.assertEqual(
            best.loc["V5_no_volatility_block", "best_by_effective_assets"],
            "0.50",
        )

    def test_decision_labels_work(self):
        row = pd.Series(
            {
                "max_weight_cap": 0.60,
                "delta_average_effective_number_of_assets_vs_baseline": 1.0,
                "delta_robust_score_vs_baseline": 0.10,
                "delta_mandate_aware_score_vs_baseline": 0.10,
                "delta_max_drawdown_vs_baseline": 0.04,
                "delta_average_turnover_vs_baseline": -0.10,
                "delta_annualized_return_vs_baseline": 0.02,
            }
        )
        weak = row.copy()
        weak["delta_robust_score_vs_baseline"] = -0.20
        weak["delta_mandate_aware_score_vs_baseline"] = -0.20

        self.assertEqual(label_cap_sensitivity_decision(row), "cap_dominates_uncapped")
        self.assertEqual(label_cap_sensitivity_decision(weak), "uncapped_preferred")

    def test_uncapped_rows_are_preserved(self):
        all_results = build_cap_sensitivity_all_results([self._rankings_frame()])

        self.assertIn("uncapped", set(all_results["cap_label"]))
        self.assertIn("uncapped_baseline", set(all_results["decision_label"]))

    def test_combined_summary_includes_all_candidates(self):
        frames = [self._rankings_frame("V5_no_volatility_block"), self._rankings_frame("V2_reference_full")]
        all_results = build_cap_sensitivity_all_results(frames)
        best = build_cap_sensitivity_best_caps(all_results)

        summary = build_cap_sensitivity_summary(all_results, best)

        self.assertEqual(
            set(summary["base_candidate"]),
            {"V5_no_volatility_block", "V2_reference_full"},
        )

    def test_metadata_records_experiment_inputs(self):
        metadata = build_cap_sensitivity_metadata(
            returns_path="returns.csv",
            output_dir="out",
            candidates=["V2_reference_full"],
            max_weight_grid=[None, 0.60],
            episodes=60,
            seeds=[7, 21],
            max_folds=None,
            transaction_cost=0.001,
            base_config_path="configs/config.yaml",
            reports={"V2_reference_full": {"output_dir": "candidate_out"}},
        )

        self.assertEqual(metadata["returns_path"], "returns.csv")
        self.assertEqual(metadata["episodes"], 60)
        self.assertEqual(metadata["max_weight_grid"], ["uncapped", 0.60])
        self.assertIn("V2_reference_full", metadata["candidate_output_dirs"])

    def test_output_markdown_summary_is_created(self):
        all_results = build_cap_sensitivity_all_results([self._rankings_frame()])
        best = build_cap_sensitivity_best_caps(all_results)
        summary = build_cap_sensitivity_summary(all_results, best)

        markdown = build_cap_sensitivity_markdown(summary, best)

        self.assertIn("Cap Sensitivity Experiment Summary", markdown)
        self.assertIn("V5_no_volatility_block", markdown)

    def test_write_outputs_smoke_with_synthetic_tables(self):
        from src.experiments.run_cap_sensitivity_experiment import (
            write_cap_sensitivity_outputs,
        )

        all_results = build_cap_sensitivity_all_results([self._rankings_frame()])
        deltas = build_cap_sensitivity_pairwise_deltas(all_results)
        best = build_cap_sensitivity_best_caps(all_results)
        summary = build_cap_sensitivity_summary(all_results, best)
        markdown = build_cap_sensitivity_markdown(summary, best)
        metadata = {"test": True}
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_cap_sensitivity_outputs(
                output_dir=Path(temp_dir),
                all_results=all_results,
                pairwise_deltas=deltas,
                best_caps=best,
                summary=summary,
                markdown=markdown,
                metadata=metadata,
            )

            self.assertTrue(Path(paths["all_results"]).exists())
            self.assertTrue(Path(paths["metadata"]).exists())

    def _rankings_frame(self, base_candidate: str = "V5_no_volatility_block") -> pd.DataFrame:
        prefix = base_candidate
        return pd.DataFrame(
            [
                {
                    "candidate_name": f"{prefix}_cap_uncapped",
                    "base_candidate": base_candidate,
                    "max_weight_cap": pd.NA,
                    "split": "test",
                    "mandate_aware_score": 0.20,
                    "robust_score": 0.30,
                    "annualized_return": 0.04,
                    "sharpe": 0.30,
                    "max_drawdown": -0.25,
                    "average_turnover": 0.60,
                    "average_effective_number_of_assets": 1.10,
                    "average_max_weight": 0.96,
                    "delta_mandate_aware_score_vs_baseline": 0.0,
                    "delta_robust_score_vs_baseline": 0.0,
                    "delta_annualized_return_vs_baseline": 0.0,
                    "delta_sharpe_vs_baseline": 0.0,
                    "delta_max_drawdown_vs_baseline": 0.0,
                    "delta_average_turnover_vs_baseline": 0.0,
                    "delta_average_effective_number_of_assets_vs_baseline": 0.0,
                    "delta_average_max_weight_vs_baseline": 0.0,
                },
                {
                    "candidate_name": f"{prefix}_cap_0p60",
                    "base_candidate": base_candidate,
                    "max_weight_cap": 0.60,
                    "split": "test",
                    "mandate_aware_score": 0.50,
                    "robust_score": 0.60,
                    "annualized_return": 0.07,
                    "sharpe": 0.70,
                    "max_drawdown": -0.16,
                    "average_turnover": 0.35,
                    "average_effective_number_of_assets": 2.40,
                    "average_max_weight": 0.60,
                    "delta_mandate_aware_score_vs_baseline": 0.30,
                    "delta_robust_score_vs_baseline": 0.30,
                    "delta_annualized_return_vs_baseline": 0.03,
                    "delta_sharpe_vs_baseline": 0.40,
                    "delta_max_drawdown_vs_baseline": 0.09,
                    "delta_average_turnover_vs_baseline": -0.25,
                    "delta_average_effective_number_of_assets_vs_baseline": 1.30,
                    "delta_average_max_weight_vs_baseline": -0.36,
                },
                {
                    "candidate_name": f"{prefix}_cap_0p50",
                    "base_candidate": base_candidate,
                    "max_weight_cap": 0.50,
                    "split": "test",
                    "mandate_aware_score": 0.40,
                    "robust_score": 0.45,
                    "annualized_return": 0.02,
                    "sharpe": 0.50,
                    "max_drawdown": -0.18,
                    "average_turnover": 0.40,
                    "average_effective_number_of_assets": 2.90,
                    "average_max_weight": 0.50,
                    "delta_mandate_aware_score_vs_baseline": 0.20,
                    "delta_robust_score_vs_baseline": 0.15,
                    "delta_annualized_return_vs_baseline": -0.02,
                    "delta_sharpe_vs_baseline": 0.20,
                    "delta_max_drawdown_vs_baseline": 0.07,
                    "delta_average_turnover_vs_baseline": -0.20,
                    "delta_average_effective_number_of_assets_vs_baseline": 1.80,
                    "delta_average_max_weight_vs_baseline": -0.46,
                },
            ]
        )


if __name__ == "__main__":
    unittest.main()
