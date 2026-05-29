"""Tests for mandate profile comparison reporting."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.mandate_profile_comparison_report import (
    KEY_CANDIDATE_V3_CLEAN,
    build_mandate_profile_comparison_report,
    build_profile_rankings,
    build_profile_winners,
    score_strategies_for_profiles,
)
from src.risk.mandate_profiles import get_default_mandate_profiles


class MandateProfileComparisonReportTests(unittest.TestCase):
    def test_profile_thresholds_loaded_correctly(self):
        profiles = get_default_mandate_profiles()

        self.assertEqual(profiles["conservative"].max_drawdown_limit, -0.10)
        self.assertEqual(profiles["moderate"].max_weight_limit, 0.80)
        self.assertEqual(profiles["aggressive"].max_turnover_limit, 1.50)

    def test_conservative_penalizes_risk_more_than_aggressive(self):
        strategies = pd.DataFrame(
            [
                {
                    "strategy_name": "risky",
                    "strategy_type": "td3",
                    "robust_score": 0.9,
                    "max_drawdown": -0.30,
                    "annualized_volatility": 0.30,
                    "average_max_weight": 1.0,
                    "average_effective_number_of_assets": 1.0,
                    "average_turnover": 1.0,
                }
            ]
        )

        scores = score_strategies_for_profiles(strategies)
        conservative = scores[scores["profile"] == "conservative"].iloc[0]
        aggressive = scores[scores["profile"] == "aggressive"].iloc[0]

        self.assertLess(conservative["profile_score"], aggressive["profile_score"])
        self.assertFalse(bool(conservative["profile_eligible"]))
        self.assertTrue(bool(aggressive["profile_eligible"]))

    def test_ranking_changes_when_thresholds_differ(self):
        strategies = pd.DataFrame(
            [
                {
                    "strategy_name": "safe_lower_score",
                    "strategy_type": "td3",
                    "robust_score": 0.5,
                    "max_drawdown": -0.08,
                    "annualized_volatility": 0.10,
                    "average_max_weight": 0.50,
                    "average_effective_number_of_assets": 2.0,
                    "average_turnover": 0.20,
                },
                {
                    "strategy_name": "risky_higher_score",
                    "strategy_type": "benchmark",
                    "robust_score": 0.9,
                    "max_drawdown": -0.30,
                    "annualized_volatility": 0.30,
                    "average_max_weight": 1.00,
                    "average_effective_number_of_assets": 1.0,
                    "average_turnover": 1.00,
                },
            ]
        )

        rankings = build_profile_rankings(score_strategies_for_profiles(strategies))
        conservative_top = rankings[rankings["profile"] == "conservative"].iloc[0]
        aggressive_top = rankings[rankings["profile"] == "aggressive"].iloc[0]

        self.assertEqual(conservative_top["strategy_name"], "safe_lower_score")
        self.assertEqual(aggressive_top["strategy_name"], "risky_higher_score")

    def test_winners_identify_best_td3_and_benchmark(self):
        strategies = pd.DataFrame(
            [
                {
                    "strategy_name": KEY_CANDIDATE_V3_CLEAN,
                    "strategy_type": "td3",
                    "robust_score": 0.7,
                    "max_drawdown": -0.09,
                    "annualized_volatility": 0.12,
                    "average_max_weight": 0.50,
                    "average_effective_number_of_assets": 3.0,
                    "average_turnover": 0.20,
                },
                {
                    "strategy_name": "BuyHold_GLD",
                    "strategy_type": "benchmark",
                    "robust_score": 0.6,
                    "max_drawdown": -0.15,
                    "annualized_volatility": 0.16,
                    "average_max_weight": 1.00,
                    "average_effective_number_of_assets": 1.0,
                    "average_turnover": 0.0,
                },
            ]
        )

        winners = build_profile_winners(build_profile_rankings(score_strategies_for_profiles(strategies)))

        conservative = winners[winners["profile"] == "conservative"].iloc[0]
        self.assertEqual(conservative["best_td3_candidate"], KEY_CANDIDATE_V3_CLEAN)
        self.assertEqual(conservative["best_benchmark"], "BuyHold_GLD")

    def test_output_files_are_created_and_summary_is_reporting_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_dir = root / "final"
            output_dir = root / "out"
            final_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "strategy_name": KEY_CANDIDATE_V3_CLEAN,
                        "strategy_type": "td3",
                        "strategy_group": "td3_best_constrained",
                        "robust_score": 0.7,
                        "mandate_aware_score": 0.6,
                        "annualized_return": 0.10,
                        "annualized_volatility": 0.12,
                        "sharpe": 0.8,
                        "max_drawdown": -0.09,
                        "average_turnover": 0.20,
                        "average_effective_number_of_assets": 3.0,
                        "average_max_weight": 0.50,
                    },
                    {
                        "strategy_name": "BuyHold_GLD",
                        "strategy_type": "benchmark",
                        "strategy_group": "benchmark_eligible",
                        "robust_score": 0.6,
                        "mandate_aware_score": 0.5,
                        "annualized_return": 0.08,
                        "annualized_volatility": 0.16,
                        "sharpe": 0.5,
                        "max_drawdown": -0.15,
                        "average_turnover": 0.0,
                        "average_effective_number_of_assets": 1.0,
                        "average_max_weight": 1.0,
                    },
                ]
            ).to_csv(final_dir / "final_constrained_td3_main_ranking.csv", index=False)

            result = build_mandate_profile_comparison_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
            )

            self.assertTrue((output_dir / "mandate_profile_strategy_scores.csv").exists())
            self.assertTrue((output_dir / "mandate_profile_winners.csv").exists())
            self.assertTrue((output_dir / "mandate_profile_rankings.csv").exists())
            self.assertTrue((output_dir / "mandate_profile_summary.md").exists())
            self.assertTrue((output_dir / "mandate_profile_metadata.json").exists())
            self.assertIn("reporting-only", result["summary"])
            self.assertIn("does not retrain", result["summary"])

            metadata = json.loads((output_dir / "mandate_profile_metadata.json").read_text())
            self.assertTrue(metadata["reporting_only"])
            self.assertIn("conservative", metadata["profile_thresholds"])


if __name__ == "__main__":
    unittest.main()
