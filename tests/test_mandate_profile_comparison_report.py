"""Tests for mandate profile comparison reporting."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.mandate_profile_comparison_report import (
    KEY_CANDIDATE_V3_CLEAN,
    KEY_CANDIDATE_V3_CLEAN_ASSET_SPECIFIC,
    KEY_CANDIDATE_V5_ASSET_SPECIFIC,
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
        self.assertEqual(profiles["moderate"].max_annualized_volatility, 0.15)
        self.assertEqual(profiles["aggressive"].max_average_turnover, 0.20)
        self.assertNotIn("max_weight_limit", profiles["moderate"].to_dict())

    def test_conservative_penalizes_risk_more_than_aggressive(self):
        strategies = pd.DataFrame(
            [
                {
                    "strategy_name": "risky",
                    "strategy_type": "td3",
                    "robust_score": 0.9,
                    "max_drawdown": -0.22,
                    "annualized_volatility": 0.22,
                    "average_max_weight": 1.0,
                    "average_effective_number_of_assets": 1.5,
                    "average_turnover": 0.20,
                }
            ]
        )

        scores = score_strategies_for_profiles(strategies)
        conservative = scores[scores["profile"] == "conservative"].iloc[0]
        aggressive = scores[scores["profile"] == "aggressive"].iloc[0]

        self.assertLess(conservative["profile_score"], aggressive["profile_score"])
        self.assertFalse(bool(conservative["profile_eligible"]))
        self.assertTrue(bool(aggressive["profile_eligible"]))
        self.assertNotIn("max_weight_pass", scores.columns)

    def test_ranking_changes_when_thresholds_differ(self):
        strategies = pd.DataFrame(
            [
                {
                    "strategy_name": "safe_lower_score",
                    "strategy_type": "td3",
                    "robust_score": 0.5,
                    "max_drawdown": -0.08,
                    "annualized_volatility": 0.09,
                    "average_max_weight": 0.95,
                    "average_effective_number_of_assets": 3.2,
                    "average_turnover": 0.03,
                },
                {
                    "strategy_name": "risky_higher_score",
                    "strategy_type": "benchmark",
                    "robust_score": 0.9,
                    "max_drawdown": -0.22,
                    "annualized_volatility": 0.22,
                    "average_max_weight": 1.00,
                    "average_effective_number_of_assets": 1.5,
                    "average_turnover": 0.20,
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
                    "annualized_volatility": 0.09,
                    "average_max_weight": 0.50,
                    "average_effective_number_of_assets": 3.0,
                    "average_turnover": 0.05,
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
            self.assertEqual(metadata["mandate_profile_source"], "src/risk/mandate_profiles.py")
            self.assertNotIn("max_weight_limit", metadata["profile_thresholds"]["conservative"])
            self.assertIn("not official", metadata["max_weight_mandate_note"])

    def test_asset_specific_combined_report_includes_v5_and_v3_clean_ranks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            td3_dir = root / "td3"
            combined_dir = root / "combined"
            benchmark_dir = root / "benchmarks"
            histories_dir = benchmark_dir / "histories"
            output_dir = root / "out"
            td3_dir.mkdir()
            combined_dir.mkdir()
            histories_dir.mkdir(parents=True)

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
                json.dumps({"cost_model": cost_model}),
                encoding="utf-8",
            )
            (combined_dir / "asset_specific_cost_benchmark_comparison_metadata.json").write_text(
                json.dumps(
                    {
                        "cost_model": cost_model,
                        "combined_score_scope": "test_scope",
                        "benchmark_dir": str(benchmark_dir),
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "date": ["2024-01-05", "2024-01-12"],
                    "strategy_return": [0.01, 0.0],
                    "transaction_cost_mode": ["asset_specific", "asset_specific"],
                }
            ).to_csv(histories_dir / "trend_spy_cash_12p_history.csv", index=False)
            pd.DataFrame(
                [
                    _asset_specific_strategy_row(
                        KEY_CANDIDATE_V5_ASSET_SPECIFIC,
                        "td3",
                        robust_score=0.80,
                        max_drawdown=-0.09,
                    ),
                    _asset_specific_strategy_row(
                        KEY_CANDIDATE_V3_CLEAN_ASSET_SPECIFIC,
                        "td3",
                        robust_score=0.72,
                        max_drawdown=-0.10,
                    ),
                    _asset_specific_strategy_row(
                        "trend_spy_cash_12p",
                        "benchmark",
                        robust_score=0.65,
                        max_drawdown=-0.08,
                    ),
                ]
            ).to_csv(combined_dir / "asset_specific_cost_combined_ranking.csv", index=False)

            result = build_mandate_profile_comparison_report(
                final_report_dir=str(td3_dir),
                combined_report_dir=str(combined_dir),
                benchmark_dir=str(benchmark_dir),
                output_dir=str(output_dir),
                asset_specific_only=True,
            )

            self.assertTrue((output_dir / "mandate_profile_strategy_scores.csv").exists())
            self.assertTrue((output_dir / "mandate_profile_winners.csv").exists())
            self.assertTrue((output_dir / "mandate_profile_summary.md").exists())
            self.assertEqual(
                set(result["winners"]["best_td3_candidate"].dropna()),
                {KEY_CANDIDATE_V5_ASSET_SPECIFIC},
            )
            self.assertTrue((result["winners"]["v5_asset_specific_rank"] == 1.0).all())
            self.assertIn("asset-specific-cost universe", result["summary"])
            self.assertIn("statistical proof", result["summary"])

            metadata = json.loads((output_dir / "mandate_profile_metadata.json").read_text())
            self.assertTrue(metadata["asset_specific_only"])
            self.assertEqual(metadata["source_metadata"]["input_mode"], "asset_specific_combined")

    def test_asset_specific_mode_rejects_scalar_benchmark_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            td3_dir = root / "td3"
            combined_dir = root / "combined"
            benchmark_dir = root / "benchmarks"
            histories_dir = benchmark_dir / "histories"
            output_dir = root / "out"
            td3_dir.mkdir()
            combined_dir.mkdir()
            histories_dir.mkdir(parents=True)

            cost_model = {"transaction_cost_mode": "asset_specific"}
            (td3_dir / "asset_specific_cost_metadata.json").write_text(
                json.dumps({"cost_model": cost_model}),
                encoding="utf-8",
            )
            (combined_dir / "asset_specific_cost_benchmark_comparison_metadata.json").write_text(
                json.dumps({"cost_model": cost_model, "benchmark_dir": str(benchmark_dir)}),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    _asset_specific_strategy_row(
                        KEY_CANDIDATE_V5_ASSET_SPECIFIC,
                        "td3",
                        robust_score=0.80,
                        max_drawdown=-0.09,
                    )
                ]
            ).to_csv(combined_dir / "asset_specific_cost_combined_ranking.csv", index=False)
            pd.DataFrame(
                {
                    "date": ["2024-01-05"],
                    "strategy_return": [0.01],
                    "transaction_cost_mode": ["scalar"],
                }
            ).to_csv(histories_dir / "trend_spy_cash_12p_history.csv", index=False)

            with self.assertRaisesRegex(ValueError, "asset-specific benchmark history"):
                build_mandate_profile_comparison_report(
                    final_report_dir=str(td3_dir),
                    combined_report_dir=str(combined_dir),
                    benchmark_dir=str(benchmark_dir),
                    output_dir=str(output_dir),
                    asset_specific_only=True,
                )


if __name__ == "__main__":
    unittest.main()


def _asset_specific_strategy_row(
    strategy_name: str,
    strategy_type: str,
    *,
    robust_score: float,
    max_drawdown: float,
) -> dict:
    return {
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "strategy_group": "td3_best_constrained" if strategy_type == "td3" else "benchmark",
        "robust_score": robust_score,
        "mandate_aware_score": robust_score,
        "annualized_return": 0.10,
        "annualized_volatility": 0.12,
        "sharpe": 0.8,
        "max_drawdown": max_drawdown,
        "average_turnover": 0.20,
        "average_effective_number_of_assets": 3.0 if strategy_type == "td3" else 2.0,
        "average_max_weight": 0.50,
        "transaction_cost_mode": "asset_specific",
    }
