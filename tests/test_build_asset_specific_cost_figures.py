import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.build_asset_specific_cost_figures import (
    FIGURE_FILENAMES,
    build_asset_specific_cost_figures,
)


class BuildAssetSpecificCostFiguresTest(unittest.TestCase):
    def test_builds_expected_figures_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td3_dir = root / "td3"
            benchmark_dir = root / "combined"
            stat_dir = root / "stat"
            wrc_dir = root / "wrc"
            regime_dir = root / "regime"
            mandate_dir = root / "mandate"
            output_dir = root / "figures"
            for path in [td3_dir, benchmark_dir, stat_dir, wrc_dir, regime_dir, mandate_dir]:
                path.mkdir()
            _write_td3_selected(td3_dir)
            _write_combined_ranking(benchmark_dir)
            _write_statistical_validation(stat_dir)
            _write_wrc(wrc_dir)
            _write_regime(regime_dir)
            _write_mandate_winners(mandate_dir)

            result = build_asset_specific_cost_figures(
                td3_report_dir=str(td3_dir),
                benchmark_comparison_dir=str(benchmark_dir),
                statistical_validation_dir=str(stat_dir),
                white_reality_check_dir=str(wrc_dir),
                regime_analysis_dir=str(regime_dir),
                mandate_profile_dir=str(mandate_dir),
                output_dir=str(output_dir),
            )

            for filename in FIGURE_FILENAMES.values():
                path = output_dir / filename
                self.assertTrue(path.exists(), filename)
                self.assertGreater(path.stat().st_size, 0)
            metadata_path = output_dir / "asset_specific_cost_final_figures_metadata.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(str(td3_dir), metadata["td3_report_dir"])
            self.assertEqual(set(result["figure_paths"]), set(FIGURE_FILENAMES))
            self.assertIn("reporting-only", result["summary"])

    def test_missing_inputs_create_placeholders_and_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "figures"
            result = build_asset_specific_cost_figures(
                td3_report_dir=str(root / "missing_td3"),
                benchmark_comparison_dir=str(root / "missing_combined"),
                statistical_validation_dir=str(root / "missing_stat"),
                white_reality_check_dir=str(root / "missing_wrc"),
                regime_analysis_dir=str(root / "missing_regime"),
                mandate_profile_dir=str(root / "missing_mandate"),
                output_dir=str(output_dir),
            )

            self.assertTrue(result["warnings"])
            for filename in FIGURE_FILENAMES.values():
                self.assertTrue((output_dir / filename).exists(), filename)


def _write_td3_selected(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "candidate_name": "V5_no_volatility_block_cap_0p50",
                "mandate_aware_score": 0.69,
            },
            {
                "candidate_name": "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
                "mandate_aware_score": 0.66,
            },
            {
                "candidate_name": "V4_real_garch_current_cap_0p50",
                "mandate_aware_score": 0.65,
            },
        ]
    ).to_csv(path / "asset_specific_cost_selected_candidates.csv", index=False)


def _write_combined_ranking(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "strategy_name": "V5_no_volatility_block_cap_0p50",
                "strategy_type": "td3",
                "max_drawdown": -0.11,
                "mandate_aware_score": 0.69,
            },
            {
                "strategy_name": "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
                "strategy_type": "td3",
                "max_drawdown": -0.08,
                "mandate_aware_score": 0.66,
            },
            {
                "strategy_name": "trend_spy_cash_12p",
                "strategy_type": "benchmark",
                "max_drawdown": -0.18,
                "mandate_aware_score": 0.55,
            },
            {
                "strategy_name": "BuyHold_GLD",
                "strategy_type": "benchmark",
                "max_drawdown": -0.20,
                "mandate_aware_score": 0.50,
            },
        ]
    ).to_csv(path / "asset_specific_cost_combined_ranking.csv", index=False)


def _write_statistical_validation(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "candidate": "V5_no_volatility_block_cap_0p50",
                "benchmark": "trend_spy_cash_12p",
                "metric": "sharpe",
                "probability_candidate_beats": 0.60,
            },
            {
                "candidate": "V5_no_volatility_block_cap_0p50",
                "benchmark": "BuyHold_GLD",
                "metric": "sharpe",
                "probability_candidate_beats": 0.40,
            },
        ]
    ).to_csv(path / "statistical_validation_pairwise_bootstrap.csv", index=False)


def _write_wrc(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "benchmark": "trend_spy_cash_12p",
                "p_value": 0.52,
                "best_candidate_by_mean_diff": "V5_no_volatility_block_cap_0p50",
            },
            {
                "benchmark": "BuyHold_GLD",
                "p_value": 0.97,
                "best_candidate_by_mean_diff": "V5_no_volatility_block_cap_0p50",
            },
        ]
    ).to_csv(path / "white_reality_check_summary.csv", index=False)


def _write_regime(path: Path) -> None:
    rows = []
    for regime in ["Calendar 2023", "Calendar 2024"]:
        for strategy, score in [
            ("V5_no_volatility_block_cap_0p50", 0.7),
            ("V4_real_garch_current_cap_0p50", 0.6),
            ("V3_real_macro_vintage_clean_no_dxy_cap_0p70", 0.5),
        ]:
            rows.append(
                {
                    "regime_name": regime,
                    "strategy_name": strategy,
                    "mandate_style_score": score,
                }
            )
    pd.DataFrame(rows).to_csv(path / "regime_strategy_metrics.csv", index=False)


def _write_mandate_winners(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "profile": "conservative",
                "overall_winner": "V4_real_garch_current_cap_0p50",
                "overall_winner_type": "td3",
                "overall_winner_score": 0.72,
            },
            {
                "profile": "moderate",
                "overall_winner": "V5_no_volatility_block_cap_0p50",
                "overall_winner_type": "td3",
                "overall_winner_score": 0.79,
            },
        ]
    ).to_csv(path / "mandate_profile_winners.csv", index=False)


if __name__ == "__main__":
    unittest.main()
