import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.build_final_figures import (
    FIGURE_FILENAMES,
    build_final_figures,
)


class BuildFinalFiguresTest(unittest.TestCase):
    def test_builds_expected_figures_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_dir = root / "final"
            regime_dir = root / "regime"
            output_dir = root / "figures"
            final_dir.mkdir()
            regime_dir.mkdir()
            _write_final_report(final_dir)
            _write_regime_report(regime_dir)

            result = build_final_figures(
                final_report_dir=str(final_dir),
                regime_analysis_dir=str(regime_dir),
                output_dir=str(output_dir),
            )

            for filename in FIGURE_FILENAMES.values():
                figure_path = output_dir / filename
                self.assertTrue(figure_path.exists(), filename)
                self.assertGreater(figure_path.stat().st_size, 0)
            self.assertTrue((output_dir / "final_figures_summary.md").exists())
            metadata_path = output_dir / "final_figures_metadata.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(str(final_dir), metadata["final_report_dir"])
            self.assertEqual(str(regime_dir), metadata["regime_analysis_dir"])
            self.assertEqual(set(result["figure_paths"]), set(FIGURE_FILENAMES))

    def test_missing_optional_strategy_group_warns_but_still_creates_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_dir = root / "final"
            regime_dir = root / "regime"
            output_dir = root / "figures"
            final_dir.mkdir()
            regime_dir.mkdir()
            _write_final_report(final_dir, include_strategy_group=False)
            _write_regime_report(regime_dir)

            result = build_final_figures(
                final_report_dir=str(final_dir),
                regime_analysis_dir=str(regime_dir),
                output_dir=str(output_dir),
            )

            self.assertTrue(any("strategy_group missing" in warning for warning in result["warnings"]))
            self.assertTrue((output_dir / "mandate_score_vs_max_drawdown.png").exists())
            self.assertTrue((output_dir / "robust_score_vs_max_drawdown.png").exists())
            self.assertTrue((output_dir / "effective_assets_vs_mandate_score.png").exists())

    def test_heatmap_falls_back_to_sharpe_when_mandate_style_score_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_dir = root / "final"
            regime_dir = root / "regime"
            output_dir = root / "figures"
            final_dir.mkdir()
            regime_dir.mkdir()
            _write_final_report(final_dir)
            _write_regime_report(regime_dir, include_mandate_style_score=False)

            result = build_final_figures(
                final_report_dir=str(final_dir),
                regime_analysis_dir=str(regime_dir),
                output_dir=str(output_dir),
            )

            self.assertTrue(any("using Sharpe" in warning for warning in result["warnings"]))
            self.assertTrue((output_dir / "regime_mandate_heatmap.png").exists())


def _write_final_report(path: Path, include_strategy_group: bool = True) -> None:
    rows = [
        {
            "strategy_name": "V3_cap_0.60",
            "strategy_group": "td3_best_constrained",
            "strategy_type": "td3_capped",
            "max_drawdown": -0.14,
            "mandate_aware_score": 0.59,
            "robust_score": 0.70,
            "average_effective_number_of_assets": 2.5,
        },
        {
            "strategy_name": "V4_cap_0.50",
            "strategy_group": "td3_best_constrained",
            "strategy_type": "td3_capped",
            "max_drawdown": -0.16,
            "mandate_aware_score": 0.58,
            "robust_score": 0.72,
            "average_effective_number_of_assets": 3.1,
        },
        {
            "strategy_name": "BuyHold_GLD",
            "strategy_group": "benchmark_eligible",
            "strategy_type": "benchmark",
            "max_drawdown": -0.21,
            "mandate_aware_score": 0.52,
            "robust_score": 0.69,
            "average_effective_number_of_assets": 1.0,
        },
        {
            "strategy_name": "momentum_winner_12p",
            "strategy_group": "benchmark_not_eligible",
            "strategy_type": "benchmark",
            "max_drawdown": -0.51,
            "mandate_aware_score": 0.0,
            "robust_score": 0.86,
            "average_effective_number_of_assets": 1.0,
        },
    ]
    df = pd.DataFrame(rows)
    if not include_strategy_group:
        df = df.drop(columns=["strategy_group"])
    df.to_csv(path / "final_constrained_td3_main_ranking.csv", index=False)


def _write_regime_report(path: Path, include_mandate_style_score: bool = True) -> None:
    rows = []
    for regime_name in ["Calendar 2023", "Calendar 2024"]:
        for strategy_name, score, sharpe in [
            ("V3_cap_0.60", 0.50, 0.80),
            ("V4_cap_0.50", 0.55, 0.90),
            ("BuyHold_GLD", 0.40, 0.60),
            ("trend_spy_cash_12p", 0.35, 0.50),
        ]:
            row = {
                "regime_name": regime_name,
                "strategy_name": strategy_name,
                "sharpe": sharpe,
            }
            if include_mandate_style_score:
                row["mandate_style_score"] = score
            rows.append(row)
    pd.DataFrame(rows).to_csv(path / "regime_strategy_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "regime_name": "Calendar 2023",
                "best_by_mandate_style_score": "V4_cap_0.50",
            },
            {
                "regime_name": "Calendar 2024",
                "best_by_mandate_style_score": "V3_cap_0.60",
            },
        ]
    ).to_csv(path / "regime_winners_summary.csv", index=False)


if __name__ == "__main__":
    unittest.main()
