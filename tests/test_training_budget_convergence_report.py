import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.training_budget_convergence_report import (
    build_training_budget_convergence_report,
    decide_overall_conclusion,
)


class TrainingBudgetConvergenceReportTest(unittest.TestCase):
    def test_report_builds_outputs_and_flags_material_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root, "zero_cash", "V3", "0p70", 30, sharpe=0.50, drawdown=-0.10, turnover=0.10)
            _write_case(root, "zero_cash", "V3", "0p70", 60, sharpe=0.60, drawdown=-0.10, turnover=0.10)
            _write_case(root, "zero_cash", "V3", "0p70", 100, sharpe=0.75, drawdown=-0.11, turnover=0.10)
            _write_case(root, "zero_cash", "V3", "0p70", 150, sharpe=0.61, drawdown=-0.16, turnover=0.10)

            result = build_training_budget_convergence_report(output_dir=str(root))

            for path in result["paths"].values():
                self.assertTrue(Path(path).exists())
            summary = result["summary"]
            ep100 = summary[summary["episodes"] == 100].iloc[0]
            ep150 = summary[summary["episodes"] == 150].iloc[0]
            self.assertTrue(bool(ep100["material_sharpe_change"]))
            self.assertTrue(bool(ep150["material_drawdown_change"]))
            self.assertIn("potentially_undertrained", result["by_candidate"].iloc[0]["conclusion"])

    def test_turnover_relative_change_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root, "bil_cash", "V7", "0p80", 30, sharpe=0.60, drawdown=-0.10, turnover=0.10)
            _write_case(root, "bil_cash", "V7", "0p80", 60, sharpe=0.60, drawdown=-0.10, turnover=0.10)
            _write_case(root, "bil_cash", "V7", "0p80", 100, sharpe=0.60, drawdown=-0.10, turnover=0.14)
            _write_case(root, "bil_cash", "V7", "0p80", 150, sharpe=0.60, drawdown=-0.10, turnover=0.10)

            result = build_training_budget_convergence_report(output_dir=str(root))
            ep100 = result["summary"][result["summary"]["episodes"] == 100].iloc[0]
            self.assertTrue(bool(ep100["material_turnover_change"]))

    def test_overall_conclusion_for_all_adequate(self):
        frame = pd.DataFrame({"conclusion": ["60_episodes_appears_adequate", "60_episodes_appears_adequate"]})
        self.assertEqual(
            decide_overall_conclusion(frame),
            "60 episodes appears adequate for the corrected limited protocol.",
        )

    def test_longer_training_degradation_is_not_undertraining_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root, "bil_cash", "V8", "0p70", 30, sharpe=0.69, drawdown=-0.10, turnover=0.10)
            _write_case(root, "bil_cash", "V8", "0p70", 60, sharpe=0.70, drawdown=-0.10, turnover=0.10)
            _write_case(root, "bil_cash", "V8", "0p70", 100, sharpe=0.55, drawdown=-0.11, turnover=0.14)
            _write_case(root, "bil_cash", "V8", "0p70", 150, sharpe=0.50, drawdown=-0.10, turnover=0.13)

            result = build_training_budget_convergence_report(output_dir=str(root))
            row = result["by_candidate"].iloc[0]

            self.assertEqual(row["conclusion"], "longer_training_degrades_or_destabilizes")
            self.assertFalse(bool(row["sixty_episode_undertraining_evidence"]))
            self.assertTrue(bool(row["longer_training_degrades_sharpe_materially"]))
            self.assertIn("does not support rerunning", result["markdown"])

    def test_observed_pattern_concludes_sixty_not_obviously_undertrained(self):
        frame = pd.DataFrame(
            {
                "conclusion": [
                    "60_episodes_appears_adequate",
                    "60_episodes_appears_adequate",
                    "60_episodes_appears_adequate",
                    "budget_sensitive_but_not_undertrained",
                ],
                "sixty_episode_undertraining_evidence": [False, False, False, False],
            }
        )

        conclusion = decide_overall_conclusion(frame)

        self.assertIn("does not support rerunning", conclusion)
        self.assertIn("not obviously undertrained", conclusion)
        self.assertIn("optional publication-grade confirmation", conclusion)

    def test_material_longer_sharpe_improvement_is_undertraining_evidence(self):
        frame = pd.DataFrame(
            {
                "conclusion": ["60_episodes_potentially_undertrained"],
                "sixty_episode_undertraining_evidence": [True],
            }
        )

        self.assertIn("material Sharpe improvement", decide_overall_conclusion(frame))


def _write_case(
    root: Path,
    cash: str,
    candidate: str,
    cap: str,
    episodes: int,
    sharpe: float,
    drawdown: float,
    turnover: float,
) -> None:
    case_dir = root / "cases" / f"{cash}_{candidate}_cap_{cap}" / f"episodes_{episodes}"
    case_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "cash_assumption": cash,
                "candidate_name": f"{candidate}_cap_{cap}",
                "base_candidate": candidate,
                "cap_label": cap,
                "split": "test",
                "episodes": episodes,
                "annualized_return": 0.10,
                "annualized_volatility": 0.12,
                "sharpe": sharpe,
                "max_drawdown": drawdown,
                "average_turnover": turnover,
                "average_effective_number_of_assets": 3.0,
                "average_max_weight": 0.5,
                "mean_cash_weight": 0.1,
                "mean_btc_weight": 0.05,
                "robust_score": 0.7,
                "mandate_aware_score": 0.6,
                "completed_test_histories": 20,
            }
        ]
    ).to_csv(case_dir / "training_budget_case_summary.csv", index=False)
    pd.DataFrame(
        {
            "split": ["test", "test"],
            "mean_sharpe": [sharpe - 0.01, sharpe + 0.01],
            "mean_max_drawdown": [drawdown, drawdown - 0.01],
            "mean_average_turnover": [turnover, turnover + 0.01],
        }
    ).to_csv(case_dir / "seed_level_aggregate_by_strategy_split.csv", index=False)


if __name__ == "__main__":
    unittest.main()
