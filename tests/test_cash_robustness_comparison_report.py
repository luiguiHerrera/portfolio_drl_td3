"""Tests for zero-CASH vs BIL-CASH robustness comparison reports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.cash_robustness_comparison_report import (
    build_cash_robustness_comparison_report,
)


class CashRobustnessComparisonReportTests(unittest.TestCase):
    def test_report_detects_winner_change_and_computes_deltas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zero_dir = root / "zero"
            bil_dir = root / "bil"
            output_dir = root / "out"
            self._write_experiment(
                zero_dir,
                rows=[
                    self._row("A_cap_0p50", "A", "0.50", 0.60, 0.70),
                    self._row("B_cap_0p50", "B", "0.50", 0.50, 0.80),
                ],
            )
            self._write_experiment(
                bil_dir,
                rows=[
                    self._row("A_cap_0p70", "A", "0.70", 0.65, 0.72),
                    self._row("B_cap_0p70", "B", "0.70", 0.80, 0.90),
                ],
            )

            report = build_cash_robustness_comparison_report(
                zero_dir=str(zero_dir),
                bil_dir=str(bil_dir),
                output_dir=str(output_dir),
            )

            winner = report["winner_summary"].iloc[0]
            self.assertEqual(winner["zero_cash_winner_by_mandate"], "A_cap_0p50")
            self.assertEqual(winner["bil_cash_winner_by_mandate"], "B_cap_0p70")
            self.assertTrue(bool(winner["winner_changed"]))
            comparison = report["candidate_comparison"].set_index("candidate")
            self.assertAlmostEqual(comparison.loc["B", "delta_mandate_aware"], 0.30)
            self.assertTrue(bool(comparison.loc["B", "cap_changed"]))

    def test_report_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zero_dir = root / "zero"
            bil_dir = root / "bil"
            output_dir = root / "out"
            rows = [self._row("A_cap_0p50", "A", "0.50", 0.60, 0.70)]
            self._write_experiment(zero_dir, rows=rows)
            self._write_experiment(bil_dir, rows=rows)

            build_cash_robustness_comparison_report(
                zero_dir=str(zero_dir),
                bil_dir=str(bil_dir),
                output_dir=str(output_dir),
            )

            expected = {
                "cash_robustness_candidate_comparison.csv",
                "cash_robustness_all_candidate_caps.csv",
                "cash_robustness_winner_summary.csv",
                "cash_robustness_summary.md",
                "cash_robustness_metadata.json",
            }
            self.assertEqual(
                expected,
                {path.name for path in output_dir.iterdir()},
            )

    def test_candidate_set_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zero_dir = root / "zero"
            bil_dir = root / "bil"
            self._write_experiment(
                zero_dir,
                rows=[self._row("A_cap_0p50", "A", "0.50", 0.60, 0.70)],
            )
            self._write_experiment(
                bil_dir,
                rows=[self._row("B_cap_0p50", "B", "0.50", 0.60, 0.70)],
            )

            with self.assertRaisesRegex(ValueError, "Candidate sets do not match"):
                build_cash_robustness_comparison_report(
                    zero_dir=str(zero_dir),
                    bil_dir=str(bil_dir),
                    output_dir=str(root / "out"),
                )

    def test_missing_summary_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zero_dir = root / "zero"
            bil_dir = root / "bil"
            self._write_experiment(
                zero_dir,
                rows=[self._row("A_cap_0p50", "A", "0.50", 0.60, 0.70)],
            )
            self._write_experiment(
                bil_dir,
                rows=[self._row("A_cap_0p50", "A", "0.50", 0.60, 0.70)],
            )
            (bil_dir / "cap_sensitivity_summary.csv").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "summary"):
                build_cash_robustness_comparison_report(
                    zero_dir=str(zero_dir),
                    bil_dir=str(bil_dir),
                    output_dir=str(root / "out"),
                )

    def _write_experiment(self, directory: Path, rows: list[dict]) -> None:
        directory.mkdir(parents=True)
        all_results = pd.DataFrame(rows)
        all_results.to_csv(directory / "cap_sensitivity_all_results.csv", index=False)
        summary_rows = []
        for base_candidate, group in all_results.groupby("base_candidate"):
            best = group.sort_values(
                ["mandate_aware_score", "robust_score"],
                ascending=[False, False],
            ).iloc[0]
            summary_rows.append(
                {
                    "base_candidate": base_candidate,
                    "best_cap_by_mandate_aware_score": best["cap_label"],
                    "best_cap_mandate_aware_score": best["mandate_aware_score"],
                    "best_cap_by_robust_score": best["cap_label"],
                    "best_cap_robust_score": best["robust_score"],
                }
            )
        pd.DataFrame(summary_rows).to_csv(
            directory / "cap_sensitivity_summary.csv",
            index=False,
        )

    @staticmethod
    def _row(
        candidate_name: str,
        base_candidate: str,
        cap_label: str,
        mandate: float,
        robust: float,
    ) -> dict:
        return {
            "candidate_name": candidate_name,
            "base_candidate": base_candidate,
            "max_weight_cap": float(cap_label),
            "cap_label": cap_label,
            "mandate_aware_score": mandate,
            "robust_score": robust,
            "max_drawdown": -0.10,
            "annualized_return": 0.08,
            "annualized_volatility": 0.12,
            "sharpe": 0.70,
            "average_turnover": 0.20,
            "average_effective_number_of_assets": 2.5,
            "mean_cash_weight": 0.15,
        }


if __name__ == "__main__":
    unittest.main()
