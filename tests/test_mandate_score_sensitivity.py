"""Tests for mandate score sensitivity reporting."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.mandate_score_sensitivity import (
    SCENARIOS,
    MandateScenario,
    assign_scenario_bucket,
    build_mandate_score_sensitivity,
    build_sensitivity_summary,
    calculate_recovery_required,
    score_scenario,
)


class MandateScoreSensitivityTests(unittest.TestCase):
    def test_strict_classifies_minus_016_as_eligible_yellow(self):
        strict = _scenario("strict")

        self.assertEqual(assign_scenario_bucket(-0.16, strict), "eligible_yellow")

    def test_base_classifies_minus_016_as_clean(self):
        base = _scenario("base")

        self.assertEqual(assign_scenario_bucket(-0.16, base), "clean_mandate")

    def test_flexible_classifies_minus_032_as_eligible_red(self):
        flexible = _scenario("flexible")

        self.assertEqual(assign_scenario_bucket(-0.32, flexible), "eligible_red")

    def test_not_eligible_gets_zero_score(self):
        scored = score_scenario(self._source_frame(), _scenario("base"))
        row = scored.set_index("strategy_name").loc["Momentum"]

        self.assertEqual(row["mandate_bucket"], "not_eligible")
        self.assertAlmostEqual(float(row["scenario_mandate_aware_score"]), 0.0)

    def test_recovery_formula_is_correct(self):
        self.assertAlmostEqual(calculate_recovery_required(-0.20), 0.25)
        self.assertAlmostEqual(calculate_recovery_required(-0.25), 1.0 / 3.0)

    def test_top_ranking_by_scenario_works(self):
        scored = score_scenario(self._source_frame(), _scenario("base"))
        top = scored.sort_values("scenario_rank").iloc[0]

        self.assertEqual(top["strategy_name"], "V5_cap_0.60")

    def test_summary_detects_when_best_td3_beats_best_benchmark(self):
        scored = score_scenario(self._source_frame(), _scenario("base"))
        summary = build_sensitivity_summary(scored)

        self.assertTrue(bool(summary.iloc[0]["td3_beats_best_benchmark"]))
        self.assertEqual(summary.iloc[0]["best_td3_strategy"], "V5_cap_0.60")

    def test_full_report_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            self._source_frame().to_csv(
                input_dir / "capped_td3_vs_benchmarks_summary.csv",
                index=False,
            )

            result = build_mandate_score_sensitivity(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
            )

            for path in result["paths"].values():
                self.assertTrue(Path(path).exists())

    def _source_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strategy_name": "V5_cap_0.60",
                    "strategy_type": "td3_capped",
                    "constraint_status": "cap_0.60",
                    "robust_score": 0.70,
                    "max_drawdown": -0.16,
                    "annualized_return": 0.08,
                    "sharpe": 0.70,
                    "average_turnover": 0.35,
                    "average_effective_number_of_assets": 2.4,
                    "average_max_weight": 0.60,
                },
                {
                    "strategy_name": "BuyHold_GLD",
                    "strategy_type": "benchmark",
                    "constraint_status": "benchmark",
                    "robust_score": 0.65,
                    "max_drawdown": -0.19,
                    "annualized_return": 0.10,
                    "sharpe": 0.80,
                    "average_turnover": 0.01,
                    "average_effective_number_of_assets": 1.0,
                    "average_max_weight": 1.0,
                },
                {
                    "strategy_name": "Momentum",
                    "strategy_type": "benchmark",
                    "constraint_status": "benchmark",
                    "robust_score": 0.90,
                    "max_drawdown": -0.45,
                    "annualized_return": 0.40,
                    "sharpe": 1.20,
                    "average_turnover": 0.40,
                    "average_effective_number_of_assets": 1.0,
                    "average_max_weight": 1.0,
                },
            ]
        )


def _scenario(name: str) -> MandateScenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise AssertionError(f"Missing scenario {name}")


if __name__ == "__main__":
    unittest.main()
