"""Tests for mandate-aware scoring layer."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.mandate_aware_score import (
    add_mandate_aware_scores,
    assign_drawdown_bucket,
    calculate_recovery_required,
    get_drawdown_multiplier,
    write_mandate_aware_outputs,
)
from src.analysis.robust_score import DEFAULT_COMPOSITE_WEIGHTS


class MandateAwareScoreTests(unittest.TestCase):
    def test_exact_bucket_boundaries(self):
        cases = [
            (-0.01, "clean_mandate"),
            (-0.20, "clean_mandate"),
            (-0.2001, "eligible_yellow"),
            (-0.25, "eligible_yellow"),
            (-0.2501, "eligible_red"),
            (-0.30, "eligible_red"),
            (-0.3001, "not_eligible"),
        ]

        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(assign_drawdown_bucket(value), expected)

    def test_recovery_required_and_continuous_multipliers(self):
        cases = [
            (-0.01, 0.010101, 0.989899),
            (-0.20, 0.25, 0.75),
            (-0.2001, 0.2001 / 0.7999, 1.0 - (0.2001 / 0.7999)),
            (-0.25, 1.0 / 3.0, 2.0 / 3.0),
            (-0.30, 0.30 / 0.70, 1.0 - (0.30 / 0.70)),
        ]

        for value, expected_recovery, expected_multiplier in cases:
            with self.subTest(value=value):
                self.assertAlmostEqual(
                    calculate_recovery_required(value),
                    expected_recovery,
                    places=6,
                )
                self.assertAlmostEqual(
                    get_drawdown_multiplier(value),
                    expected_multiplier,
                    places=6,
                )

    def test_mandate_aware_score_calculation(self):
        scored = add_mandate_aware_scores(self._input_frame()).set_index("strategy_name")

        self.assertAlmostEqual(scored.loc["clean", "mandate_aware_score"], 0.60)
        self.assertAlmostEqual(
            scored.loc["yellow", "mandate_aware_score"],
            0.60 * (1.0 - (0.22 / 0.78)),
        )
        self.assertAlmostEqual(
            scored.loc["red", "mandate_aware_score"],
            0.40 * (1.0 - (0.27 / 0.73)),
        )
        self.assertAlmostEqual(scored.loc["ineligible", "mandate_aware_score"], 0.00)
        self.assertGreater(scored.loc["ineligible", "drawdown_multiplier"], 0.0)

    def test_ranks_are_computed(self):
        scored = add_mandate_aware_scores(self._input_frame())

        self.assertIn("performance_robust_rank", scored.columns)
        self.assertIn("mandate_aware_rank", scored.columns)
        self.assertIn("mandate_bucket_rank", scored.columns)
        clean = scored.set_index("strategy_name").loc["clean"]
        self.assertEqual(int(clean["mandate_aware_rank"]), 1)
        ineligible = scored.set_index("strategy_name").loc["ineligible"]
        self.assertEqual(int(ineligible["performance_robust_rank"]), 1)
        self.assertEqual(float(ineligible["mandate_aware_score"]), 0.0)

    def test_output_files_are_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "comparison"
            input_dir.mkdir()
            self._input_frame().to_csv(
                input_dir / "protocol_comparison_summary.csv",
                index=False,
            )

            report = write_mandate_aware_outputs(
                input_dir=str(input_dir),
                output_dir=str(Path(temp_dir) / "mandate"),
            )

            for path in report["paths"].values():
                self.assertTrue(Path(path).exists())
            self.assertIn("mandate_bucket", report["bucket_summary"].columns)
            self.assertIn("mean_recovery_required", report["bucket_summary"].columns)
            self.assertIn("mean_drawdown_multiplier", report["bucket_summary"].columns)
            self.assertIn("is_eligible", report["eligibility_flags"].columns)
            self.assertIn("recovery_required", report["eligibility_flags"].columns)

    def test_not_eligible_forces_score_to_zero_despite_positive_multiplier(self):
        frame = pd.DataFrame(
            [
                {
                    "strategy_name": "high_robust_not_eligible",
                    "strategy_type": "benchmark",
                    "robust_score": 0.95,
                    "max_drawdown": -0.3001,
                },
                {
                    "strategy_name": "lower_robust_clean",
                    "strategy_type": "td3",
                    "robust_score": 0.50,
                    "max_drawdown": -0.10,
                },
            ]
        )

        scored = add_mandate_aware_scores(frame).set_index("strategy_name")

        self.assertEqual(
            scored.loc["high_robust_not_eligible", "mandate_bucket"],
            "not_eligible",
        )
        self.assertGreater(
            scored.loc["high_robust_not_eligible", "drawdown_multiplier"],
            0.0,
        )
        self.assertEqual(
            scored.loc["high_robust_not_eligible", "mandate_aware_score"],
            0.0,
        )
        self.assertEqual(
            int(scored.loc["high_robust_not_eligible", "performance_robust_rank"]),
            1,
        )
        self.assertEqual(
            int(scored.loc["lower_robust_clean", "mandate_aware_rank"]),
            1,
        )

    def test_production_robust_score_is_not_modified(self):
        before = dict(DEFAULT_COMPOSITE_WEIGHTS)

        add_mandate_aware_scores(self._input_frame())

        self.assertEqual(DEFAULT_COMPOSITE_WEIGHTS, before)

    def _input_frame(self):
        return pd.DataFrame(
            [
                {
                    "strategy_name": "clean",
                    "strategy_type": "benchmark",
                    "robust_score": 0.80,
                    "max_drawdown": -0.20,
                    "sharpe": 1.0,
                    "sortino": 1.2,
                    "calmar": 1.0,
                    "average_turnover": 0.1,
                    "average_effective_number_of_assets": 3.0,
                    "dsr_method": "date_averaged",
                },
                {
                    "strategy_name": "yellow",
                    "strategy_type": "td3",
                    "robust_score": 0.60,
                    "max_drawdown": -0.22,
                    "sharpe": 0.8,
                    "sortino": 1.0,
                    "calmar": 0.8,
                    "average_turnover": 0.4,
                    "average_effective_number_of_assets": 1.5,
                    "dsr_method": "median_run",
                },
                {
                    "strategy_name": "red",
                    "strategy_type": "benchmark",
                    "robust_score": 0.40,
                    "max_drawdown": -0.27,
                    "sharpe": 0.7,
                    "sortino": 0.9,
                    "calmar": 0.6,
                    "average_turnover": 0.2,
                    "average_effective_number_of_assets": 2.0,
                    "dsr_method": "date_averaged",
                },
                {
                    "strategy_name": "ineligible",
                    "strategy_type": "benchmark",
                    "robust_score": 0.90,
                    "max_drawdown": -0.31,
                    "sharpe": 1.4,
                    "sortino": 2.0,
                    "calmar": 0.9,
                    "average_turnover": 0.6,
                    "average_effective_number_of_assets": 1.0,
                    "dsr_method": "date_averaged",
                },
            ]
        )


if __name__ == "__main__":
    unittest.main()
