"""Tests for robust_score bias audit utilities."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.audit_robust_score_bias import (
    build_robust_score_bias_audit,
)
from src.analysis.robust_score import DEFAULT_COMPOSITE_WEIGHTS


class RobustScoreBiasAuditTests(unittest.TestCase):
    def test_audit_outputs_are_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir = self._write_protocol_metrics(temp_dir)
            output_dir = Path(temp_dir) / "audit"

            report = build_robust_score_bias_audit(
                comparison_dir=str(comparison_dir),
                output_dir=str(output_dir),
            )

            for path in report["paths"].values():
                self.assertTrue(Path(path).exists())

    def test_sensitivity_variants_are_computed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_robust_score_bias_audit(
                comparison_dir=str(self._write_protocol_metrics(temp_dir)),
                output_dir=str(Path(temp_dir) / "audit"),
            )

        variants = set(report["rank_sensitivity"]["variant"])
        self.assertIn("current", variants)
        self.assertIn("no_dsr", variants)
        self.assertIn("half_dsr", variants)
        self.assertIn("double_drawdown", variants)
        self.assertIn("mandate_style", variants)

    def test_bias_flags_work(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_robust_score_bias_audit(
                comparison_dir=str(self._write_protocol_metrics(temp_dir)),
                output_dir=str(Path(temp_dir) / "audit"),
            )
            flags = report["flags"].set_index("strategy_name")

        self.assertTrue(flags.loc["high_momentum", "high_drawdown_flag"])
        self.assertTrue(flags.loc["high_momentum", "high_turnover_flag"])
        self.assertTrue(flags.loc["high_momentum", "single_asset_strategy_flag"])
        self.assertTrue(flags.loc["td3_candidate", "dsr_method_median_run_flag"])
        self.assertTrue(flags.loc["balanced", "dsr_method_date_averaged_flag"])

    def test_rank_changes_are_computed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_robust_score_bias_audit(
                comparison_dir=str(self._write_protocol_metrics(temp_dir)),
                output_dir=str(Path(temp_dir) / "audit"),
            )
            sensitivity = report["rank_sensitivity"]

        self.assertIn("rank_change_vs_current", sensitivity.columns)
        self.assertFalse(sensitivity["rank_change_vs_current"].isna().any())

    def test_production_robust_score_weights_are_not_modified(self):
        before = dict(DEFAULT_COMPOSITE_WEIGHTS)
        with tempfile.TemporaryDirectory() as temp_dir:
            build_robust_score_bias_audit(
                comparison_dir=str(self._write_protocol_metrics(temp_dir)),
                output_dir=str(Path(temp_dir) / "audit"),
            )

        self.assertEqual(DEFAULT_COMPOSITE_WEIGHTS, before)

    def _write_protocol_metrics(self, temp_dir: str) -> Path:
        comparison_dir = Path(temp_dir) / "comparison"
        comparison_dir.mkdir()
        metrics = pd.DataFrame(
            [
                {
                    "strategy_name": "high_momentum",
                    "strategy_type": "benchmark",
                    "robust_score": 0.90,
                    "sharpe": 1.5,
                    "sortino": 2.0,
                    "calmar": 0.8,
                    "max_drawdown": -0.45,
                    "average_turnover": 0.80,
                    "average_max_weight": 1.0,
                    "average_effective_number_of_assets": 1.0,
                    "cash_above_10pct": 0.0,
                    "date_averaged_dsr_n25": 0.95,
                    "pooled_dsr_n25": 0.95,
                    "dsr_method": "date_averaged",
                },
                {
                    "strategy_name": "balanced",
                    "strategy_type": "benchmark",
                    "robust_score": 0.70,
                    "sharpe": 0.9,
                    "sortino": 1.5,
                    "calmar": 0.9,
                    "max_drawdown": -0.12,
                    "average_turnover": 0.10,
                    "average_max_weight": 0.35,
                    "average_effective_number_of_assets": 3.0,
                    "cash_above_10pct": 0.0,
                    "date_averaged_dsr_n25": 0.70,
                    "pooled_dsr_n25": 0.70,
                    "dsr_method": "date_averaged",
                },
                {
                    "strategy_name": "td3_candidate",
                    "strategy_type": "td3",
                    "robust_score": 0.40,
                    "sharpe": 0.5,
                    "sortino": 0.8,
                    "calmar": 0.6,
                    "max_drawdown": -0.20,
                    "average_turnover": 0.40,
                    "average_max_weight": 0.90,
                    "average_effective_number_of_assets": 1.2,
                    "cash_above_10pct": 0.05,
                    "median_run_dsr_n25": 0.20,
                    "pooled_dsr_n25": 0.80,
                    "dsr_method": "median_run",
                },
            ]
        )
        metrics.to_csv(comparison_dir / "protocol_comparison_metrics.csv", index=False)
        return comparison_dir


if __name__ == "__main__":
    unittest.main()
