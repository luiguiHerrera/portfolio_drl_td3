from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.paper_aligned_comparison import BASE_METRIC_COLUMNS
from src.analysis.paper_seed_aggregated_comparison import (
    CANONICAL_SHARPE_NAME,
    LEGACY_RATIO_NAME,
    PRIMARY_AGGREGATION_METHOD,
    WRC_STATISTIC_NAME,
    aggregate_seed_metrics,
    validate_seed_aggregated_output,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/paper_seed_aggregated_comparison"


class PaperSeedAggregatedComparisonTests(unittest.TestCase):
    def test_metrics_are_aggregated_after_per_seed_calculation(self) -> None:
        rows = []
        for seed, sharpe in [(7, 0.2), (21, 1.0)]:
            row = {metric: float(index + seed / 1000.0) for index, metric in enumerate(BASE_METRIC_COLUMNS)}
            row.update({"seed": seed, "sharpe": sharpe, "max_drawdown": -0.10 - seed / 1000.0, "dsr_score": 0.4 + seed / 1000.0})
            rows.append(row)
        seed_metrics = pd.DataFrame(rows)
        average_path = {metric: 99.0 for metric in BASE_METRIC_COLUMNS}
        average_path.update(
            {
                "protocol": "zero_cash",
                "strategy_name": "candidate",
                "strategy_type": "TD3",
                "date_averaged_dsr_n25": 0.9,
            }
        )
        aggregated = aggregate_seed_metrics(seed_metrics, average_path)
        self.assertAlmostEqual(aggregated["sharpe"], 0.6)
        self.assertNotEqual(aggregated["sharpe"], average_path["sharpe"])
        self.assertAlmostEqual(
            aggregated["annualized_volatility"],
            seed_metrics["annualized_volatility"].mean(),
        )
        self.assertEqual(aggregated["n_aligned_seeds"], 2)

    def test_generated_metadata_declares_expected_seed_estimand(self) -> None:
        methodology = json.loads((OUTPUT / "metadata/methodology.json").read_text(encoding="utf-8"))
        self.assertEqual(methodology["primary_td3_aggregation"], PRIMARY_AGGREGATION_METHOD)
        self.assertEqual(
            methodology["primary_estimand"],
            "expected_seed_performance_of_the_training_algorithm",
        )
        self.assertEqual(methodology["average_return_path"]["role"], "synthetic diagnostic only")

    def test_canonical_sharpe_legacy_ratio_and_wrc_are_distinct(self) -> None:
        reference = pd.read_csv(OUTPUT / "ranking/named_statistical_reference.csv")
        self.assertEqual(set(reference["canonical_ranking_statistic_name"]), {CANONICAL_SHARPE_NAME})
        self.assertEqual(set(reference["pairwise_statistic_name"]), {LEGACY_RATIO_NAME})
        self.assertEqual(set(reference["wrc_statistic_name"]), {WRC_STATISTIC_NAME})
        self.assertFalse(reference["numerically_comparable_to_canonical_sharpe"].astype(bool).any())
        self.assertFalse(reference["wrc_uses_cagr_to_volatility_ratio"].astype(bool).any())

    def test_benchmark_seed_dispersion_is_unavailable_not_zero(self) -> None:
        metrics = pd.read_csv(OUTPUT / "metrics/seed_aggregated_strategy_metrics.csv")
        benchmarks = metrics[metrics["strategy_type"].str.lower() == "benchmark"]
        self.assertTrue(benchmarks["std_sharpe"].isna().all())
        self.assertTrue(
            benchmarks["seed_dispersion_status"]
            .eq("not_observed_neutral_not_zero_in_stability_component")
            .all()
        )

    def test_paper_consumes_seed_aggregated_artifacts(self) -> None:
        paper = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
        self.assertIn("outputs/paper_seed_aggregated_comparison/paper/seed_results_macros.tex", paper)
        self.assertIn("outputs/paper_seed_aggregated_comparison/paper/seed_combined_table_rows.tex", paper)
        self.assertIn("outputs/paper_seed_aggregated_comparison/paper/named_statistical_table_rows.tex", paper)
        self.assertNotIn("outputs/paper_aligned_comparison/paper/aligned_combined_table_rows.tex", paper)

    def test_generated_package_validates(self) -> None:
        validation = validate_seed_aggregated_output(OUTPUT)
        self.assertEqual(validation["status"], "PASS")
        self.assertEqual(validation["metrics_computed_before_seed_aggregation"], "PASS")
        self.assertEqual(validation["canonical_sharpe_and_legacy_ratio_distinct"], "PASS")


if __name__ == "__main__":
    unittest.main()
