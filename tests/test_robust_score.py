import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.robust_score import (
    build_robust_score_report,
    compute_annualized_sharpe,
    compute_composite_robust_score,
    compute_deflated_sharpe_ratio,
    compute_discipline_score,
    compute_probabilistic_sharpe_ratio,
    estimate_expected_max_sharpe,
    normalize_metric_series,
)


class RobustScoreTest(unittest.TestCase):
    def test_psr_returns_value_in_unit_interval(self):
        returns = pd.Series([0.01, -0.002, 0.006, 0.004, -0.001, 0.003])

        value = compute_probabilistic_sharpe_ratio(returns)

        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_psr_increases_for_better_return_series(self):
        weak = pd.Series([0.001, -0.002, 0.000, 0.001, -0.001, 0.000] * 4)
        strong = pd.Series([0.010, 0.004, 0.006, 0.008, 0.002, 0.005] * 4)

        self.assertGreater(
            compute_probabilistic_sharpe_ratio(strong),
            compute_probabilistic_sharpe_ratio(weak),
        )

    def test_psr_handles_zero_volatility_safely(self):
        returns = pd.Series([0.01] * 12)

        self.assertEqual(compute_probabilistic_sharpe_ratio(returns), 0.0)

    def test_dsr_decreases_as_trials_increase(self):
        returns = pd.Series([0.012, -0.002, 0.007, 0.004, 0.003, 0.006] * 12)

        few_trials = compute_deflated_sharpe_ratio(returns, n_trials=1)
        many_trials = compute_deflated_sharpe_ratio(returns, n_trials=20)

        self.assertLessEqual(many_trials, few_trials)

    def test_dsr_with_one_trial_is_close_to_psr(self):
        returns = pd.Series([0.012, -0.002, 0.007, 0.004, 0.003, 0.006] * 12)

        psr = compute_probabilistic_sharpe_ratio(returns)
        dsr = compute_deflated_sharpe_ratio(returns, n_trials=1)

        self.assertAlmostEqual(dsr, psr, places=12)

    def test_expected_max_sharpe_increases_with_trials(self):
        low = estimate_expected_max_sharpe(sharpe_std=0.5, n_trials=5)
        high = estimate_expected_max_sharpe(sharpe_std=0.5, n_trials=50)

        self.assertGreater(high, low)

    def test_dsr_penalizes_more_for_fifty_trials_than_ten(self):
        returns = pd.Series([0.012, -0.002, 0.007, 0.004, 0.003, 0.006] * 12)

        dsr_10 = compute_deflated_sharpe_ratio(returns, n_trials=10)
        dsr_50 = compute_deflated_sharpe_ratio(returns, n_trials=50)

        self.assertLessEqual(dsr_50, dsr_10)

    def test_invalid_n_trials_rejected(self):
        returns = pd.Series([0.01, -0.002, 0.006, 0.004])

        with self.assertRaises(ValueError):
            compute_deflated_sharpe_ratio(returns, n_trials=0)

    def test_compute_annualized_sharpe_returns_finite_value(self):
        returns = pd.Series([0.01, -0.002, 0.006, 0.004, -0.001, 0.003])

        value = compute_annualized_sharpe(returns)

        self.assertTrue(np.isfinite(value))

    def test_normalization_handles_equal_values(self):
        normalized = normalize_metric_series(pd.Series([3.0, 3.0, 3.0]))

        self.assertTrue((normalized == 0.5).all())

    def test_normalization_handles_lower_is_better(self):
        normalized = normalize_metric_series(
            pd.Series([0.1, 0.2, 0.3]),
            higher_is_better=False,
        )

        self.assertEqual(float(normalized.iloc[0]), 1.0)
        self.assertEqual(float(normalized.iloc[2]), 0.0)

    def test_discipline_score_penalizes_unjustified_cash(self):
        metrics = pd.DataFrame(
            {
                "unjustified_cash_excess": [0.0, 0.5],
                "turnover": [0.2, 0.2],
                "effective_assets": [2.0, 2.0],
            },
        )

        score = compute_discipline_score(metrics)

        self.assertGreater(score.iloc[0], score.iloc[1])

    def test_discipline_score_penalizes_excessive_turnover(self):
        metrics = pd.DataFrame(
            {
                "unjustified_cash_excess": [0.0, 0.0],
                "turnover": [0.2, 0.9],
                "effective_assets": [2.0, 2.0],
            },
        )

        score = compute_discipline_score(metrics)

        self.assertGreater(score.iloc[0], score.iloc[1])

    def test_composite_robust_score_returns_unit_interval(self):
        metrics = self._synthetic_metrics()

        scored = compute_composite_robust_score(metrics)

        self.assertTrue(((scored["robust_score"] >= 0.0) & (scored["robust_score"] <= 1.0)).all())

    def test_composite_robust_score_ranks_superior_strategy_higher(self):
        metrics = self._synthetic_metrics()

        scored = compute_composite_robust_score(metrics).set_index("strategy")

        self.assertGreater(
            scored.loc["superior", "robust_score"],
            scored.loc["inferior", "robust_score"],
        )

    def test_build_robust_score_report_creates_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_report_fixture(root)

            report = build_robust_score_report(str(root))

            self.assertTrue(Path(report["ranking_path"]).exists())
            self.assertTrue(Path(report["component_details_path"]).exists())
            self.assertTrue(Path(report["warnings_path"]).exists())
            self.assertIn("robust_score", report["ranking"].columns)
            self.assertIn("dsr_n10", report["ranking"].columns)
            self.assertIn("dsr_n25", report["ranking"].columns)
            self.assertIn("dsr_n50", report["ranking"].columns)
            self.assertIn("pooled_dsr_n25", report["ranking"].columns)
            self.assertIn("median_run_dsr_n25", report["ranking"].columns)
            self.assertIn("date_averaged_dsr_n25", report["ranking"].columns)
            self.assertIn("Pooled DSR can overstate evidence", report["warnings"])

    def test_report_pooled_dsr_can_differ_from_median_run_dsr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_overlapping_run_dsr_fixture(root)

            report = build_robust_score_report(str(root))

            row = report["ranking"].set_index("strategy").loc["OverlapStrategy"]
            self.assertNotAlmostEqual(
                row["pooled_dsr_n25"],
                row["median_run_dsr_n25"],
                places=6,
            )

    def test_composite_score_uses_median_run_dsr_when_available(self):
        metrics = pd.DataFrame(
            {
                "strategy": ["median_high", "pooled_high"],
                "type": ["drl", "drl"],
                "sharpe": [0.5, 0.5],
                "sortino": [1.0, 1.0],
                "calmar": [1.0, 1.0],
                "max_drawdown": [-0.1, -0.1],
                "worst_drawdown": [-0.2, -0.2],
                "std_sharpe": [0.2, 0.2],
                "turnover": [0.2, 0.2],
                "effective_assets": [1.5, 1.5],
                "unjustified_cash_excess": [0.0, 0.0],
                "median_run_dsr_n25": [0.8, 0.2],
                "pooled_dsr_n25": [0.1, 0.9],
            },
        )

        scored = compute_composite_robust_score(metrics).set_index("strategy")

        self.assertAlmostEqual(scored.loc["median_high", "dsr_score"], 0.8)
        self.assertAlmostEqual(scored.loc["pooled_high", "dsr_score"], 0.2)
        self.assertEqual(scored.loc["median_high", "dsr_method"], "median_run")
        self.assertGreater(
            scored.loc["median_high", "robust_score"],
            scored.loc["pooled_high", "robust_score"],
        )

    def test_dsr_score_fallback_logic_uses_date_then_pooled_then_sharpe(self):
        metrics = pd.DataFrame(
            {
                "strategy": ["date_available", "pooled_available", "sharpe_fallback"],
                "type": ["drl", "drl", "drl"],
                "sharpe": [0.1, 0.2, 0.9],
                "sortino": [1.0, 1.0, 1.0],
                "calmar": [1.0, 1.0, 1.0],
                "max_drawdown": [-0.1, -0.1, -0.1],
                "worst_drawdown": [-0.2, -0.2, -0.2],
                "std_sharpe": [0.2, 0.2, 0.2],
                "turnover": [0.2, 0.2, 0.2],
                "effective_assets": [1.5, 1.5, 1.5],
                "unjustified_cash_excess": [0.0, 0.0, 0.0],
                "date_averaged_dsr_n25": [0.7, np.nan, np.nan],
                "pooled_dsr_n25": [0.2, 0.6, np.nan],
            },
        )

        scored = compute_composite_robust_score(metrics).set_index("strategy")

        self.assertAlmostEqual(scored.loc["date_available", "dsr_score"], 0.7)
        self.assertEqual(scored.loc["date_available", "dsr_method"], "date_averaged")
        self.assertAlmostEqual(scored.loc["pooled_available", "dsr_score"], 0.6)
        self.assertEqual(scored.loc["pooled_available", "dsr_method"], "pooled")
        self.assertEqual(
            scored.loc["sharpe_fallback", "dsr_method"],
            "fallback_from_sharpe",
        )

    def _synthetic_metrics(self):
        return pd.DataFrame(
            {
                "strategy": ["superior", "inferior"],
                "type": ["benchmark", "benchmark"],
                "sharpe": [1.5, 0.1],
                "sortino": [2.0, 0.2],
                "calmar": [1.2, -0.1],
                "max_drawdown": [-0.05, -0.4],
                "worst_drawdown": [-0.08, -0.5],
                "std_sharpe": [0.2, 1.0],
                "turnover": [0.2, 0.8],
                "effective_assets": [2.0, 1.0],
                "unjustified_cash_excess": [0.0, 0.4],
            },
        )

    def _write_report_fixture(self, root: Path):
        aggregate = pd.DataFrame(
            {
                "strategy": ["StrategyA", "StrategyB"],
                "strategy_type": ["drl", "benchmark"],
                "split": ["test", "test"],
                "mean_sharpe": [1.0, 0.2],
                "std_sharpe": [0.1, 0.4],
                "mean_sortino": [1.5, 0.1],
                "mean_calmar": [1.0, -0.1],
                "mean_max_drawdown": [-0.1, -0.3],
                "worst_max_drawdown": [-0.2, -0.4],
                "mean_average_turnover": [0.2, 0.0],
                "mean_average_effective_number_of_assets": [1.5, np.nan],
                "cash_above_10_rate": [0.0, 0.0],
            },
        )
        aggregate.to_csv(root / "overall_aggregate_by_strategy_split.csv", index=False)

        run_dir = root / "F1_StrategyA_seed_1"
        run_dir.mkdir()
        pd.DataFrame(
            {
                "financial_net_return": [
                    0.01,
                    0.004,
                    -0.002,
                    0.006,
                    0.003,
                    0.005,
                ],
            },
        ).to_csv(run_dir / "test_policy_history.csv", index=False)

        pd.DataFrame(
            {
                "fold": ["F1"] * 6,
                "split": ["test"] * 6,
                "strategy": ["StrategyB"] * 6,
                "date": pd.date_range("2024-01-01", periods=6, freq="W"),
                "equity_curve": [1.00, 1.01, 1.00, 1.02, 1.01, 1.03],
            },
        ).to_csv(root / "benchmark_equity_curves_by_fold.csv", index=False)

    def _write_overlapping_run_dsr_fixture(self, root: Path):
        aggregate = pd.DataFrame(
            {
                "strategy": ["OverlapStrategy"],
                "strategy_type": ["drl"],
                "split": ["test"],
                "mean_sharpe": [0.5],
                "std_sharpe": [0.5],
                "mean_sortino": [1.0],
                "mean_calmar": [1.0],
                "mean_max_drawdown": [-0.1],
                "worst_max_drawdown": [-0.2],
                "mean_average_turnover": [0.2],
                "mean_average_effective_number_of_assets": [1.5],
                "cash_above_10_rate": [0.0],
            },
        )
        aggregate.to_csv(root / "overall_aggregate_by_strategy_split.csv", index=False)
        dates = pd.date_range("2024-01-05", periods=12, freq="W-FRI")
        run_returns = {
            1: [0.020, -0.004, 0.018, -0.003, 0.016, 0.002] * 2,
            2: [-0.018, 0.004, -0.016, 0.003, -0.014, -0.002] * 2,
        }
        for seed, returns in run_returns.items():
            run_dir = root / f"F1_OverlapStrategy_seed_{seed}"
            run_dir.mkdir()
            pd.DataFrame(
                {
                    "date": dates,
                    "financial_net_return": returns,
                },
            ).to_csv(run_dir / "test_policy_history.csv", index=False)


if __name__ == "__main__":
    unittest.main()
