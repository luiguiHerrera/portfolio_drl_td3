"""Tests for reward incentive audit reporting."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.audit_reward_incentives import (
    compare_experiment_folders,
    compute_reward_incentive_flags,
    load_reward_incentive_metrics,
    write_reward_incentive_audit,
)
from src.rewards.reward import compute_risk_aware_reward


class RewardIncentiveAuditTests(unittest.TestCase):
    def test_flags_are_computed_correctly(self):
        frame = pd.DataFrame(
            [
                {
                    "strategy": "V6_financial_state",
                    "strategy_type": "td3",
                    "average_turnover": 0.10,
                    "average_effective_number_of_assets": 1.10,
                    "sharpe": 0.20,
                    "robust_score": 0.10,
                    "mandate_aware_score": 0.05,
                    "max_drawdown": -0.20,
                    "abs_validation_test_sharpe_gap": 0.10,
                },
                {
                    "strategy": "V5_no_volatility_block",
                    "strategy_type": "td3",
                    "average_turnover": 0.30,
                    "average_effective_number_of_assets": 1.30,
                    "sharpe": 0.90,
                    "robust_score": 0.65,
                    "mandate_aware_score": 0.45,
                    "max_drawdown": -0.18,
                    "abs_validation_test_sharpe_gap": 0.20,
                },
                {
                    "strategy": "high_turnover",
                    "strategy_type": "benchmark",
                    "average_turnover": 0.70,
                    "average_effective_number_of_assets": 2.00,
                    "sharpe": 0.80,
                    "robust_score": 0.50,
                    "max_drawdown": -0.18,
                    "abs_validation_test_sharpe_gap": 0.20,
                },
            ]
        )

        scored = compute_reward_incentive_flags(frame).set_index("strategy")

        self.assertTrue(scored.loc["V6_financial_state", "high_concentration_flag"])
        self.assertTrue(scored.loc["V6_financial_state", "extreme_concentration_flag"])
        self.assertTrue(
            scored.loc["V6_financial_state", "low_turnover_high_concentration_flag"]
        )
        self.assertTrue(
            scored.loc[
                "V6_financial_state",
                "suspicious_or_lazy_concentration_candidate",
            ]
        )
        self.assertFalse(
            scored.loc["V6_financial_state", "justified_concentration_candidate"]
        )
        self.assertTrue(
            scored.loc["V5_no_volatility_block", "justified_concentration_candidate"]
        )
        self.assertTrue(scored.loc["high_turnover", "high_turnover_flag"])

    def test_structural_single_asset_benchmark_is_not_flagged_lazy(self):
        frame = pd.DataFrame(
            [
                {
                    "strategy": "BuyHold_GLD",
                    "strategy_type": "benchmark",
                    "average_turnover": 0.0,
                    "average_effective_number_of_assets": 1.0,
                    "sharpe": 0.9,
                    "robust_score": 0.7,
                    "mandate_aware_score": 0.5,
                    "max_drawdown": -0.18,
                    "abs_validation_test_sharpe_gap": pd.NA,
                }
            ]
        )

        scored = compute_reward_incentive_flags(frame).iloc[0]

        self.assertEqual(scored["strategy_role"], "single_asset_benchmark")
        self.assertEqual(scored["concentration_origin"], "structural")
        self.assertEqual(
            scored["concentration_classification"],
            "structural_concentration_benchmark",
        )
        self.assertTrue(scored["structural_concentration_flag"])
        self.assertFalse(scored["suspicious_or_lazy_concentration_candidate"])
        self.assertIn("Structural single-asset benchmark", scored["concentration_reason"])

    def test_load_reward_incentive_metrics_merges_robust_and_mandate_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "experiment"
            mandate_dir = Path(temp_dir) / "mandate"
            experiment_dir.mkdir()
            mandate_dir.mkdir()
            self._write_experiment_outputs(experiment_dir, robust_score=0.55)
            pd.DataFrame(
                [
                    {
                        "strategy_name": "V2_reference_full",
                        "strategy_type": "td3",
                        "mandate_aware_score": 0.42,
                        "mandate_bucket": "eligible_yellow",
                        "drawdown_multiplier": 0.70,
                        "recovery_required": 0.30,
                    }
                ]
            ).to_csv(mandate_dir / "mandate_aware_ranking.csv", index=False)

            metrics = load_reward_incentive_metrics(
                str(experiment_dir),
                mandate_dir=str(mandate_dir),
                experiment_label="synthetic",
            )

            self.assertEqual(len(metrics), 1)
            row = metrics.iloc[0]
            self.assertEqual(row["strategy"], "V2_reference_full")
            self.assertAlmostEqual(row["robust_score"], 0.55)
            self.assertAlmostEqual(row["mandate_aware_score"], 0.42)
            self.assertAlmostEqual(row["validation_test_sharpe_gap"], -0.20)
            self.assertAlmostEqual(row["validation_test_return_gap"], -0.01)

    def test_compare_experiment_folders_computes_deltas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = Path(temp_dir) / "baseline"
            primary = Path(temp_dir) / "primary"
            baseline.mkdir()
            primary.mkdir()
            self._write_experiment_outputs(baseline, robust_score=0.30, turnover=0.20)
            self._write_experiment_outputs(primary, robust_score=0.45, turnover=0.35)

            comparison = compare_experiment_folders(
                str(baseline),
                str(primary),
                baseline_label="baseline",
                primary_label="primary",
            )

            self.assertEqual(len(comparison), 1)
            self.assertAlmostEqual(comparison["delta_robust_score"].iloc[0], 0.15)
            self.assertAlmostEqual(comparison["delta_average_turnover"].iloc[0], 0.15)
            self.assertIn("delta_cumulative_return", comparison.columns)
            self.assertIn("delta_annualized_return", comparison.columns)

    def test_write_reward_incentive_audit_creates_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "experiment"
            output_dir = Path(temp_dir) / "audit"
            experiment_dir.mkdir()
            self._write_experiment_outputs(experiment_dir, robust_score=0.55)

            report = write_reward_incentive_audit(
                primary_dir=str(experiment_dir),
                output_dir=str(output_dir),
                baseline_dir=None,
            )

            for path in report["paths"].values():
                self.assertTrue(Path(path).exists())
            self.assertIn("lazy_concentration_candidate", report["flags"].columns)
            self.assertIn("average_turnover", report["audit"].columns)
            self.assertIn("cumulative_return", report["audit"].columns)
            self.assertIn("concentration_reason", report["audit"].columns)
            self.assertIn(
                "n_learned_extreme_concentration_models",
                report["summary"].columns,
            )

    def test_production_reward_logic_is_not_modified(self):
        reward = compute_risk_aware_reward(
            portfolio_return=0.04,
            transaction_cost=0.01,
            turnover=0.3,
            weights=pd.Series([0.5, 0.5]).to_numpy(),
            portfolio_value=100000.0,
            peak_portfolio_value=100000.0,
            reward_config={},
        )

        self.assertAlmostEqual(reward, 0.03)

    def _write_experiment_outputs(
        self,
        experiment_dir: Path,
        robust_score: float,
        turnover: float = 0.25,
    ) -> None:
        pd.DataFrame(
            [
                {
                    "strategy": "V2_reference_full",
                    "split": "validation",
                    "strategy_type": "drl",
                    "n_folds": 1,
                    "n_seeds": 1,
                    "n_observations": 10,
                    "mean_sharpe": 0.70,
                    "std_sharpe": 0.0,
                    "mean_sortino": 1.0,
                    "mean_calmar": 0.8,
                    "mean_cumulative_return": 0.05,
                    "mean_annualized_return": 0.05,
                    "mean_annualized_volatility": 0.10,
                    "mean_max_drawdown": -0.18,
                    "mean_average_turnover": turnover,
                    "mean_average_effective_number_of_assets": 1.20,
                    "mean_average_max_weight": 0.95,
                    "mean_cash_weight": 0.05,
                    "cash_above_10_rate": 0.0,
                    "unjustified_cash_excess": 0.0,
                    "mean_cash_penalty": 0.0,
                    "mean_cash_breach": 0.0,
                    "mean_turnover_penalty": 0.001,
                    "mean_transaction_cost": 0.0002,
                    "worst_max_drawdown": -0.20,
                },
                {
                    "strategy": "V2_reference_full",
                    "split": "test",
                    "strategy_type": "drl",
                    "n_folds": 1,
                    "n_seeds": 1,
                    "n_observations": 10,
                    "mean_sharpe": 0.50,
                    "std_sharpe": 0.0,
                    "mean_sortino": 0.9,
                    "mean_calmar": 0.7,
                    "mean_cumulative_return": 0.04,
                    "mean_annualized_return": 0.04,
                    "mean_annualized_volatility": 0.10,
                    "mean_max_drawdown": -0.19,
                    "mean_average_turnover": turnover,
                    "mean_average_effective_number_of_assets": 1.20,
                    "mean_average_max_weight": 0.95,
                    "mean_cash_weight": 0.05,
                    "cash_above_10_rate": 0.0,
                    "unjustified_cash_excess": 0.0,
                    "mean_cash_penalty": 0.0,
                    "mean_cash_breach": 0.0,
                    "mean_turnover_penalty": 0.001,
                    "mean_transaction_cost": 0.0002,
                    "worst_max_drawdown": -0.20,
                },
            ]
        ).to_csv(experiment_dir / "overall_aggregate_by_strategy_split.csv", index=False)
        pd.DataFrame(
            [
                {
                    "strategy": "V2_reference_full",
                    "type": "drl",
                    "robust_score": robust_score,
                    "dsr_score": 0.1,
                    "median_run_dsr_n25": 0.1,
                    "date_averaged_dsr_n25": 0.2,
                    "dsr_method": "median_run",
                }
            ]
        ).to_csv(experiment_dir / "robust_score_ranking.csv", index=False)


if __name__ == "__main__":
    unittest.main()
