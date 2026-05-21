"""Tests for concentration penalty experiment runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.experiments.run_concentration_penalty_experiment import (
    build_candidate_run_config_with_concentration,
    build_concentration_grid_candidates,
    build_concentration_penalty_rankings,
    concentration_candidate_name,
    label_concentration_decision,
    parse_lambda_grid,
    run_concentration_penalty_experiment,
)


class ConcentrationPenaltyExperimentTests(unittest.TestCase):
    def test_parse_lambda_grid(self):
        self.assertEqual(parse_lambda_grid("0,0.01,0.03"), [0.0, 0.01, 0.03])

    def test_creates_isolated_candidate_names(self):
        self.assertEqual(
            concentration_candidate_name("V5_no_volatility_block", 0.01),
            "V5_no_volatility_block_lconc_0p01",
        )

    def test_grid_candidates_preserve_base_candidate(self):
        base = {
            "name": "V5_no_volatility_block",
            "feature_version": "v5",
            "exclude_blocks": ["volatility"],
            "use_dynamic_cash": True,
            "cash_risk_off_column": "risk_off_state",
        }

        candidates = build_concentration_grid_candidates(base, [0.0, 0.03])

        self.assertEqual(candidates[0]["name"], "V5_no_volatility_block_lconc_0p00")
        self.assertEqual(candidates[1]["lambda_concentration"], 0.03)
        self.assertEqual(base["name"], "V5_no_volatility_block")

    def test_overrides_only_lambda_concentration_without_mutating_base_config(self):
        base_config = self._base_config()
        candidate = {
            "name": "V5_no_volatility_block_lconc_0p03",
            "feature_version": "v5",
            "exclude_blocks": ["volatility"],
            "use_dynamic_cash": True,
            "cash_risk_off_column": "risk_off_state",
            "lambda_concentration": 0.03,
        }

        config = build_candidate_run_config_with_concentration(
            base_config=base_config,
            candidate=candidate,
            seed=7,
            episodes=5,
            batch_size=32,
            actor_learning_rate=0.0005,
            critic_learning_rate=0.0005,
        )

        self.assertEqual(config["reward"]["lambda_concentration"], 0.03)
        self.assertEqual(base_config["reward"]["lambda_concentration"], 0.0)
        self.assertEqual(config["reward"]["lambda_transaction_cost"], 1.0)

    def test_decision_labels_work_on_synthetic_rows(self):
        self.assertEqual(
            label_concentration_decision(
                pd.Series(
                    {
                        "lambda_concentration": 0.01,
                        "delta_average_effective_number_of_assets_vs_baseline": 0.25,
                        "delta_robust_score_vs_baseline": 0.0,
                        "delta_mandate_aware_score_vs_baseline": 0.0,
                    }
                )
            ),
            "dominates_baseline",
        )
        self.assertEqual(
            label_concentration_decision(
                pd.Series(
                    {
                        "lambda_concentration": 0.01,
                        "delta_average_effective_number_of_assets_vs_baseline": 0.25,
                        "delta_robust_score_vs_baseline": -0.10,
                        "delta_mandate_aware_score_vs_baseline": 0.0,
                    }
                )
            ),
            "diversifies_but_hurts_performance",
        )
        self.assertEqual(
            label_concentration_decision(
                pd.Series(
                    {
                        "lambda_concentration": 0.01,
                        "delta_average_effective_number_of_assets_vs_baseline": 0.05,
                        "delta_robust_score_vs_baseline": 0.10,
                    }
                )
            ),
            "no_behavioral_improvement",
        )

    def test_baseline_deltas_are_computed(self):
        summary = pd.DataFrame(
            [
                self._summary_row("base", 0.0, 1.1, 0.50, 0.40),
                self._summary_row("soft", 0.01, 1.4, 0.48, 0.39),
            ]
        )

        rankings = build_concentration_penalty_rankings(summary)
        soft = rankings.set_index("candidate_name").loc["soft"]

        self.assertAlmostEqual(
            soft["delta_average_effective_number_of_assets_vs_baseline"],
            0.30,
        )
        self.assertAlmostEqual(soft["delta_robust_score_vs_baseline"], -0.02)
        self.assertEqual(soft["decision_label"], "dominates_baseline")

    def test_smoke_mode_creates_expected_files_with_mocked_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "concentration_smoke"
            with (
                patch(
                    "src.experiments.run_concentration_penalty_experiment._build_base_config",
                    return_value=self._base_config(),
                ),
                patch(
                    "src.experiments.run_concentration_penalty_experiment.build_returns_dataset_from_config",
                    return_value=self._returns(),
                ),
                patch(
                    "src.experiments.run_concentration_penalty_experiment._build_feature_context",
                    return_value={},
                ),
                patch(
                    "src.experiments.run_concentration_penalty_experiment._candidate_raw_features",
                    return_value=self._returns(),
                ),
                patch(
                    "src.experiments.run_concentration_penalty_experiment._candidate_auxiliary_features",
                    return_value=pd.DataFrame({"risk_off_state": [0, 0, 0]}, index=self._returns().index),
                ),
                patch(
                    "src.experiments.run_concentration_penalty_experiment.build_ablation_fold_datasets",
                    return_value={
                        "train_returns": self._returns(),
                        "validation_returns": self._returns(),
                        "test_returns": self._returns(),
                    },
                ),
                patch(
                    "src.experiments.run_concentration_penalty_experiment.train_td3_ablation_on_datasets",
                    return_value={},
                ),
                patch(
                    "src.experiments.run_concentration_penalty_experiment._build_experiment_result",
                    return_value=self._experiment_result(),
                ),
                patch(
                    "src.experiments.run_concentration_penalty_experiment.save_basic_experiment_outputs",
                    return_value={},
                ),
                patch(
                    "src.experiments.run_concentration_penalty_experiment.build_robust_score_report",
                    return_value={
                        "ranking": pd.DataFrame(
                            [
                                {
                                    "strategy": "V5_no_volatility_block_lconc_0p00",
                                    "robust_score": 0.4,
                                    "median_run_dsr_n25": 0.1,
                                    "date_averaged_dsr_n25": 0.1,
                                    "dsr_method": "median_run",
                                },
                                {
                                    "strategy": "V5_no_volatility_block_lconc_0p01",
                                    "robust_score": 0.5,
                                    "median_run_dsr_n25": 0.1,
                                    "date_averaged_dsr_n25": 0.1,
                                    "dsr_method": "median_run",
                                },
                            ]
                        )
                    },
                ),
            ):
                report = run_concentration_penalty_experiment(
                    output_dir=str(output_dir),
                    lambda_concentration_grid=[0.0, 0.01],
                    episodes=5,
                    seeds=[7],
                    max_folds=1,
                    smoke=True,
                )

            for path in report["paths"].values():
                self.assertTrue(Path(path).exists())
            self.assertTrue(report["metadata"]["smoke_mode"])
            self.assertIn("experiment-only", report["metadata"]["experiment_only_warning"])

    def _base_config(self):
        return {
            "environment": {"transaction_cost": 0.001},
            "reward": {
                "lambda_return": 1.0,
                "lambda_transaction_cost": 1.0,
                "lambda_turnover": 0.0005,
                "lambda_concentration": 0.0,
                "lambda_drawdown": 0.02,
            },
            "training": {"seed": 7, "episodes": 5},
            "td3": {
                "batch_size": 32,
                "actor_learning_rate": 0.0005,
                "critic_learning_rate": 0.0005,
            },
            "features": {"version": "v5"},
        }

    def _returns(self):
        return pd.DataFrame(
            {
                "SPY": [0.01, 0.02, -0.01],
                "TLT": [0.0, 0.01, 0.0],
                "GLD": [0.01, 0.0, 0.01],
                "BTC-USD": [0.02, -0.01, 0.03],
                "CASH": [0.0, 0.0, 0.0],
            },
            index=pd.date_range("2022-01-07", periods=3, freq="W-FRI"),
        )

    def _experiment_result(self):
        metrics = pd.DataFrame(
            {
                "sharpe_ratio": [0.5],
                "sortino_ratio": [0.8],
                "calmar_ratio": [0.7],
                "cumulative_return": [0.03],
                "annualized_return": [0.05],
                "annualized_volatility": [0.10],
                "max_drawdown": [-0.20],
            },
            index=["agent"],
        )
        diagnostics = {
            "average_turnover": 0.3,
            "average_effective_number_of_assets": 1.1,
            "average_max_weight": 0.9,
        }
        policy_history = pd.DataFrame(
            {
                "cash_weight": [0.0, 0.0],
                "cash_risk_off_state": [0, 0],
                "cash_penalty": [0.0, 0.0],
                "cash_breach": [0.0, 0.0],
                "turnover_penalty": [0.0, 0.0],
                "transaction_cost": [0.0003, 0.0003],
            }
        )
        return {
            "validation_metrics_table": metrics,
            "test_metrics_table": metrics,
            "validation_diagnostics": diagnostics,
            "test_diagnostics": diagnostics,
            "validation_policy_history": policy_history,
            "test_policy_history": policy_history,
        }

    def _summary_row(
        self,
        name: str,
        lambda_concentration: float,
        effective_assets: float,
        robust_score: float,
        mandate_aware_score: float,
    ):
        return {
            "candidate_name": name,
            "lambda_concentration": lambda_concentration,
            "split": "test",
            "average_effective_number_of_assets": effective_assets,
            "average_max_weight": 0.9,
            "average_turnover": 0.3,
            "sharpe": 0.5,
            "robust_score": robust_score,
            "mandate_aware_score": mandate_aware_score,
            "max_drawdown": -0.2,
            "annualized_return": 0.05,
        }


if __name__ == "__main__":
    unittest.main()
