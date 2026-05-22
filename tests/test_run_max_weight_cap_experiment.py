"""Tests for experiment-only max-weight cap runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.experiments.run_feature_block_ablation as ablation_module
from src.env.portfolio_env import PortfolioEnv
from src.experiments.run_max_weight_cap_experiment import (
    CappedPortfolioEnv,
    apply_max_weight_cap_to_action,
    build_max_weight_cap_rankings,
    build_max_weight_grid_candidates,
    label_max_weight_cap_decision,
    max_weight_candidate_name,
    patched_portfolio_env,
    parse_max_weight_grid,
    project_weights_to_max_cap,
    run_max_weight_cap_experiment,
)


class MaxWeightCapExperimentTests(unittest.TestCase):
    def test_projection_preserves_sum_to_one(self):
        projected = project_weights_to_max_cap(np.array([0.90, 0.05, 0.05]), 0.60)

        self.assertAlmostEqual(float(projected.sum()), 1.0)

    def test_projection_enforces_cap(self):
        projected = project_weights_to_max_cap(np.array([0.90, 0.05, 0.05]), 0.60)

        self.assertLessEqual(float(projected.max()), 0.60 + 1e-10)

    def test_projection_preserves_non_negativity(self):
        projected = project_weights_to_max_cap(np.array([0.90, 0.05, 0.05]), 0.60)

        self.assertTrue((projected >= 0.0).all())

    def test_uncapped_baseline_does_not_change_weights(self):
        weights = np.array([0.90, 0.05, 0.05])

        projected = apply_max_weight_cap_to_action(weights, None)

        np.testing.assert_allclose(projected, weights)

    def test_capped_env_with_none_matches_normal_env_for_step(self):
        returns = pd.DataFrame(
            {"SPY": [0.01], "TLT": [0.0], "CASH": [0.0]},
            index=pd.date_range("2024-01-05", periods=1, freq="W-FRI"),
        )
        features = pd.DataFrame(
            {"feature": [1.0]},
            index=returns.index,
        )
        action = np.array([0.90, 0.05, 0.05])
        normal_env = PortfolioEnv(
            returns=returns,
            features=features,
            transaction_cost=0.001,
            reward_config={"lambda_return": 1.0, "lambda_transaction_cost": 1.0},
        )

        class UncappedEnv(CappedPortfolioEnv):
            max_weight_cap = None

        capped_env = UncappedEnv(
            returns=returns,
            features=features,
            transaction_cost=0.001,
            reward_config={"lambda_return": 1.0, "lambda_transaction_cost": 1.0},
        )

        normal_env.reset()
        capped_env.reset()
        _, normal_reward, _, normal_info = normal_env.step(action)
        _, capped_reward, _, capped_info = capped_env.step(action)

        np.testing.assert_allclose(capped_info["weights"], normal_info["weights"])
        self.assertAlmostEqual(capped_info["portfolio_return"], normal_info["portfolio_return"])
        self.assertAlmostEqual(capped_info["turnover"], normal_info["turnover"])
        self.assertAlmostEqual(capped_reward, normal_reward)

    def test_uncapped_patch_context_does_not_replace_ablation_env(self):
        original = ablation_module.PortfolioEnv

        with patched_portfolio_env(None):
            self.assertIs(ablation_module.PortfolioEnv, original)

        self.assertIs(ablation_module.PortfolioEnv, original)

    def test_capped_patch_context_is_restored(self):
        original = ablation_module.PortfolioEnv

        with patched_portfolio_env(0.8):
            self.assertIsNot(ablation_module.PortfolioEnv, original)

        self.assertIs(ablation_module.PortfolioEnv, original)

    def test_parse_max_weight_grid(self):
        self.assertEqual(parse_max_weight_grid("uncapped,0.8,0.7"), [None, 0.8, 0.7])

    def test_isolated_candidate_names(self):
        self.assertEqual(
            max_weight_candidate_name("V5_no_volatility_block", None),
            "V5_no_volatility_block_cap_uncapped",
        )
        self.assertEqual(
            max_weight_candidate_name("V5_no_volatility_block", 0.8),
            "V5_no_volatility_block_cap_0p80",
        )

    def test_grid_candidates_preserve_base_candidate(self):
        base = {
            "name": "V5_no_volatility_block",
            "feature_version": "v5",
            "exclude_blocks": ["volatility"],
            "use_dynamic_cash": True,
            "cash_risk_off_column": "risk_off_state",
        }

        candidates = build_max_weight_grid_candidates(base, [None, 0.8])

        self.assertEqual(candidates[0]["name"], "V5_no_volatility_block_cap_uncapped")
        self.assertEqual(candidates[1]["max_weight_cap"], 0.8)
        self.assertEqual(base["name"], "V5_no_volatility_block")

    def test_baseline_deltas_are_computed(self):
        summary = pd.DataFrame(
            [
                self._summary_row("base", None, 1.1, 0.9, 0.50, 0.40),
                self._summary_row("cap", 0.8, 1.4, 0.7, 0.49, 0.39),
            ]
        )

        rankings = build_max_weight_cap_rankings(summary)
        cap = rankings.set_index("candidate_name").loc["cap"]

        self.assertAlmostEqual(
            cap["delta_average_effective_number_of_assets_vs_baseline"],
            0.30,
        )
        self.assertAlmostEqual(cap["delta_average_max_weight_vs_baseline"], -0.20)
        self.assertEqual(cap["decision_label"], "dominates_baseline")

    def test_decision_labels_work_on_synthetic_rows(self):
        self.assertEqual(
            label_max_weight_cap_decision(
                pd.Series(
                    {
                        "max_weight_cap": 0.8,
                        "delta_average_effective_number_of_assets_vs_baseline": 0.30,
                        "delta_average_max_weight_vs_baseline": -0.10,
                        "delta_robust_score_vs_baseline": 0.0,
                        "delta_mandate_aware_score_vs_baseline": 0.0,
                        "delta_max_drawdown_vs_baseline": 0.0,
                    }
                )
            ),
            "dominates_baseline",
        )
        self.assertEqual(
            label_max_weight_cap_decision(
                pd.Series(
                    {
                        "max_weight_cap": 0.7,
                        "delta_average_effective_number_of_assets_vs_baseline": 0.30,
                        "delta_average_max_weight_vs_baseline": -0.10,
                        "delta_robust_score_vs_baseline": -0.10,
                    }
                )
            ),
            "diversifies_but_hurts_performance",
        )
        self.assertEqual(
            label_max_weight_cap_decision(
                pd.Series(
                    {
                        "max_weight_cap": 0.7,
                        "delta_average_effective_number_of_assets_vs_baseline": 0.05,
                    }
                )
            ),
            "no_behavioral_improvement",
        )

    def test_smoke_mode_creates_expected_files_with_mocked_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "cap_smoke"
            with (
                patch(
                    "src.experiments.run_max_weight_cap_experiment._build_base_config",
                    return_value=self._base_config(),
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment.build_returns_dataset_from_config",
                    return_value=self._returns(),
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment._build_feature_context",
                    return_value={},
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment._candidate_raw_features",
                    return_value=self._returns(),
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment._candidate_auxiliary_features",
                    return_value=pd.DataFrame(
                        {"risk_off_state": [0, 0, 0]},
                        index=self._returns().index,
                    ),
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment.build_ablation_fold_datasets",
                    return_value={
                        "train_returns": self._returns(),
                        "validation_returns": self._returns(),
                        "test_returns": self._returns(),
                    },
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment.train_td3_ablation_on_datasets",
                    return_value={},
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment._build_experiment_result",
                    return_value=self._experiment_result(),
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment.save_basic_experiment_outputs",
                    return_value={},
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment.build_robust_score_report",
                    return_value={
                        "ranking": pd.DataFrame(
                            [
                                {
                                    "strategy": "V5_no_volatility_block_cap_uncapped",
                                    "robust_score": 0.4,
                                    "median_run_dsr_n25": 0.1,
                                    "date_averaged_dsr_n25": 0.1,
                                    "dsr_method": "median_run",
                                },
                                {
                                    "strategy": "V5_no_volatility_block_cap_0p80",
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
                report = run_max_weight_cap_experiment(
                    output_dir=str(output_dir),
                    max_weight_grid=[None, 0.8],
                    episodes=5,
                    seeds=[7],
                    max_folds=1,
                    smoke=True,
                )

            for path in report["paths"].values():
                self.assertTrue(Path(path).exists())
            self.assertTrue(report["metadata"]["smoke_mode"])
            self.assertIn("experiment-only", report["metadata"]["experiment_only_warning"])

    def test_selected_candidate_is_passed_to_feature_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "cap_smoke"
            with (
                patch(
                    "src.experiments.run_max_weight_cap_experiment._build_base_config",
                    return_value=self._base_config(),
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment.build_returns_dataset_from_config",
                    return_value=self._returns(),
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment._build_feature_context",
                    return_value={},
                ) as build_feature_context,
                patch(
                    "src.experiments.run_max_weight_cap_experiment._candidate_raw_features",
                    return_value=self._returns(),
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment._candidate_auxiliary_features",
                    return_value=pd.DataFrame(index=self._returns().index),
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment.build_ablation_fold_datasets",
                    return_value={
                        "train_returns": self._returns(),
                        "validation_returns": self._returns(),
                        "test_returns": self._returns(),
                    },
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment.train_td3_ablation_on_datasets",
                    return_value={},
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment._build_experiment_result",
                    return_value=self._experiment_result(),
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment.save_basic_experiment_outputs",
                    return_value={},
                ),
                patch(
                    "src.experiments.run_max_weight_cap_experiment.build_robust_score_report",
                    return_value={
                        "ranking": pd.DataFrame(
                            [
                                {
                                    "strategy": "V3_real_macro_current_cap_uncapped",
                                    "robust_score": 0.4,
                                    "median_run_dsr_n25": 0.1,
                                    "date_averaged_dsr_n25": 0.1,
                                    "dsr_method": "median_run",
                                }
                            ]
                        )
                    },
                ),
            ):
                run_max_weight_cap_experiment(
                    output_dir=str(output_dir),
                    candidate="V3_real_macro_current",
                    max_weight_grid=[None],
                    episodes=5,
                    seeds=[7],
                    max_folds=1,
                    smoke=True,
                )

            selected_candidates = build_feature_context.call_args.args[2]
            self.assertEqual(selected_candidates[0]["name"], "V3_real_macro_current")

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
        cap: float | None,
        effective_assets: float,
        average_max_weight: float,
        robust_score: float,
        mandate_aware_score: float,
    ):
        return {
            "candidate_name": name,
            "max_weight_cap": cap,
            "split": "test",
            "average_effective_number_of_assets": effective_assets,
            "average_max_weight": average_max_weight,
            "average_turnover": 0.3,
            "sharpe": 0.5,
            "robust_score": robust_score,
            "mandate_aware_score": mandate_aware_score,
            "max_drawdown": -0.2,
            "annualized_return": 0.05,
        }


if __name__ == "__main__":
    unittest.main()
