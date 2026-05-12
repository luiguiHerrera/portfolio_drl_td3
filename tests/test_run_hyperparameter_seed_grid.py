"""Tests for the hyperparameter by seed robustness runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.experiments.run_hyperparameter_grid import DEFAULT_HYPERPARAMETER_EXPERIMENTS
from src.experiments.run_hyperparameter_seed_grid import run_hyperparameter_seed_grid


class RunHyperparameterSeedGridTests(unittest.TestCase):
    def test_run_hyperparameter_seed_grid_returns_expected_top_level_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_seed_grid("config.yaml", output_dir=temp_dir)

        self.assertEqual(
            set(result.keys()),
            {
                "grid_output_dir",
                "aggregate_results_path",
                "ranking_results_path",
                "aggregate_results",
                "ranking_results",
                "experiment_outputs",
            },
        )

    def test_uses_default_experiments_when_experiments_is_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner() as runner_mock:
                result = run_hyperparameter_seed_grid("config.yaml", output_dir=temp_dir)

        self.assertEqual(runner_mock.call_count, len(DEFAULT_HYPERPARAMETER_EXPERIMENTS))
        self.assertEqual(len(result["aggregate_results"]), len(DEFAULT_HYPERPARAMETER_EXPERIMENTS))

    def test_accepts_custom_experiments_list(self):
        experiments = self._custom_experiments()

        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner() as runner_mock:
                result = run_hyperparameter_seed_grid(
                    "config.yaml",
                    output_dir=temp_dir,
                    experiments=experiments,
                )

        self.assertEqual(runner_mock.call_count, len(experiments))
        self.assertEqual(len(result["aggregate_results"]), len(experiments))

    def test_calls_run_seed_sensitivity_once_per_experiment(self):
        experiments = self._custom_experiments()

        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner() as runner_mock:
                run_hyperparameter_seed_grid(
                    "config.yaml",
                    output_dir=temp_dir,
                    experiments=experiments,
                )

        self.assertEqual(runner_mock.call_count, len(experiments))

    def test_each_call_passes_experiment_hyperparameters(self):
        experiments = self._custom_experiments()

        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner() as runner_mock:
                run_hyperparameter_seed_grid(
                    "config.yaml",
                    output_dir=temp_dir,
                    experiments=experiments,
                    seeds=[7, 21],
                )

        first_call_kwargs = runner_mock.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["episodes"], experiments[0]["episodes"])
        self.assertEqual(first_call_kwargs["batch_size"], experiments[0]["batch_size"])
        self.assertEqual(
            first_call_kwargs["actor_learning_rate"],
            experiments[0]["actor_learning_rate"],
        )
        self.assertEqual(
            first_call_kwargs["critic_learning_rate"],
            experiments[0]["critic_learning_rate"],
        )
        self.assertEqual(first_call_kwargs["seeds"], [7, 21])

    def test_aggregate_results_has_one_row_per_experiment(self):
        experiments = self._custom_experiments()

        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_seed_grid(
                    "config.yaml",
                    output_dir=temp_dir,
                    experiments=experiments,
                )

        self.assertEqual(len(result["aggregate_results"]), len(experiments))

    def test_aggregate_results_contains_robust_metric_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_seed_grid(
                    "config.yaml",
                    output_dir=temp_dir,
                    experiments=self._custom_experiments(),
                )

        expected_columns = {
            "robust_test_agent_sharpe_score_05",
            "robust_test_agent_sharpe_score_10",
            "robust_test_agent_sortino_score_05",
            "robust_test_agent_information_ratio_score_05",
            "robust_test_agent_capm_alpha_score_05",
            "positive_sharpe_rate",
            "win_rate_best_policy_agent",
            "worst_test_agent_sharpe",
            "mean_minus_worst_sharpe_gap",
        }
        self.assertTrue(expected_columns.issubset(set(result["aggregate_results"].columns)))

    def test_saves_aggregate_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_seed_grid(
                    "config.yaml",
                    output_dir=temp_dir,
                    experiments=self._custom_experiments(),
                )

            self.assertTrue(Path(result["aggregate_results_path"]).is_file())

    def test_saves_ranking_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_seed_grid(
                    "config.yaml",
                    output_dir=temp_dir,
                    experiments=self._custom_experiments(),
                )

            self.assertTrue(Path(result["ranking_results_path"]).is_file())

    def test_ranking_is_sorted_by_robust_sharpe_descending(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_seed_grid(
                    "config.yaml",
                    output_dir=temp_dir,
                    experiments=self._custom_experiments(),
                )

        self.assertEqual(list(result["ranking_results"]["experiment_id"]), ["Y", "X"])

    def test_experiment_outputs_is_keyed_by_experiment_id(self):
        experiments = self._custom_experiments()

        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_seed_grid(
                    "config.yaml",
                    output_dir=temp_dir,
                    experiments=experiments,
                )

        self.assertEqual(set(result["experiment_outputs"].keys()), {"X", "Y"})

    def _patched_runner(self):
        return patch(
            "src.experiments.run_hyperparameter_seed_grid.run_seed_sensitivity",
            side_effect=self._mock_run_seed_sensitivity,
        )

    @staticmethod
    def _custom_experiments() -> list[dict]:
        return [
            {
                "experiment_id": "X",
                "description": "custom_low",
                "episodes": 2,
                "batch_size": 8,
                "actor_learning_rate": 0.0001,
                "critic_learning_rate": 0.0001,
            },
            {
                "experiment_id": "Y",
                "description": "custom_high",
                "episodes": 3,
                "batch_size": 16,
                "actor_learning_rate": 0.0005,
                "critic_learning_rate": 0.0005,
            },
        ]

    @staticmethod
    def _mock_run_seed_sensitivity(
        base_config_path: str,
        output_dir: str,
        experiment_name: str,
        seeds: list[int],
        episodes: int,
        batch_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
    ) -> dict:
        robust_sharpe = 0.2 if "experiment_X" in experiment_name else 1.3
        summary = RunHyperparameterSeedGridTests._summary_dict(robust_sharpe)
        return {
            "output_dir": str(Path(output_dir) / experiment_name),
            "results_path": str(Path(output_dir) / experiment_name / "seed_sensitivity_results.csv"),
            "summary_path": str(Path(output_dir) / experiment_name / "seed_sensitivity_summary.csv"),
            "results": pd.DataFrame({"seed": seeds, "episodes": episodes}),
            "summary": pd.DataFrame([summary]),
            "experiment_outputs": {},
        }

    @staticmethod
    def _summary_dict(robust_sharpe: float) -> dict:
        return {
            "n_seeds": 2,
            "mean_test_agent_sharpe": robust_sharpe + 0.1,
            "std_test_agent_sharpe": 0.2,
            "robust_test_agent_sharpe_score_05": robust_sharpe,
            "robust_test_agent_sharpe_score_10": robust_sharpe - 0.1,
            "mean_test_agent_sortino": robust_sharpe + 0.2,
            "std_test_agent_sortino": 0.3,
            "robust_test_agent_sortino_score_05": robust_sharpe + 0.05,
            "mean_test_agent_information_ratio_vs_equal_weight_rebalanced_net": (
                robust_sharpe + 0.3
            ),
            "std_test_agent_information_ratio_vs_equal_weight_rebalanced_net": 0.4,
            "robust_test_agent_information_ratio_score_05": robust_sharpe + 0.1,
            "mean_test_agent_capm_beta_vs_SPY": 0.9,
            "mean_test_agent_capm_alpha_vs_SPY": robust_sharpe + 0.01,
            "std_test_agent_capm_alpha_vs_SPY": 0.05,
            "robust_test_agent_capm_alpha_score_05": robust_sharpe,
            "mean_test_agent_cumulative_return": 0.2,
            "mean_test_agent_max_drawdown": -0.1,
            "worst_test_agent_sharpe": robust_sharpe - 0.5,
            "worst_test_agent_cumulative_return": -0.05,
            "worst_test_agent_max_drawdown": -0.2,
            "positive_sharpe_rate": 0.5 if robust_sharpe < 1.0 else 1.0,
            "positive_sortino_rate": 1.0,
            "positive_capm_alpha_rate": 0.5,
            "positive_information_ratio_rate": 0.5,
            "win_rate_best_policy_agent": 0.5,
            "win_rate_vs_best_individual_buyhold_by_sharpe": 0.5,
            "mean_test_average_turnover": 0.2,
            "mean_test_average_effective_number_of_assets": 3.0,
            "mean_minus_worst_sharpe_gap": 0.6,
        }


if __name__ == "__main__":
    unittest.main()
