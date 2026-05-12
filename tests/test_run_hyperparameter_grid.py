"""Tests for the hyperparameter grid experiment runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.experiments.run_hyperparameter_grid import (
    DEFAULT_HYPERPARAMETER_EXPERIMENTS,
    run_hyperparameter_grid,
)


class RunHyperparameterGridTests(unittest.TestCase):
    def test_run_hyperparameter_grid_returns_expected_top_level_keys(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_grid(config_path, output_dir=temp_dir)

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

    def test_uses_default_hyperparameter_experiments_when_experiments_is_none(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner() as runner_mock:
                result = run_hyperparameter_grid(config_path, output_dir=temp_dir)

        self.assertEqual(runner_mock.call_count, len(DEFAULT_HYPERPARAMETER_EXPERIMENTS))
        self.assertEqual(
            len(result["aggregate_results"]),
            len(DEFAULT_HYPERPARAMETER_EXPERIMENTS),
        )

    def test_accepts_custom_experiments_list(self):
        experiments = self._custom_experiments()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner() as runner_mock:
                result = run_hyperparameter_grid(
                    config_path,
                    output_dir=temp_dir,
                    experiments=experiments,
                )

        self.assertEqual(runner_mock.call_count, len(experiments))
        self.assertEqual(len(result["aggregate_results"]), len(experiments))

    def test_writes_one_config_file_per_experiment(self):
        experiments = self._custom_experiments()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_grid(
                    config_path,
                    output_dir=temp_dir,
                    experiments=experiments,
                )

            config_files = list((Path(result["grid_output_dir"]) / "configs").glob("*.yaml"))

        self.assertEqual(len(config_files), len(experiments))

    def test_saved_config_contains_overridden_hyperparameters(self):
        experiments = self._custom_experiments()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_grid(
                    config_path,
                    output_dir=temp_dir,
                    experiments=experiments,
                )

            config_path = Path(result["aggregate_results"].iloc[0]["config_path"])
            with config_path.open("r", encoding="utf-8") as file:
                saved_config = yaml.safe_load(file)

        self.assertEqual(saved_config["training"]["episodes"], experiments[0]["episodes"])
        self.assertEqual(saved_config["td3"]["batch_size"], experiments[0]["batch_size"])
        self.assertEqual(
            saved_config["td3"]["actor_learning_rate"],
            experiments[0]["actor_learning_rate"],
        )
        self.assertEqual(
            saved_config["td3"]["critic_learning_rate"],
            experiments[0]["critic_learning_rate"],
        )

    def test_calls_run_and_save_basic_experiment_once_per_experiment(self):
        experiments = self._custom_experiments()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner() as runner_mock:
                run_hyperparameter_grid(
                    config_path,
                    output_dir=temp_dir,
                    experiments=experiments,
                )

        self.assertEqual(runner_mock.call_count, len(experiments))

    def test_aggregate_results_has_one_row_per_experiment(self):
        experiments = self._custom_experiments()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_grid(
                    config_path,
                    output_dir=temp_dir,
                    experiments=experiments,
                )

        self.assertEqual(len(result["aggregate_results"]), len(experiments))

    def test_aggregate_results_includes_key_columns(self):
        experiments = self._custom_experiments()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_grid(
                    config_path,
                    output_dir=temp_dir,
                    experiments=experiments,
                )

        expected_columns = {
            "experiment_id",
            "test_agent_sharpe_ratio",
            "test_agent_cumulative_return",
            "test_average_effective_number_of_assets",
            "test_equal_weight_rebalanced_net_sharpe_ratio",
            "test_best_individual_buyhold_by_sharpe",
            "test_agent_vs_best_individual_buyhold_sharpe_diff",
            "validation_best_individual_buyhold_by_sharpe",
            "validation_agent_vs_best_individual_buyhold_sharpe_diff",
        }
        self.assertTrue(expected_columns.issubset(set(result["aggregate_results"].columns)))

    def test_saves_aggregate_results_csv(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_grid(
                    config_path,
                    output_dir=temp_dir,
                    experiments=self._custom_experiments(),
                )

            self.assertTrue(Path(result["aggregate_results_path"]).is_file())

    def test_saves_ranking_results_csv(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_grid(
                    config_path,
                    output_dir=temp_dir,
                    experiments=self._custom_experiments(),
                )

            self.assertTrue(Path(result["ranking_results_path"]).is_file())

    def test_ranking_table_is_sorted_by_test_agent_sharpe_descending(self):
        experiments = self._custom_experiments()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_hyperparameter_grid(
                    config_path,
                    output_dir=temp_dir,
                    experiments=experiments,
                )

        ranking = result["ranking_results"]
        self.assertEqual(list(ranking["experiment_id"]), ["Y", "X"])

    def _patched_runner(self):
        return patch(
            "src.experiments.run_hyperparameter_grid.run_and_save_basic_experiment",
            side_effect=self._mock_run_and_save_basic_experiment,
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
    def _mock_run_and_save_basic_experiment(
        config_path: str,
        output_dir: str,
        experiment_name: str,
    ) -> dict:
        sharpe = 0.5 if "experiment_X" in experiment_name else 1.5
        cumulative_return = 0.01 if "experiment_X" in experiment_name else 0.03
        experiment_result = {
            "training_summary": {
                "final_portfolio_value": 101000.0,
                "final_total_reward": 0.01,
                "final_average_turnover": 0.2,
                "final_average_transaction_cost": 0.0002,
                "final_max_weight": 0.4,
                "final_cash_weight": 0.1,
            },
            "validation_metrics_table": RunHyperparameterGridTests._metrics_table(
                agent_sharpe=sharpe,
                agent_cumulative_return=cumulative_return,
            ),
            "test_metrics_table": RunHyperparameterGridTests._metrics_table(
                agent_sharpe=sharpe,
                agent_cumulative_return=cumulative_return,
            ),
            "validation_comparison_summary": RunHyperparameterGridTests._summary(sharpe),
            "test_comparison_summary": RunHyperparameterGridTests._summary(sharpe),
            "validation_diagnostics": RunHyperparameterGridTests._diagnostics(),
            "test_diagnostics": RunHyperparameterGridTests._diagnostics(),
        }
        return {
            "experiment_result": experiment_result,
            "saved_paths": {
                "output_dir": str(Path(output_dir) / experiment_name),
            },
        }

    @staticmethod
    def _metrics_table(agent_sharpe: float, agent_cumulative_return: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cumulative_return": [agent_cumulative_return, 0.02, 0.015, 0.01, 0.025],
                "annualized_return": [0.10, 0.08, 0.07, 0.06, 0.09],
                "annualized_volatility": [0.05, 0.06, 0.05, 0.07, 0.08],
                "sharpe_ratio": [agent_sharpe, 0.8, 0.7, 0.6, 0.9],
                "max_drawdown": [-0.02, -0.03, -0.04, -0.05, -0.06],
            },
            index=[
                "agent",
                "equal_weight_gross",
                "equal_weight_rebalanced_net",
                "buy_and_hold",
                "buy_hold_SPY",
            ],
        )

    @staticmethod
    def _summary(agent_sharpe: float) -> dict:
        return {
            "best_policy_by_sharpe": "agent" if agent_sharpe > 0.8 else "equal_weight_gross",
            "best_sharpe_ratio": max(agent_sharpe, 0.8),
            "agent_rank_by_sharpe": 1 if agent_sharpe > 0.8 else 3,
            "agent_vs_equal_weight_rebalanced_net_sharpe_diff": agent_sharpe - 0.7,
            "agent_vs_buy_and_hold_sharpe_diff": agent_sharpe - 0.6,
            "best_individual_buyhold_by_sharpe": "buy_hold_SPY",
            "best_individual_buyhold_sharpe_ratio": 0.9,
            "best_individual_buyhold_cumulative_return": 0.025,
            "agent_vs_best_individual_buyhold_sharpe_diff": agent_sharpe - 0.9,
            "agent_vs_best_individual_buyhold_cumulative_return_diff": 0.01 - 0.025,
            "agent_vs_equal_weight_rebalanced_net_cumulative_return_diff": 0.01,
            "agent_vs_buy_and_hold_cumulative_return_diff": 0.02,
        }

    @staticmethod
    def _diagnostics() -> dict:
        return {
            "average_max_weight": 0.35,
            "final_max_weight": 0.4,
            "average_cash_weight": 0.15,
            "final_cash_weight": 0.1,
            "average_herfindahl_index": 0.25,
            "final_herfindahl_index": 0.3,
            "average_effective_number_of_assets": 4.0,
            "final_effective_number_of_assets": 3.3,
            "average_entropy": 1.4,
            "final_entropy": 1.3,
            "average_turnover": 0.2,
            "final_turnover": 0.25,
            "average_transaction_cost": 0.0002,
            "final_transaction_cost": 0.00025,
        }

    def _temporary_config(self):
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "config.yaml"
        config_path.write_text(
            """
project:
  name: portfolio_drl_td3_test
  description: Temporary hyperparameter grid test config

data:
  assets:
    - SPY
    - TLT
    - GLD
    - BTC-USD
    - CASH
  frequency: weekly
  start_date: 2020-01-01
  end_date: 2024-01-01

environment:
  initial_cash: 100000
  transaction_cost: 0.001
  allow_short: false
  max_weight_per_asset: 1.0

reward:
  lambda_return: 1.0
  lambda_sharpe: 0.5
  lambda_drawdown: 1.0
  lambda_transaction_cost: 0.2
  lambda_turnover: 0.1

td3:
  actor_learning_rate: 0.0003
  critic_learning_rate: 0.0003
  gamma: 0.99
  tau: 0.005
  policy_noise: 0.2
  noise_clip: 0.5
  policy_delay: 2
  batch_size: 256
  replay_buffer_size: 100000

training:
  seed: 42
  episodes: 500
  train_ratio: 0.7
  validation_ratio: 0.15
  test_ratio: 0.15
""",
            encoding="utf-8",
        )

        class TemporaryConfig:
            def __enter__(self_inner):
                return str(config_path)

            def __exit__(self_inner, exc_type, exc_value, traceback):
                temp_dir.cleanup()
                return False

        return TemporaryConfig()


if __name__ == "__main__":
    unittest.main()
