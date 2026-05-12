"""Tests for the TD3 seed sensitivity runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.experiments.run_seed_sensitivity import run_seed_sensitivity


class RunSeedSensitivityTests(unittest.TestCase):
    def test_run_seed_sensitivity_returns_expected_top_level_keys(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_seed_sensitivity(config_path, output_dir=temp_dir, seeds=[7, 21])

        self.assertEqual(
            set(result.keys()),
            {
                "output_dir",
                "results_path",
                "summary_path",
                "results",
                "summary",
                "experiment_outputs",
            },
        )

    def test_calls_runner_once_per_seed(self):
        seeds = [7, 21, 42]

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner() as runner_mock:
                run_seed_sensitivity(config_path, output_dir=temp_dir, seeds=seeds)

        self.assertEqual(runner_mock.call_count, len(seeds))

    def test_writes_one_config_per_seed(self):
        seeds = [7, 21, 42]

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_seed_sensitivity(config_path, output_dir=temp_dir, seeds=seeds)

            config_files = list((Path(result["output_dir"]) / "configs").glob("seed_*.yaml"))

        self.assertEqual(len(config_files), len(seeds))

    def test_saved_configs_override_seed_and_hyperparameters(self):
        seeds = [7]

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_seed_sensitivity(
                    config_path,
                    output_dir=temp_dir,
                    seeds=seeds,
                    episodes=100,
                    batch_size=64,
                    actor_learning_rate=0.0003,
                    critic_learning_rate=0.0003,
                )

            saved_config_path = Path(result["output_dir"]) / "configs" / "seed_7.yaml"
            with saved_config_path.open("r", encoding="utf-8") as file:
                saved_config = yaml.safe_load(file)

        self.assertEqual(saved_config["training"]["seed"], 7)
        self.assertEqual(saved_config["training"]["episodes"], 100)
        self.assertEqual(saved_config["td3"]["batch_size"], 64)
        self.assertEqual(saved_config["td3"]["actor_learning_rate"], 0.0003)
        self.assertEqual(saved_config["td3"]["critic_learning_rate"], 0.0003)

    def test_results_has_one_row_per_seed(self):
        seeds = [7, 21, 42]

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_seed_sensitivity(config_path, output_dir=temp_dir, seeds=seeds)

        self.assertEqual(len(result["results"]), len(seeds))

    def test_summary_contains_win_rates(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_seed_sensitivity(config_path, output_dir=temp_dir, seeds=[7, 21])

        self.assertIn("win_rate_vs_best_individual_buyhold_by_sharpe", result["summary"].columns)
        self.assertIn("win_rate_best_policy_agent", result["summary"].columns)

    def test_saves_results_csv(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_seed_sensitivity(config_path, output_dir=temp_dir, seeds=[7, 21])

            self.assertTrue(Path(result["results_path"]).is_file())

    def test_saves_summary_csv(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_runner():
                result = run_seed_sensitivity(config_path, output_dir=temp_dir, seeds=[7, 21])

            self.assertTrue(Path(result["summary_path"]).is_file())

    def _patched_runner(self):
        return patch(
            "src.experiments.run_seed_sensitivity.run_and_save_basic_experiment",
            side_effect=self._mock_run_and_save_basic_experiment,
        )

    @staticmethod
    def _mock_run_and_save_basic_experiment(
        config_path: str,
        output_dir: str,
        experiment_name: str,
    ) -> dict:
        seed = int(experiment_name.replace("seed_", ""))
        agent_sharpe = 1.2 if seed == 7 else 0.7
        agent_cumulative_return = 0.30 if seed == 7 else 0.10
        experiment_result = {
            "validation_metrics_table": RunSeedSensitivityTests._metrics_table(
                agent_sharpe=agent_sharpe,
                agent_cumulative_return=agent_cumulative_return,
            ),
            "test_metrics_table": RunSeedSensitivityTests._metrics_table(
                agent_sharpe=agent_sharpe,
                agent_cumulative_return=agent_cumulative_return,
            ),
            "validation_comparison_summary": RunSeedSensitivityTests._summary(
                agent_sharpe=agent_sharpe,
                agent_cumulative_return=agent_cumulative_return,
            ),
            "test_comparison_summary": RunSeedSensitivityTests._summary(
                agent_sharpe=agent_sharpe,
                agent_cumulative_return=agent_cumulative_return,
            ),
            "test_diagnostics": RunSeedSensitivityTests._diagnostics(),
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
                "cumulative_return": [agent_cumulative_return, 0.15, 0.12],
                "annualized_return": [0.20, 0.16, 0.14],
                "annualized_volatility": [0.10, 0.11, 0.12],
                "sharpe_ratio": [agent_sharpe, 0.9, 0.8],
                "max_drawdown": [-0.08, -0.10, -0.12],
            },
            index=["agent", "equal_weight_gross", "buy_hold_GLD"],
        )

    @staticmethod
    def _summary(agent_sharpe: float, agent_cumulative_return: float) -> dict:
        best_individual_sharpe = 0.9
        best_individual_return = 0.15
        return {
            "best_policy_by_sharpe": "agent" if agent_sharpe > best_individual_sharpe else "buy_hold_GLD",
            "agent_rank_by_sharpe": 1 if agent_sharpe > best_individual_sharpe else 2,
            "best_individual_buyhold_by_sharpe": "buy_hold_GLD",
            "best_individual_buyhold_sharpe_ratio": best_individual_sharpe,
            "best_individual_buyhold_cumulative_return": best_individual_return,
            "agent_vs_best_individual_buyhold_sharpe_diff": agent_sharpe
            - best_individual_sharpe,
            "agent_vs_best_individual_buyhold_cumulative_return_diff": (
                agent_cumulative_return - best_individual_return
            ),
        }

    @staticmethod
    def _diagnostics() -> dict:
        return {
            "average_turnover": 0.2,
            "average_effective_number_of_assets": 3.5,
            "final_max_weight": 0.6,
        }

    def _temporary_config(self):
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "config.yaml"
        config_path.write_text(
            """
project:
  name: portfolio_drl_td3_test

td3:
  actor_learning_rate: 0.001
  critic_learning_rate: 0.001
  batch_size: 256

training:
  seed: 42
  episodes: 500
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
