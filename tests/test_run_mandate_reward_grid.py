"""Tests for the mandate reward mini-grid runner."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.experiments.run_mandate_reward_grid import (
    build_mandate_reward_config,
    get_default_mandate_reward_grid,
    run_mandate_reward_grid,
)


class RunMandateRewardGridTests(unittest.TestCase):
    def test_default_grid_includes_baseline_and_mandate_runs(self):
        grid = get_default_mandate_reward_grid()
        names = {item["run_name"] for item in grid}
        lambdas = {item["lambda_mandate"] for item in grid if item["use_mandate_penalty"]}

        self.assertIn("baseline_no_mandate", names)
        self.assertIn("moderate_balanced_lambda_001", names)
        self.assertIn("moderate_balanced_lambda_003", names)
        self.assertIn("moderate_balanced_lambda_005", names)
        self.assertIn("moderate_balanced_lambda_0075", names)
        self.assertEqual(lambdas, {0.001, 0.003, 0.005, 0.0075})

    def test_generated_baseline_config_disables_mandate_penalty(self):
        baseline = get_default_mandate_reward_grid()[0]

        result = build_mandate_reward_config(
            self._base_config(),
            baseline,
            returns_path="returns.csv",
            episodes=5,
        )

        self.assertFalse(result["reward"]["use_mandate_penalty"])
        self.assertNotIn("lambda_mandate", result["reward"])
        self.assertEqual(result["training"]["episodes"], 5)

    def test_generated_mandate_config_enables_penalty_and_sets_fields(self):
        mandate = get_default_mandate_reward_grid()[1]

        result = build_mandate_reward_config(
            self._base_config(),
            mandate,
            returns_path="returns.csv",
            episodes=5,
            seed=7,
        )

        self.assertTrue(result["reward"]["use_mandate_penalty"])
        self.assertEqual(result["reward"]["lambda_mandate"], 0.001)
        self.assertEqual(result["reward"]["mandate_profile"], "moderate")
        self.assertEqual(
            result["reward"]["mandate_penalty_weights"],
            mandate["mandate_penalty_weights"],
        )
        self.assertEqual(result["data"]["returns_path"], "returns.csv")
        self.assertEqual(result["data"]["returns_date_column"], "date")
        self.assertEqual(result["training"]["seed"], 7)

    def test_runner_writes_generated_configs(self):
        grid = get_default_mandate_reward_grid()[:2]
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            returns_path = self._write_returns(temp_dir)
            with self._patched_runner():
                result = run_mandate_reward_grid(
                    base_config_path=config_path,
                    output_dir=str(Path(temp_dir) / "grid"),
                    returns_path=returns_path,
                    episodes=3,
                    grid=grid,
                    seeds=[7, 42],
                    run_diagnostics=False,
                )

            config_files = list(Path(result["configs_dir"]).glob("*.yaml"))

        self.assertEqual(len(config_files), len(grid) * 2)

    def test_runner_calls_experiment_once_per_grid_item_per_seed(self):
        grid = get_default_mandate_reward_grid()[:3]
        seeds = [7, 42]
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            returns_path = self._write_returns(temp_dir)
            with self._patched_runner() as runner_mock:
                run_mandate_reward_grid(
                    base_config_path=config_path,
                    output_dir=str(Path(temp_dir) / "grid"),
                    returns_path=returns_path,
                    grid=grid,
                    seeds=seeds,
                    run_diagnostics=False,
                )

        self.assertEqual(runner_mock.call_count, len(grid) * len(seeds))

    def test_seed_is_injected_into_generated_config(self):
        grid = get_default_mandate_reward_grid()[:1]
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            returns_path = self._write_returns(temp_dir)
            with self._patched_runner():
                result = run_mandate_reward_grid(
                    base_config_path=config_path,
                    output_dir=str(Path(temp_dir) / "grid"),
                    returns_path=returns_path,
                    grid=grid,
                    seeds=[101],
                    run_diagnostics=False,
                )
            saved_config_path = Path(result["configs_dir"]) / "baseline_no_mandate_seed_101.yaml"
            saved_config = yaml.safe_load(saved_config_path.read_text(encoding="utf-8"))

        self.assertEqual(saved_config["training"]["seed"], 101)

    def test_mandate_runs_use_correct_lambda_values(self):
        grid = get_default_mandate_reward_grid()
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            returns_path = self._write_returns(temp_dir)
            with self._patched_runner():
                result = run_mandate_reward_grid(
                    base_config_path=config_path,
                    output_dir=str(Path(temp_dir) / "grid"),
                    returns_path=returns_path,
                    grid=grid,
                    seeds=[7],
                    run_diagnostics=False,
                )

            observed_lambdas = {}
            for config_file in Path(result["configs_dir"]).glob("*.yaml"):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                if config["reward"]["use_mandate_penalty"]:
                    observed_lambdas[config_file.stem.removesuffix("_seed_7")] = (
                        config["reward"]["lambda_mandate"]
                    )

        self.assertEqual(
            observed_lambdas,
            {
                "moderate_balanced_lambda_001": 0.001,
                "moderate_balanced_lambda_003": 0.003,
                "moderate_balanced_lambda_005": 0.005,
                "moderate_balanced_lambda_0075": 0.0075,
            },
        )

    def test_summary_includes_expected_columns(self):
        grid = get_default_mandate_reward_grid()[:2]
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            returns_path = self._write_returns(temp_dir)
            with self._patched_runner():
                result = run_mandate_reward_grid(
                    base_config_path=config_path,
                    output_dir=str(Path(temp_dir) / "grid"),
                    returns_path=returns_path,
                    grid=grid,
                    seeds=[7, 42],
                    run_diagnostics=False,
                )
            summary_path_exists = Path(result["summary_path"]).exists()
            results_path_exists = Path(result["results_path"]).exists()

        expected_columns = {
            "run_name",
            "use_mandate_penalty",
            "lambda_mandate",
            "penalty_set",
            "validation_sharpe",
            "test_sharpe",
            "test_cumulative_return",
            "test_max_drawdown",
            "average_turnover",
            "average_max_weight",
            "average_effective_number_of_assets",
            "final_cash_weight",
            "mean_mandate_penalty",
            "mean_mandate_drawdown_breach",
            "mean_mandate_volatility_breach",
            "mean_mandate_max_weight_breach",
            "mean_mandate_effective_assets_breach",
            "mean_mandate_turnover_breach",
        }
        self.assertTrue(expected_columns.issubset(result["results"].columns))
        self.assertTrue(summary_path_exists)
        self.assertTrue(results_path_exists)

    def test_aggregate_summary_is_computed_correctly(self):
        grid = get_default_mandate_reward_grid()[:1]
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            returns_path = self._write_returns(temp_dir)
            with self._patched_runner():
                result = run_mandate_reward_grid(
                    base_config_path=config_path,
                    output_dir=str(Path(temp_dir) / "grid"),
                    returns_path=returns_path,
                    grid=grid,
                    seeds=[7, 42],
                    run_diagnostics=False,
                )

        aggregate = result["summary"].iloc[0]
        expected_mean = result["results"]["test_sharpe"].mean()
        expected_std = result["results"]["test_sharpe"].std()
        expected_robust = expected_mean - 0.5 * expected_std
        self.assertEqual(aggregate["n_seeds"], 2)
        self.assertAlmostEqual(aggregate["mean_test_sharpe"], expected_mean)
        self.assertAlmostEqual(aggregate["std_test_sharpe"], expected_std)
        self.assertAlmostEqual(
            aggregate["robust_test_sharpe_score_05"],
            expected_robust,
        )

    def test_missing_returns_path_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            missing_returns = str(Path(temp_dir) / "missing_returns.csv")

            with self.assertRaisesRegex(FileNotFoundError, "Returns snapshot not found"):
                run_mandate_reward_grid(
                    base_config_path=config_path,
                    output_dir=str(Path(temp_dir) / "grid"),
                    returns_path=missing_returns,
                    grid=get_default_mandate_reward_grid()[:1],
                    seeds=[42],
                    run_diagnostics=False,
                )

    def test_diagnostics_can_be_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(temp_dir)
            returns_path = self._write_returns(temp_dir)
            with self._patched_runner():
                result = run_mandate_reward_grid(
                    base_config_path=config_path,
                    output_dir=str(Path(temp_dir) / "grid"),
                    returns_path=returns_path,
                    grid=get_default_mandate_reward_grid()[:1],
                    seeds=[42],
                    run_diagnostics=False,
                )

        self.assertIsNone(result["diagnostics"])

    def test_original_base_config_object_is_not_mutated(self):
        base_config = self._base_config()
        original = copy.deepcopy(base_config)

        build_mandate_reward_config(
            base_config,
            get_default_mandate_reward_grid()[1],
            returns_path="returns.csv",
            episodes=5,
        )

        self.assertEqual(base_config, original)

    def _patched_runner(self):
        return patch(
            "src.experiments.run_mandate_reward_grid.run_and_save_basic_experiment",
            side_effect=self._mock_run_and_save_basic_experiment,
        )

    def _mock_run_and_save_basic_experiment(
        self,
        config_path: str,
        output_dir: str,
        experiment_name: str,
    ) -> dict:
        output_path = Path(output_dir) / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)
        history_path = output_path / "test_policy_history.csv"
        policy_history = self._policy_history()
        policy_history.to_csv(history_path, index=False)
        with Path(config_path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        seed = config["training"].get("seed", 0)
        lambda_mandate = config["reward"].get("lambda_mandate", 0.0)
        test_sharpe = 1.0 + seed / 100.0 - lambda_mandate

        return {
            "experiment_result": {
                "training_summary": {"final_portfolio_value": 101000.0},
                "validation_metrics_table": pd.DataFrame(),
                "test_metrics_table": pd.DataFrame(),
                "validation_comparison_summary": {
                    "agent_sharpe_ratio": 0.8,
                    "agent_cumulative_return": 0.03,
                },
                "test_comparison_summary": {
                    "agent_sharpe_ratio": test_sharpe,
                    "agent_cumulative_return": 0.05,
                    "agent_max_drawdown": -0.08,
                },
                "validation_diagnostics": {
                    "average_turnover": 0.2,
                    "average_max_weight": 0.7,
                    "average_effective_number_of_assets": 1.5,
                    "final_cash_weight": 0.1,
                },
                "test_diagnostics": {
                    "average_turnover": 0.3,
                    "average_max_weight": 0.8,
                    "average_effective_number_of_assets": 1.4,
                    "final_cash_weight": 0.2,
                },
                "test_policy_history": policy_history,
            },
            "saved_paths": {
                "output_dir": str(output_path),
                "test_policy_history": str(history_path),
            },
        }

    @staticmethod
    def _policy_history() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": ["2024-01-05", "2024-01-12"],
                "financial_net_return": [0.01, 0.02],
                "drawdown": [0.0, 0.0],
                "turnover": [0.1, 0.2],
                "mandate_penalty": [0.1, 0.2],
                "mandate_drawdown_breach": [0.0, 0.0],
                "mandate_volatility_breach": [0.01, 0.02],
                "mandate_max_weight_breach": [0.05, 0.06],
                "mandate_effective_assets_breach": [0.2, 0.3],
                "mandate_turnover_breach": [0.0, 0.0],
                "weight_SPY": [0.8, 0.7],
                "weight_TLT": [0.1, 0.1],
                "weight_GLD": [0.1, 0.1],
                "weight_BTC-USD": [0.0, 0.0],
                "weight_CASH": [0.0, 0.1],
            }
        )

    def _write_config(self, directory: str) -> str:
        path = Path(directory) / "config.yaml"
        path.write_text(
            yaml.safe_dump(self._base_config(), sort_keys=False),
            encoding="utf-8",
        )

        return str(path)

    @staticmethod
    def _write_returns(directory: str) -> str:
        path = Path(directory) / "returns.csv"
        pd.DataFrame(
            {
                "date": ["2024-01-05", "2024-01-12", "2024-01-19"],
                "SPY": [0.01, 0.02, -0.01],
                "TLT": [0.0, 0.01, 0.0],
                "GLD": [0.01, 0.0, 0.01],
                "BTC-USD": [0.02, -0.01, 0.03],
                "CASH": [0.0, 0.0, 0.0],
            }
        ).to_csv(path, index=False)

        return str(path)

    @staticmethod
    def _base_config() -> dict:
        return {
            "project": {"name": "test_project"},
            "data": {
                "assets": ["SPY", "TLT", "GLD", "BTC-USD", "CASH"],
                "frequency": "weekly",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            },
            "environment": {"initial_cash": 100000, "transaction_cost": 0.001},
            "reward": {
                "lambda_return": 1.0,
                "lambda_transaction_cost": 1.0,
            },
            "td3": {"batch_size": 32},
            "training": {"episodes": 100},
        }


if __name__ == "__main__":
    unittest.main()
