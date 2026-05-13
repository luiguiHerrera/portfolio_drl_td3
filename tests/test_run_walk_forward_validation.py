"""Tests for true walk-forward validation runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.experiments.run_walk_forward_validation import (
    DEFAULT_WALK_FORWARD_FOLDS,
    run_walk_forward_validation,
)


class RunWalkForwardValidationTests(unittest.TestCase):
    def test_run_walk_forward_validation_returns_expected_top_level_keys(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_dependencies():
                result = run_walk_forward_validation(config_path, output_dir=temp_dir)

        self.assertEqual(
            set(result.keys()),
            {
                "output_dir",
                "results_path",
                "summary_path",
                "results",
                "summary",
                "fold_outputs",
            },
        )

    def test_default_folds_are_used(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_dependencies() as (_, train_mock):
                result = run_walk_forward_validation(config_path, output_dir=temp_dir)

        self.assertEqual(train_mock.call_count, len(DEFAULT_WALK_FORWARD_FOLDS))
        self.assertEqual(len(result["results"]), len(DEFAULT_WALK_FORWARD_FOLDS))

    def test_custom_folds_are_accepted(self):
        folds = self._custom_folds()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_dependencies() as (_, train_mock):
                result = run_walk_forward_validation(
                    config_path,
                    output_dir=temp_dir,
                    folds=folds,
                )

        self.assertEqual(train_mock.call_count, len(folds))
        self.assertEqual(len(result["results"]), len(folds))

    def test_train_td3_on_datasets_is_called_once_per_fold(self):
        folds = self._custom_folds()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_dependencies() as (_, train_mock):
                run_walk_forward_validation(config_path, output_dir=temp_dir, folds=folds)

        self.assertEqual(train_mock.call_count, len(folds))

    def test_results_have_one_row_per_fold(self):
        folds = self._custom_folds()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_dependencies():
                result = run_walk_forward_validation(config_path, output_dir=temp_dir, folds=folds)

        self.assertEqual(len(result["results"]), len(folds))

    def test_summary_robust_scores_are_correct(self):
        folds = self._custom_folds()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_dependencies():
                result = run_walk_forward_validation(config_path, output_dir=temp_dir, folds=folds)

        summary = result["summary"].iloc[0]
        expected_score_05 = (
            summary["mean_test_agent_sharpe"] - 0.5 * summary["std_test_agent_sharpe"]
        )
        expected_score_10 = (
            summary["mean_test_agent_sharpe"] - 1.0 * summary["std_test_agent_sharpe"]
        )

        self.assertAlmostEqual(summary["robust_test_agent_sharpe_score_05"], expected_score_05)
        self.assertAlmostEqual(summary["robust_test_agent_sharpe_score_10"], expected_score_10)

    def test_csvs_are_saved(self):
        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_dependencies():
                result = run_walk_forward_validation(
                    config_path,
                    output_dir=temp_dir,
                    folds=self._custom_folds(),
                )

            self.assertTrue(Path(result["results_path"]).is_file())
            self.assertTrue(Path(result["summary_path"]).is_file())

    def test_fold_outputs_are_keyed_by_fold_id(self):
        folds = self._custom_folds()

        with self._temporary_config() as config_path, tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_dependencies():
                result = run_walk_forward_validation(config_path, output_dir=temp_dir, folds=folds)

        self.assertEqual(set(result["fold_outputs"].keys()), {"F1", "F2"})

    def _patched_dependencies(self):
        dataset_patcher = patch(
            "src.experiments.run_walk_forward_validation.build_walk_forward_datasets",
            side_effect=self._mock_build_walk_forward_datasets,
        )
        train_patcher = patch(
            "src.experiments.run_walk_forward_validation.train_td3_on_datasets",
            side_effect=self._mock_train_td3_on_datasets,
        )

        class PatchedDependencies:
            def __enter__(self_inner):
                dataset_mock = dataset_patcher.__enter__()
                train_mock = train_patcher.__enter__()
                return dataset_mock, train_mock

            def __exit__(self_inner, exc_type, exc_value, traceback):
                train_patcher.__exit__(exc_type, exc_value, traceback)
                dataset_patcher.__exit__(exc_type, exc_value, traceback)
                return False

        return PatchedDependencies()

    @staticmethod
    def _mock_build_walk_forward_datasets(config_path: str, fold: dict) -> dict:
        index = pd.date_range("2024-01-01", periods=3, freq="W-FRI")
        returns = pd.DataFrame({"SPY": [0.01, 0.02, -0.01], "CASH": [0.0, 0.0, 0.0]}, index=index)
        features = pd.DataFrame({"feature": [0.1, 0.2, 0.3]}, index=index)

        return {
            "train_returns": returns,
            "validation_returns": returns,
            "test_returns": returns,
            "train_features": features,
            "validation_features": features,
            "test_features": features,
            "feature_scaler": {"mean": features.mean(), "std": features.std(ddof=1)},
        }

    @staticmethod
    def _mock_train_td3_on_datasets(datasets: dict, config: dict) -> dict:
        seed = config["training"]["seed"]
        sharpe = 1.0 + (config["td3"]["batch_size"] / 100.0)
        if seed == 99:
            sharpe += 0.1

        return {
            "episode_logs": [
                {
                    "episode": 1,
                    "final_portfolio_value": 101000.0,
                    "total_reward": 0.1,
                    "average_turnover": 0.2,
                    "average_transaction_cost": 0.001,
                    "max_weight": 0.6,
                    "cash_weight": 0.1,
                }
            ],
            "validation_comparison": {
                "metrics_table": RunWalkForwardValidationTests._metrics_table(sharpe)
            },
            "test_comparison": {
                "metrics_table": RunWalkForwardValidationTests._metrics_table(sharpe)
            },
            "validation_evaluation": {
                "diagnostics": RunWalkForwardValidationTests._diagnostics()
            },
            "test_evaluation": {
                "diagnostics": RunWalkForwardValidationTests._diagnostics()
            },
        }

    @staticmethod
    def _metrics_table(agent_sharpe: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cumulative_return": [0.10, 0.06, 0.05, 0.04, 0.08],
                "annualized_return": [0.20, 0.12, 0.10, 0.08, 0.16],
                "annualized_volatility": [0.10, 0.11, 0.12, 0.13, 0.14],
                "sharpe_ratio": [agent_sharpe, 0.8, 0.7, 0.6, 0.9],
                "max_drawdown": [-0.05, -0.08, -0.09, -0.10, -0.11],
                "sortino_ratio": [agent_sharpe + 0.1, 0.9, 0.8, 0.7, 1.0],
                "calmar_ratio": [agent_sharpe + 0.2, 1.0, 0.9, 0.8, 1.1],
                "information_ratio_vs_equal_weight_rebalanced_net": [
                    agent_sharpe - 0.7,
                    0.1,
                    0.0,
                    -0.1,
                    0.2,
                ],
                "capm_beta_vs_SPY": [0.9, 0.8, 0.7, 0.6, 1.0],
                "capm_alpha_vs_SPY": [0.02, 0.01, 0.0, -0.01, 0.03],
            },
            index=[
                "agent",
                "equal_weight_gross",
                "equal_weight_rebalanced_net",
                "buy_and_hold",
                "buy_hold_GLD",
            ],
        )

    @staticmethod
    def _diagnostics() -> dict:
        return {
            "average_turnover": 0.2,
            "average_effective_number_of_assets": 3.5,
            "final_max_weight": 0.6,
            "average_transaction_cost": 0.001,
            "final_weights": {"SPY": 0.6, "CASH": 0.4},
        }

    @staticmethod
    def _custom_folds() -> list[dict]:
        return [
            {
                "fold_id": "F1",
                "description": "fold_one",
                "train_start": "2021-01-01",
                "train_end": "2021-12-31",
                "validation_start": "2022-01-01",
                "validation_end": "2022-06-30",
                "test_start": "2022-07-01",
                "test_end": "2022-12-31",
            },
            {
                "fold_id": "F2",
                "description": "fold_two",
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "validation_start": "2023-01-01",
                "validation_end": "2023-06-30",
                "test_start": "2023-07-01",
                "test_end": "2023-12-31",
            },
        ]

    def _temporary_config(self):
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "config.yaml"
        config_path.write_text(
            """
project:
  name: portfolio_drl_td3_walk_forward_test

data:
  assets:
    - SPY
    - CASH
  frequency: weekly

environment:
  initial_cash: 100000
  transaction_cost: 0.001

reward:
  lambda_return: 1.0

td3:
  actor_learning_rate: 0.0003
  critic_learning_rate: 0.0003
  gamma: 0.99
  tau: 0.005
  policy_noise: 0.2
  noise_clip: 0.5
  policy_delay: 2
  batch_size: 32
  replay_buffer_size: 1000

training:
  seed: 42
  episodes: 3
  train_ratio: 0.6
  validation_ratio: 0.2
  test_ratio: 0.2
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
