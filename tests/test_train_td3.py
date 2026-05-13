"""Tests for the minimal TD3 training loop."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.train.train_td3 import train_td3


class TrainTD3Tests(unittest.TestCase):
    def setUp(self):
        self.train_returns = self._returns_frame(periods=4, start="2024-01-05")
        self.validation_returns = self._returns_frame(periods=2, start="2024-02-02")
        self.test_returns = self._returns_frame(periods=2, start="2024-02-16")
        self.train_features = self._features_frame(self.train_returns.index)
        self.validation_features = self._features_frame(self.validation_returns.index)
        self.test_features = self._features_frame(self.test_returns.index)
        self.datasets = {
            "train_returns": self.train_returns,
            "validation_returns": self.validation_returns,
            "test_returns": self.test_returns,
            "train_features": self.train_features,
            "validation_features": self.validation_features,
            "test_features": self.test_features,
            "feature_scaler": {
                "mean": self.train_features.mean(),
                "std": self.train_features.std(),
            },
        }

    def test_train_td3_returns_expected_keys(self):
        result, _ = self._run_train_td3()
        expected_keys = {
            "agent",
            "replay_buffer",
            "episode_logs",
            "train_returns",
            "train_features",
            "validation_returns",
            "validation_features",
            "test_returns",
            "test_features",
            "validation_evaluation",
            "test_evaluation",
            "validation_comparison",
            "test_comparison",
        }

        self.assertEqual(set(result.keys()), expected_keys)

    def test_episode_logs_length_equals_configured_episodes(self):
        result, _ = self._run_train_td3()

        self.assertEqual(len(result["episode_logs"]), 2)

    def test_replay_buffer_contains_transitions_after_training(self):
        result, _ = self._run_train_td3()

        self.assertGreater(len(result["replay_buffer"]), 0)

    def test_each_episode_log_contains_required_summary_fields(self):
        result, _ = self._run_train_td3()

        for episode_log in result["episode_logs"]:
            self.assertIn("final_portfolio_value", episode_log)
            self.assertIn("total_reward", episode_log)
            self.assertIn("steps", episode_log)
            self.assertIn("critic_1_loss", episode_log)
            self.assertIn("critic_2_loss", episode_log)
            self.assertIn("actor_loss", episode_log)
            self.assertIn("average_turnover", episode_log)
            self.assertIn("average_transaction_cost", episode_log)
            self.assertIn("final_weights", episode_log)
            self.assertIn("max_weight", episode_log)
            self.assertIn("cash_weight", episode_log)

    def test_episode_log_weight_diagnostics_are_well_formed(self):
        result, _ = self._run_train_td3()
        expected_assets = set(self.train_returns.columns)

        for episode_log in result["episode_logs"]:
            self.assertIsInstance(episode_log["final_weights"], dict)
            self.assertEqual(set(episode_log["final_weights"].keys()), expected_assets)
            self.assertGreaterEqual(episode_log["max_weight"], 0.0)
            self.assertLessEqual(episode_log["max_weight"], 1.0)
            self.assertGreaterEqual(episode_log["cash_weight"], 0.0)
            self.assertLessEqual(episode_log["cash_weight"], 1.0)

    def test_validation_evaluation_contains_episode_and_metrics(self):
        result, _ = self._run_train_td3()

        self.assertEqual(
            set(result["validation_evaluation"].keys()),
            {"episode", "metrics", "diagnostics", "policy_history"},
        )

    def test_test_evaluation_contains_episode_and_metrics(self):
        result, _ = self._run_train_td3()

        self.assertEqual(
            set(result["test_evaluation"].keys()),
            {"episode", "metrics", "diagnostics", "policy_history"},
        )

    def test_validation_and_test_metrics_contain_expected_keys(self):
        result, _ = self._run_train_td3()
        expected_metric_keys = {
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        }

        self.assertEqual(
            set(result["validation_evaluation"]["metrics"].keys()),
            expected_metric_keys,
        )
        self.assertEqual(
            set(result["test_evaluation"]["metrics"].keys()),
            expected_metric_keys,
        )

    def test_validation_and_test_comparisons_have_expected_top_level_keys(self):
        result, _ = self._run_train_td3()
        expected_keys = {"agent", "benchmarks", "metrics_table"}

        self.assertEqual(set(result["validation_comparison"].keys()), expected_keys)
        self.assertEqual(set(result["test_comparison"].keys()), expected_keys)

    def test_validation_and_test_comparisons_include_basic_benchmarks(self):
        result, _ = self._run_train_td3()
        expected_benchmarks = {
            "equal_weight_gross",
            "equal_weight_rebalanced_net",
            "buy_and_hold",
            "buy_hold_SPY",
            "buy_hold_TLT",
            "buy_hold_GLD",
            "buy_hold_BTC-USD",
            "buy_hold_CASH",
        }

        self.assertEqual(
            set(result["validation_comparison"]["benchmarks"].keys()),
            expected_benchmarks,
        )
        self.assertEqual(
            set(result["test_comparison"]["benchmarks"].keys()),
            expected_benchmarks,
        )

    def test_validation_and_test_metrics_tables_include_policy_rows(self):
        result, _ = self._run_train_td3()
        expected_rows = {
            "agent",
            "equal_weight_gross",
            "equal_weight_rebalanced_net",
            "buy_and_hold",
            "buy_hold_SPY",
            "buy_hold_TLT",
            "buy_hold_GLD",
            "buy_hold_BTC-USD",
            "buy_hold_CASH",
        }

        self.assertEqual(
            set(result["validation_comparison"]["metrics_table"].index),
            expected_rows,
        )
        self.assertEqual(
            set(result["test_comparison"]["metrics_table"].index),
            expected_rows,
        )

    def test_no_files_are_written(self):
        _, temp_dir = self._run_train_td3()

        self.assertEqual(
            sorted(path.name for path in temp_dir.iterdir()),
            ["config.yaml"],
        )

    def _run_train_td3(self):
        with self._temporary_config() as (config_path, temp_dir):
            with patch(
                "src.train.train_td3.prepare_train_validation_test_datasets",
                return_value=self.datasets,
            ):
                result = train_td3(config_path)

        return result, temp_dir

    def _temporary_config(self):
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)
        config_path = temp_path / "config.yaml"
        config_path.write_text(
            """
project:
  name: portfolio_drl_td3_test
  description: Temporary test config for TD3 training loop

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
  batch_size: 2
  replay_buffer_size: 50

training:
  seed: 42
  episodes: 2
  train_ratio: 0.7
  validation_ratio: 0.15
  test_ratio: 0.15
""",
            encoding="utf-8",
        )

        class TemporaryConfig:
            def __enter__(self_inner):
                return str(config_path), temp_path

            def __exit__(self_inner, exc_type, exc_value, traceback):
                return False

        self.addCleanup(temp_dir.cleanup)
        return TemporaryConfig()

    @staticmethod
    def _returns_frame(periods: int, start: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "SPY": [0.01, 0.02, -0.01, 0.00][:periods],
                "TLT": [0.00, 0.01, 0.01, -0.01][:periods],
                "GLD": [0.02, -0.01, 0.00, 0.01][:periods],
                "BTC-USD": [0.03, -0.02, 0.04, -0.01][:periods],
                "CASH": [0.00, 0.00, 0.00, 0.00][:periods],
            },
            index=pd.date_range(start, periods=periods, freq="W-FRI"),
        )

    @staticmethod
    def _features_frame(index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature_a": [float(value) for value in range(len(index))],
                "feature_b": [float(value + 1) for value in range(len(index))],
                "feature_c": [0.5 for _ in range(len(index))],
            },
            index=index,
        )


if __name__ == "__main__":
    unittest.main()
