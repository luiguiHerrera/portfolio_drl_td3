"""Tests for model dataset preparation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.prepare_dataset import prepare_train_validation_test_datasets


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self):
        self.returns = pd.DataFrame(
            {
                "SPY": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02],
                "TLT": [0.00, 0.01, 0.01, -0.01, 0.02, 0.00],
                "GLD": [0.02, -0.01, 0.00, 0.01, -0.02, 0.02],
                "BTC-USD": [0.03, -0.02, 0.04, -0.01, 0.05, -0.03],
                "CASH": [0.0] * 6,
            },
            index=pd.date_range("2024-01-05", periods=6, freq="W-FRI"),
        )
        self.raw_features = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_b": [10.0, 12.0, 14.0, 16.0, 18.0],
            },
            index=self.returns.index[1:],
        )

    def test_prepare_dataset_returns_all_expected_keys(self):
        result = self._run_prepare_dataset()
        expected_keys = {
            "train_returns",
            "validation_returns",
            "test_returns",
            "train_features",
            "validation_features",
            "test_features",
            "feature_scaler",
        }

        self.assertEqual(set(result.keys()), expected_keys)

    def test_features_are_normalized_while_returns_are_not(self):
        result = self._run_prepare_dataset()

        expected_train_returns = self.returns.loc[self.raw_features.index[:3]]
        pd.testing.assert_frame_equal(result["train_returns"], expected_train_returns)
        self.assertFalse(result["train_features"].equals(self.raw_features.iloc[:3]))

    def test_returns_and_features_indexes_match_within_each_split(self):
        result = self._run_prepare_dataset()

        self.assertTrue(result["train_returns"].index.equals(result["train_features"].index))
        self.assertTrue(
            result["validation_returns"].index.equals(result["validation_features"].index)
        )
        self.assertTrue(result["test_returns"].index.equals(result["test_features"].index))

    def test_scaler_is_fit_from_train_features_only(self):
        result = self._run_prepare_dataset()
        expected_train_features = self.raw_features.iloc[:3]

        pd.testing.assert_series_equal(
            result["feature_scaler"]["mean"],
            expected_train_features.mean(),
        )
        pd.testing.assert_series_equal(
            result["feature_scaler"]["std"],
            expected_train_features.std().mask(expected_train_features.std() == 0.0, 1.0),
        )

    def test_chronological_split_order_is_preserved(self):
        result = self._run_prepare_dataset()

        self.assertTrue(result["train_features"].index.is_monotonic_increasing)
        self.assertTrue(result["validation_features"].index.is_monotonic_increasing)
        self.assertTrue(result["test_features"].index.is_monotonic_increasing)
        self.assertLess(result["train_features"].index[-1], result["validation_features"].index[0])
        self.assertLess(result["validation_features"].index[-1], result["test_features"].index[0])

    def test_cash_remains_in_returns_columns(self):
        result = self._run_prepare_dataset()

        self.assertIn("CASH", result["train_returns"].columns)
        self.assertIn("CASH", result["validation_returns"].columns)
        self.assertIn("CASH", result["test_returns"].columns)

    def _run_prepare_dataset(self):
        with self._temporary_config() as config_path:
            with patch(
                "src.data.prepare_dataset.build_returns_dataset",
                return_value=self.returns,
            ), patch(
                "src.data.prepare_dataset.build_features",
                return_value=self.raw_features,
            ):
                return prepare_train_validation_test_datasets(config_path)

    def _temporary_config(self):
        config_text = """
project:
  name: portfolio_drl_td3_test
  description: Temporary test config for dataset preparation

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
  train_ratio: 0.6
  validation_ratio: 0.2
  test_ratio: 0.2
"""
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "config.yaml"
        config_path.write_text(config_text, encoding="utf-8")

        class TemporaryConfig:
            def __enter__(self_inner):
                return str(config_path)

            def __exit__(self_inner, exc_type, exc_value, traceback):
                temp_dir.cleanup()

        return TemporaryConfig()


if __name__ == "__main__":
    unittest.main()
