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
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "feature_b": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            },
            index=self.returns.index,
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

        shifted_features = self.raw_features.shift(1).dropna()
        expected_train_returns = self.returns.loc[shifted_features.index[:3]]
        pd.testing.assert_frame_equal(result["train_returns"], expected_train_returns)
        self.assertFalse(result["train_features"].equals(shifted_features.iloc[:3]))

    def test_features_are_shifted_before_alignment_with_returns(self):
        result = self._run_prepare_dataset()
        shifted_features = self.raw_features.shift(1).dropna()
        aligned_returns = self.returns.loc[shifted_features.index]
        result_features = pd.concat(
            [
                result["train_features"],
                result["validation_features"],
                result["test_features"],
            ]
        )
        result_returns = pd.concat(
            [
                result["train_returns"],
                result["validation_returns"],
                result["test_returns"],
            ]
        )

        self.assertTrue(result_features.index.equals(aligned_returns.index))
        self.assertTrue(result_returns.index.equals(result_features.index))
        pd.testing.assert_series_equal(
            result["feature_scaler"]["mean"],
            shifted_features.iloc[:3].mean(),
        )

    def test_returns_and_features_indexes_match_within_each_split(self):
        result = self._run_prepare_dataset()

        self.assertTrue(result["train_returns"].index.equals(result["train_features"].index))
        self.assertTrue(
            result["validation_returns"].index.equals(result["validation_features"].index)
        )
        self.assertTrue(result["test_returns"].index.equals(result["test_features"].index))

    def test_scaler_is_fit_from_train_features_only(self):
        result = self._run_prepare_dataset()
        expected_train_features = self.raw_features.shift(1).dropna().iloc[:3]

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

    def test_prepare_dataset_uses_v2_features_when_configured(self):
        with self._temporary_config(
            """
features:
  version: v2
  market_asset: SPY
  short_window: 4
  long_window: 12
  ewma_span: 12
"""
        ) as config_path:
            with patch(
                "src.data.prepare_dataset.build_returns_dataset",
                return_value=self.returns,
            ), patch(
                "src.data.prepare_dataset.build_configured_features",
                return_value=self.raw_features,
            ) as build_configured_features_mock:
                result = prepare_train_validation_test_datasets(config_path)

        build_configured_features_mock.assert_called_once()
        called_returns, called_config = build_configured_features_mock.call_args.args
        pd.testing.assert_frame_equal(called_returns, self.returns)
        self.assertEqual(called_config["features"]["version"], "v2")
        self.assertTrue(result["train_returns"].index.equals(result["train_features"].index))

    def test_prepare_dataset_shifts_v2_raw_features_exactly_once_before_alignment(self):
        with self._temporary_config(
            """
features:
  version: v2
  market_asset: SPY
"""
        ) as config_path:
            with patch(
                "src.data.prepare_dataset.build_returns_dataset",
                return_value=self.returns,
            ), patch(
                "src.data.prepare_dataset.build_configured_features",
                return_value=self.raw_features,
            ), patch(
                "src.data.prepare_dataset.normalize_train_validation_test",
                side_effect=self._identity_normalize_train_validation_test,
            ):
                result = prepare_train_validation_test_datasets(config_path)

        features = pd.concat(
            [
                result["train_features"],
                result["validation_features"],
                result["test_features"],
            ]
        )
        returns = pd.concat(
            [
                result["train_returns"],
                result["validation_returns"],
                result["test_returns"],
            ]
        )
        aligned_date = self.returns.index[1]
        previous_date = self.returns.index[0]

        self.assertEqual(returns.loc[aligned_date, "SPY"], self.returns.loc[aligned_date, "SPY"])
        self.assertEqual(
            features.loc[aligned_date, "feature_a"],
            self.raw_features.loc[previous_date, "feature_a"],
        )
        self.assertNotEqual(
            features.loc[aligned_date, "feature_a"],
            self.raw_features.loc[aligned_date, "feature_a"],
        )

    def test_prepare_dataset_respects_config_end_date_from_return_builder(self):
        returns = self._returns_with_row_after_config_end_date()

        with self._temporary_config(end_date="2024-03-01") as config_path:
            with patch(
                "src.data.build_dataset.download_prices",
                return_value=pd.DataFrame(),
            ), patch(
                "src.data.build_dataset.compute_returns",
                return_value=returns,
            ), patch(
                "src.data.prepare_dataset.build_configured_features",
                side_effect=self._raw_features_for_returns,
            ):
                result = prepare_train_validation_test_datasets(config_path)

        end_date = pd.Timestamp("2024-03-01")
        for key in (
            "train_returns",
            "validation_returns",
            "test_returns",
            "train_features",
            "validation_features",
            "test_features",
        ):
            self.assertLessEqual(result[key].index.max(), end_date)

    def test_prepare_dataset_with_v2_respects_config_end_date_from_return_builder(self):
        returns = self._returns_with_row_after_config_end_date()

        with self._temporary_config(
            """
features:
  version: v2
  market_asset: SPY
""",
            end_date="2024-03-01",
        ) as config_path:
            with patch(
                "src.data.build_dataset.download_prices",
                return_value=pd.DataFrame(),
            ), patch(
                "src.data.build_dataset.compute_returns",
                return_value=returns,
            ), patch(
                "src.data.prepare_dataset.build_configured_features",
                side_effect=self._raw_features_for_returns,
            ) as build_configured_features_mock:
                result = prepare_train_validation_test_datasets(config_path)

        end_date = pd.Timestamp("2024-03-01")
        called_returns = build_configured_features_mock.call_args.args[0]
        self.assertLessEqual(called_returns.index.max(), end_date)
        self.assertLessEqual(result["test_returns"].index.max(), end_date)
        self.assertLessEqual(result["test_features"].index.max(), end_date)

    def _run_prepare_dataset(self):
        with self._temporary_config() as config_path:
            with patch(
                "src.data.prepare_dataset.build_returns_dataset",
                return_value=self.returns,
            ), patch(
                "src.data.prepare_dataset.build_configured_features",
                return_value=self.raw_features,
            ):
                return prepare_train_validation_test_datasets(config_path)

    def _temporary_config(self, features_section: str = "", end_date: str = "2024-01-01"):
        config_text = f"""
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
  end_date: {end_date}

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
""" + features_section
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "config.yaml"
        config_path.write_text(config_text, encoding="utf-8")

        class TemporaryConfig:
            def __enter__(self_inner):
                return str(config_path)

            def __exit__(self_inner, exc_type, exc_value, traceback):
                temp_dir.cleanup()

        return TemporaryConfig()

    @staticmethod
    def _identity_normalize_train_validation_test(
        train_features: pd.DataFrame,
        validation_features: pd.DataFrame,
        test_features: pd.DataFrame,
    ):
        scaler = {
            "mean": train_features.mean(),
            "std": train_features.std().mask(train_features.std() == 0.0, 1.0),
        }

        return train_features, validation_features, test_features, scaler

    @staticmethod
    def _returns_with_row_after_config_end_date() -> pd.DataFrame:
        index = pd.date_range("2024-01-05", periods=10, freq="W-FRI")

        return pd.DataFrame(
            {
                "SPY": [value / 100.0 for value in range(10)],
                "TLT": [value / 200.0 for value in range(10)],
                "GLD": [value / 300.0 for value in range(10)],
                "BTC-USD": [value / 50.0 for value in range(10)],
                "CASH": [0.0] * 10,
            },
            index=index,
        )

    @staticmethod
    def _raw_features_for_returns(returns: pd.DataFrame, config: dict) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature_a": range(len(returns)),
                "feature_b": range(10, 10 + len(returns)),
            },
            index=returns.index,
        )


if __name__ == "__main__":
    unittest.main()
