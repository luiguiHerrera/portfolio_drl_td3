"""Tests for the minimal returns dataset builder."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.build_dataset import build_returns_dataset


class BuildDatasetTests(unittest.TestCase):
    def setUp(self):
        self.assets = ["SPY", "TLT", "GLD", "BTC-USD", "CASH"]
        self.frequency = "weekly"
        self.start_date = "2020-01-01"
        self.end_date = "2024-01-01"
        self.prices = pd.DataFrame(
            {
                "SPY": [100.0, 101.0],
                "TLT": [90.0, 91.0],
                "GLD": [180.0, 181.0],
                "BTC-USD": [30000.0, 30100.0],
            },
            index=pd.date_range("2020-01-03", periods=2, freq="W-FRI"),
        )
        self.returns = pd.DataFrame(
            {
                "SPY": [0.01],
                "TLT": [0.01],
                "GLD": [0.01],
                "BTC-USD": [0.01],
                "CASH": [0.0],
            },
            index=pd.date_range("2020-01-10", periods=1, freq="W-FRI"),
        )

    def test_build_returns_dataset_returns_dataframe(self):
        with self._temporary_config() as config_path:
            with self._patched_pipeline():
                result = build_returns_dataset(config_path)

        self.assertIsInstance(result, pd.DataFrame)

    def test_build_returns_dataset_calls_download_prices_with_config_values(self):
        with self._temporary_config() as config_path:
            with self._patched_pipeline() as mocks:
                build_returns_dataset(config_path)

        mocks["download_prices"].assert_called_once_with(
            self.assets,
            self.start_date,
            self.end_date,
            extra_assets=[],
        )

    def test_build_returns_dataset_calls_compute_returns_with_config_values(self):
        with self._temporary_config() as config_path:
            with self._patched_pipeline() as mocks:
                build_returns_dataset(config_path)

        mocks["compute_returns"].assert_called_once_with(
            self.prices,
            self.assets,
            self.frequency,
            cash_return_model="zero",
            cash_proxy_asset=None,
        )

    def test_build_returns_dataset_requests_bil_when_cash_model_is_bil_proxy(self):
        with self._temporary_config(
            extra_data_lines="""
  cash_return_model: bil_proxy
"""
        ) as config_path:
            with self._patched_pipeline() as mocks:
                build_returns_dataset(config_path)

        mocks["download_prices"].assert_called_once_with(
            self.assets,
            self.start_date,
            self.end_date,
            extra_assets=["BIL"],
        )
        mocks["compute_returns"].assert_called_once_with(
            self.prices,
            self.assets,
            self.frequency,
            cash_return_model="bil_proxy",
            cash_proxy_asset=None,
        )

    def test_build_returns_dataset_preserves_cash_in_final_dataframe(self):
        with self._temporary_config() as config_path:
            with self._patched_pipeline():
                result = build_returns_dataset(config_path)

        self.assertIn("CASH", result.columns)

    def test_build_returns_dataset_applies_config_date_boundaries_after_returns(self):
        out_of_bounds_returns = pd.DataFrame(
            {
                "SPY": [0.01, 0.02, 0.03],
                "TLT": [0.01, 0.02, 0.03],
                "GLD": [0.01, 0.02, 0.03],
                "BTC-USD": [0.01, 0.02, 0.03],
                "CASH": [0.0, 0.0, 0.0],
            },
            index=pd.to_datetime(["2019-12-27", "2020-01-03", "2024-01-05"]),
        )

        with self._temporary_config() as config_path:
            with self._patched_pipeline(returns=out_of_bounds_returns):
                result = build_returns_dataset(config_path)

        self.assertGreaterEqual(result.index.min(), pd.Timestamp(self.start_date))
        self.assertLessEqual(result.index.max(), pd.Timestamp(self.end_date))
        self.assertNotIn(pd.Timestamp("2019-12-27"), result.index)
        self.assertNotIn(pd.Timestamp("2024-01-05"), result.index)

    def test_build_returns_dataset_uses_configured_returns_snapshot_without_download(self):
        snapshot = self.returns.reset_index(names="date")

        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = Path(temp_dir) / "returns.csv"
            config_path = Path(temp_dir) / "config.yaml"
            snapshot.to_csv(returns_path, index=False)
            config_path.write_text(
                self._config_text(
                    extra_data_lines=f"""
  returns_path: {returns_path}
  returns_date_column: date
"""
                ),
                encoding="utf-8",
            )
            with patch("src.data.build_dataset.download_prices") as download_mock:
                result = build_returns_dataset(str(config_path))

        download_mock.assert_not_called()
        pd.testing.assert_frame_equal(result, self.returns, check_freq=False)

    def test_configured_returns_snapshot_missing_path_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_returns_path = Path(temp_dir) / "missing_returns.csv"
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                self._config_text(
                    extra_data_lines=f"""
  returns_path: {missing_returns_path}
  returns_date_column: date
"""
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "Returns snapshot not found"):
                build_returns_dataset(str(config_path))

    def _temporary_config(self, extra_data_lines: str = ""):
        config_text = self._config_text(extra_data_lines=extra_data_lines)
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "config.yaml"
        config_path.write_text(config_text, encoding="utf-8")

        class TemporaryConfig:
            def __enter__(self_inner):
                return str(config_path)

            def __exit__(self_inner, exc_type, exc_value, traceback):
                temp_dir.cleanup()

        return TemporaryConfig()

    def _config_text(self, extra_data_lines: str = ""):
        return f"""
project:
  name: portfolio_drl_td3_test
  description: Temporary test config for dataset builder

data:
  assets:
    - SPY
    - TLT
    - GLD
    - BTC-USD
    - CASH
  frequency: {self.frequency}
  start_date: {self.start_date}
  end_date: {self.end_date}
{extra_data_lines.rstrip()}
environment:
  initial_cash: 100000
  transaction_cost: 0.001
  allow_short: false
  max_weight_per_asset: 1.0

reward:
  lambda_return: 1.0
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
"""

    def _patched_pipeline(self, returns: pd.DataFrame | None = None):
        if returns is None:
            returns = self.returns

        download_patcher = patch(
            "src.data.build_dataset.download_prices",
            return_value=self.prices,
        )
        compute_patcher = patch(
            "src.data.build_dataset.compute_returns",
            return_value=returns,
        )

        class PatchedPipeline:
            def __enter__(self_inner):
                download_mock = download_patcher.start()
                compute_mock = compute_patcher.start()
                return {
                    "download_prices": download_mock,
                    "compute_returns": compute_mock,
                }

            def __exit__(self_inner, exc_type, exc_value, traceback):
                compute_patcher.stop()
                download_patcher.stop()

        return PatchedPipeline()


if __name__ == "__main__":
    unittest.main()
