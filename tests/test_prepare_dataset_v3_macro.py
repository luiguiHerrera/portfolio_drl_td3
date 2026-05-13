"""Integration-style tests for configured Feature Set V3 macro CSV preparation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.features_v2 import build_features_v2
from src.data.prepare_dataset import prepare_train_validation_test_datasets


class PrepareDatasetV3MacroTests(unittest.TestCase):
    def test_prepare_dataset_v3_macro_path_produces_macro_features_without_missing_values(self):
        end_date = "2023-11-03"
        returns = self._returns_frame()

        with self._temporary_config(
            features_section=f"""
features:
  version: v3
  market_asset: SPY
  short_window: 4
  long_window: 12
  ewma_span: 12
  macro_path: {self._macro_fixture_path()}
  macro_date_column: date
""",
            end_date=end_date,
        ) as config_path:
            with patch(
                "src.data.build_dataset.download_prices",
                return_value=pd.DataFrame(),
            ), patch(
                "src.data.build_dataset.compute_returns",
                return_value=returns,
            ):
                datasets = prepare_train_validation_test_datasets(config_path)

        expected_macro_columns = {
            "macro_DGS10",
            "macro_DGS2",
            "macro_VIX",
            "macro_DXY",
            "macro_CPI",
            "macro_yield_curve_10y_2y",
            "macro_inverted_yield_curve_regime",
            "macro_high_vix_regime",
            "macro_dollar_momentum_12p",
            "macro_strong_dollar_regime",
            "macro_cpi_momentum_12p",
            "macro_inflation_pressure_regime",
        }
        for split_name in ("train_features", "validation_features", "test_features"):
            split_features = datasets[split_name]
            self.assertTrue(expected_macro_columns.issubset(set(split_features.columns)))
            self.assertFalse(split_features.isna().any().any())

        for split_name in ("train_returns", "validation_returns", "test_returns"):
            self.assertLessEqual(datasets[split_name].index.max(), pd.Timestamp(end_date))

    def test_prepare_dataset_v3_long_macro_fixture_has_more_features_than_v2(self):
        end_date = "2024-12-31"
        returns = self._returns_frame_2020_2024()

        with self._temporary_config(
            features_section=f"""
features:
  version: v3
  market_asset: SPY
  short_window: 4
  long_window: 12
  ewma_span: 12
  macro_path: {self._long_macro_fixture_path()}
  macro_date_column: date
""",
            start_date="2020-01-01",
            end_date=end_date,
        ) as config_path:
            with patch(
                "src.data.build_dataset.download_prices",
                return_value=pd.DataFrame(),
            ), patch(
                "src.data.build_dataset.compute_returns",
                return_value=returns,
            ):
                datasets = prepare_train_validation_test_datasets(config_path)

        expected_macro_columns = {
            "macro_DGS10",
            "macro_DGS2",
            "macro_VIX",
            "macro_DXY",
            "macro_CPI",
            "macro_yield_curve_10y_2y",
            "macro_inverted_yield_curve_regime",
            "macro_high_vix_regime",
            "macro_dollar_momentum_12p",
            "macro_strong_dollar_regime",
            "macro_cpi_momentum_12p",
            "macro_inflation_pressure_regime",
        }
        for split_name in ("train_features", "validation_features", "test_features"):
            split_features = datasets[split_name]
            self.assertTrue(expected_macro_columns.issubset(set(split_features.columns)))
            self.assertFalse(split_features.isna().any().any())

        self.assertLessEqual(datasets["test_returns"].index.max(), pd.Timestamp(end_date))
        v2_feature_count = build_features_v2(returns).shape[1]
        self.assertGreater(datasets["train_features"].shape[1], v2_feature_count)

    def test_prepare_dataset_without_features_section_uses_v1_default(self):
        returns = self._small_returns_frame()
        raw_features = self._raw_features_for_returns(returns)

        with self._temporary_config() as config_path:
            with patch(
                "src.data.prepare_dataset.build_returns_dataset",
                return_value=returns,
            ), patch(
                "src.data.feature_factory.build_features",
                return_value=raw_features,
            ) as build_features_mock, patch(
                "src.data.feature_factory.build_features_v2",
            ) as build_features_v2_mock, patch(
                "src.data.feature_factory.build_features_v3",
            ) as build_features_v3_mock:
                prepare_train_validation_test_datasets(config_path)

        build_features_mock.assert_called_once_with(returns)
        build_features_v2_mock.assert_not_called()
        build_features_v3_mock.assert_not_called()

    def test_prepare_dataset_v2_dispatch_is_unchanged(self):
        returns = self._small_returns_frame()
        raw_features = self._raw_features_for_returns(returns)

        with self._temporary_config(
            features_section="""
features:
  version: v2
  market_asset: SPY
"""
        ) as config_path:
            with patch(
                "src.data.prepare_dataset.build_returns_dataset",
                return_value=returns,
            ), patch(
                "src.data.feature_factory.build_features",
            ) as build_features_mock, patch(
                "src.data.feature_factory.build_features_v2",
                return_value=raw_features,
            ) as build_features_v2_mock, patch(
                "src.data.feature_factory.build_features_v3",
            ) as build_features_v3_mock:
                prepare_train_validation_test_datasets(config_path)

        build_features_mock.assert_not_called()
        build_features_v2_mock.assert_called_once()
        self.assertIs(build_features_v2_mock.call_args.args[0], returns)
        build_features_v3_mock.assert_not_called()

    @staticmethod
    def _macro_fixture_path() -> str:
        return str(Path(__file__).parent / "fixtures" / "macro_weekly_test.csv")

    @staticmethod
    def _long_macro_fixture_path() -> str:
        return str(Path(__file__).parent / "fixtures" / "macro_weekly_2020_2024_test.csv")

    @staticmethod
    def _returns_frame() -> pd.DataFrame:
        index = pd.date_range("2023-01-06", periods=45, freq="W-FRI")
        pattern = [0.010, 0.015, -0.008, 0.012, -0.020, 0.018, 0.006, -0.011, 0.009]

        return pd.DataFrame(
            {
                "SPY": (pattern * 5)[: len(index)],
                "TLT": ([0.002, 0.004, 0.001, -0.003, 0.006] * 9)[: len(index)],
                "GLD": ([0.004, -0.002, 0.006, 0.001, 0.012] * 9)[: len(index)],
                "BTC-USD": ([0.040, -0.030, 0.050, -0.020, 0.060] * 9)[: len(index)],
                "CASH": [0.0] * len(index),
            },
            index=index,
        )

    @staticmethod
    def _small_returns_frame() -> pd.DataFrame:
        index = pd.date_range("2024-01-05", periods=6, freq="W-FRI")

        return pd.DataFrame(
            {
                "SPY": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02],
                "TLT": [0.00, 0.01, 0.01, -0.01, 0.02, 0.00],
                "GLD": [0.02, -0.01, 0.00, 0.01, -0.02, 0.02],
                "BTC-USD": [0.03, -0.02, 0.04, -0.01, 0.05, -0.03],
                "CASH": [0.0] * 6,
            },
            index=index,
        )

    @staticmethod
    def _returns_frame_2020_2024() -> pd.DataFrame:
        index = pd.date_range("2020-01-03", "2024-12-27", freq="W-FRI")
        pattern = [0.010, 0.015, -0.008, 0.012, -0.020, 0.018, 0.006, -0.011, 0.009]

        return pd.DataFrame(
            {
                "SPY": (pattern * 29)[: len(index)],
                "TLT": ([0.002, 0.004, 0.001, -0.003, 0.006] * 53)[: len(index)],
                "GLD": ([0.004, -0.002, 0.006, 0.001, 0.012] * 53)[: len(index)],
                "BTC-USD": ([0.040, -0.030, 0.050, -0.020, 0.060] * 53)[: len(index)],
                "CASH": [0.0] * len(index),
            },
            index=index,
        )

    @staticmethod
    def _raw_features_for_returns(returns: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature_a": range(len(returns)),
                "feature_b": range(10, 10 + len(returns)),
            },
            index=returns.index,
        )

    def _temporary_config(
        self,
        features_section: str = "",
        start_date: str = "2023-01-01",
        end_date: str = "2024-03-01",
    ):
        config_text = f"""
project:
  name: portfolio_drl_td3_prepare_v3_macro_test

data:
  assets:
    - SPY
    - TLT
    - GLD
    - BTC-USD
    - CASH
  frequency: weekly
  start_date: {start_date}
  end_date: {end_date}

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
""" + features_section
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "config.yaml"
        config_path.write_text(config_text, encoding="utf-8")

        class TemporaryConfig:
            def __enter__(self_inner):
                return str(config_path)

            def __exit__(self_inner, exc_type, exc_value, traceback):
                temp_dir.cleanup()
                return False

        return TemporaryConfig()


if __name__ == "__main__":
    unittest.main()
