"""Tests for Feature Set V6 financial state construction."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.feature_factory import build_configured_features
from src.data.features_v6 import SCORE_COLUMNS, build_features_v6
from src.data.prepare_dataset import prepare_train_validation_test_datasets


class FeatureSetV6Tests(unittest.TestCase):
    def setUp(self):
        self.returns = self._realistic_returns()

    def test_v6_feature_builder_returns_dataframe(self):
        features = build_features_v6(self.returns)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertFalse(features.empty)

    def test_v6_output_index_matches_input_index_after_rolling_calculations(self):
        features = build_features_v6(self.returns)
        expected_index = self.returns.loc[features.index[0] :].index

        self.assertTrue(features.index.equals(expected_index))

    def test_v6_has_no_cash_momentum_columns(self):
        features = build_features_v6(self.returns)

        self.assertFalse(
            any(
                column.startswith("CASH_")
                and (
                    "_ret_" in column
                    or "momentum" in column
                    or "trend_strength" in column
                    or "winner" in column
                )
                for column in features.columns
            )
        )

    def test_v6_score_columns_are_within_zero_one(self):
        features = build_features_v6(self.returns)
        score_columns = [column for column in SCORE_COLUMNS if column in features]

        self.assertTrue(score_columns)
        for column in score_columns:
            self.assertGreaterEqual(features[column].min(), 0.0)
            self.assertLessEqual(features[column].max(), 1.0)

    def test_v6_has_no_infinite_values(self):
        features = build_features_v6(self.returns)

        self.assertFalse(np.isinf(features.to_numpy()).any())

    def test_v6_does_not_mutate_input_returns(self):
        returns = self.returns.copy(deep=True)

        build_features_v6(returns)

        pd.testing.assert_frame_equal(returns, self.returns)

    def test_v6_includes_momentum_and_trend_columns(self):
        features = build_features_v6(self.returns)
        expected_columns = {
            "SPY_ret_4w",
            "SPY_ret_12w",
            "SPY_ret_26w",
            "SPY_ewma_ret_12w",
            "SPY_trend_strength_12w",
            "SPY_trend_strength_26w",
            "SPY_momentum_rank_12w",
            "SPY_risk_adjusted_momentum_rank_12w",
            "SPY_winner_12w_one_hot",
            "SPY_winner_risk_adjusted_12w_one_hot",
        }

        self.assertTrue(expected_columns.issubset(features.columns))

    def test_v6_includes_risk_regime_score_columns(self):
        features = build_features_v6(self.returns)
        expected_columns = {
            "market_trend_positive_score",
            "market_drawdown_stress_score",
            "market_high_vol_score",
            "correlation_stress_score",
            "risk_off_score",
        }

        self.assertTrue(expected_columns.issubset(features.columns))
        self.assertFalse(
            any(
                "probability" in column.lower()
                or "prob_" in column.lower()
                or "_prob" in column.lower()
                or column.startswith("p_")
                for column in features.columns
            )
        )

    def test_v6_score_values_match_legacy_heuristic_formula(self):
        features = build_features_v6(self.returns)
        expected_risk_off = (
            0.35 * (1.0 - features["market_trend_positive_score"])
            + 0.25 * features["market_drawdown_stress_score"]
            + 0.25 * features["market_high_vol_score"]
            + 0.15 * features["correlation_stress_score"]
        )
        expected_cash_permission = (
            0.65 * features["risk_off_score"]
            + 0.35 * features["market_drawdown_stress_score"]
        )

        pd.testing.assert_series_equal(
            features["risk_off_score"],
            expected_risk_off.rename("risk_off_score"),
        )
        pd.testing.assert_series_equal(
            features["cash_permission_score"],
            expected_cash_permission.rename("cash_permission_score"),
        )

    def test_v6_includes_volatility_proxy_columns(self):
        features = build_features_v6(self.returns)
        expected_columns = {
            "SPY_ewma_vol_4w",
            "SPY_ewma_vol_12w",
            "SPY_realized_vol_12w",
            "SPY_vol_ratio_4w_12w",
        }

        self.assertTrue(expected_columns.issubset(features.columns))

    def test_v6_includes_defensive_attractiveness_columns(self):
        features = build_features_v6(self.returns)
        expected_columns = {
            "GLD_vs_SPY_momentum_12w",
            "TLT_vs_SPY_momentum_12w",
            "GLD_vs_SPY_risk_adjusted_momentum_12w",
            "TLT_vs_SPY_risk_adjusted_momentum_12w",
            "defensive_asset_score_GLD",
            "defensive_asset_score_TLT",
            "cash_permission_score",
        }

        self.assertTrue(expected_columns.issubset(features.columns))

    def test_feature_factory_supports_v6(self):
        expected = pd.DataFrame(
            {"v6_feature": [1.0, 2.0]},
            index=self.returns.index[:2],
        )
        config = {
            "features": {
                "version": "v6",
                "market_asset": "SPY",
                "short_window": 3,
                "medium_window": 9,
                "long_window": 18,
                "ewma_short_span": 5,
                "ewma_long_span": 11,
                "correlation_window": 10,
                "zscore_window": 40,
            }
        }

        with patch(
            "src.data.feature_factory.build_features_v6",
            return_value=expected,
        ) as build_features_v6_mock:
            result = build_configured_features(self.returns, config)

        build_features_v6_mock.assert_called_once_with(
            self.returns,
            market_asset="SPY",
            short_window=3,
            medium_window=9,
            long_window=18,
            ewma_short_span=5,
            ewma_long_span=11,
            correlation_window=10,
            zscore_window=40,
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_prepare_dataset_works_with_v6_features(self):
        with self._temporary_config() as config_path:
            with patch(
                "src.data.prepare_dataset.build_returns_dataset",
                return_value=self.returns,
            ):
                datasets = prepare_train_validation_test_datasets(str(config_path))

        for split_name in ("train_features", "validation_features", "test_features"):
            split_features = datasets[split_name]
            self.assertFalse(split_features.empty)
            self.assertIn("risk_off_score", split_features.columns)
            self.assertIn("GLD_vs_SPY_momentum_12w", split_features.columns)
            self.assertFalse(split_features.isna().any().any())

    def test_v6_raises_value_error_for_invalid_windows(self):
        invalid_arguments = (
            {"short_window": 1},
            {"medium_window": 1},
            {"long_window": 1},
            {"ewma_short_span": 1},
            {"ewma_long_span": 1},
            {"correlation_window": 1},
            {"zscore_window": 1},
            {"short_window": 12, "medium_window": 4},
            {"medium_window": 26, "long_window": 12},
        )

        for kwargs in invalid_arguments:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    build_features_v6(self.returns, **kwargs)

    @staticmethod
    def _realistic_returns() -> pd.DataFrame:
        index = pd.date_range("2021-01-01", periods=140, freq="W-FRI")
        step = np.arange(len(index))

        return pd.DataFrame(
            {
                "SPY": 0.004 + 0.018 * np.sin(step / 7.0) - 0.010 * (step % 17 == 0),
                "TLT": 0.001 - 0.010 * np.sin(step / 9.0) + 0.006 * (step % 23 == 0),
                "GLD": 0.002 + 0.012 * np.cos(step / 8.0) + 0.004 * (step % 19 == 0),
                "BTC-USD": 0.006 + 0.055 * np.sin(step / 5.0) - 0.030 * (step % 13 == 0),
                "CASH": np.zeros(len(index)),
            },
            index=index,
        )

    @staticmethod
    def _temporary_config():
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "v6_config.yaml"
        config_path.write_text(
            """
project:
  name: v6_test
data:
  assets: [SPY, TLT, GLD, BTC-USD, CASH]
  frequency: weekly
environment:
  initial_cash: 100000
  transaction_cost: 0.001
reward:
  lambda_return: 1.0
td3:
  actor_learning_rate: 0.0005
  critic_learning_rate: 0.0005
  gamma: 0.99
  tau: 0.005
  policy_noise: 0.2
  noise_clip: 0.5
  policy_delay: 2
  batch_size: 32
  replay_buffer_size: 1000
training:
  seed: 42
  train_ratio: 0.6
  validation_ratio: 0.2
  test_ratio: 0.2
  episodes: 1
features:
  version: v6
  market_asset: SPY
  short_window: 4
  medium_window: 12
  long_window: 26
  ewma_short_span: 4
  ewma_long_span: 12
  correlation_window: 12
  zscore_window: 52
""",
            encoding="utf-8",
        )

        class TemporaryConfig:
            def __enter__(self_inner):
                return config_path

            def __exit__(self_inner, exc_type, exc_value, traceback):
                temp_dir.cleanup()

        return TemporaryConfig()


if __name__ == "__main__":
    unittest.main()
