"""Tests for explicit walk-forward dataset slicing."""

import unittest
from unittest.mock import patch

import pandas as pd

from src.data.walk_forward_split import build_walk_forward_datasets, slice_dataset_by_date


class WalkForwardSplitTests(unittest.TestCase):
    def test_slice_dataset_by_date_selects_inclusive_date_range(self):
        returns, features = self._returns_and_features()

        sliced_returns, sliced_features = slice_dataset_by_date(
            returns,
            features,
            "2024-01-03",
            "2024-01-05",
        )

        self.assertEqual(list(sliced_returns.index), list(returns.index[2:5]))
        self.assertTrue(sliced_returns.index.equals(sliced_features.index))

    def test_slice_dataset_by_date_empty_slice_raises_value_error(self):
        returns, features = self._returns_and_features()

        with self.assertRaises(ValueError):
            slice_dataset_by_date(returns, features, "2025-01-01", "2025-01-31")

    def test_slice_dataset_by_date_returns_and_features_indexes_align(self):
        returns, features = self._returns_and_features()
        features = features.drop(features.index[3])

        sliced_returns, sliced_features = slice_dataset_by_date(
            returns,
            features,
            "2024-01-02",
            "2024-01-06",
        )

        self.assertTrue(sliced_returns.index.equals(sliced_features.index))
        self.assertNotIn(pd.Timestamp("2024-01-04"), sliced_returns.index)

    def test_build_walk_forward_datasets_shifts_features_before_alignment(self):
        returns, raw_features = self._mock_builder_inputs()
        fold = self._fold()

        with self._patched_builders(returns, raw_features):
            datasets = build_walk_forward_datasets("config.yaml", fold)

        shifted_features = raw_features.shift(1).dropna()
        expected_train_features = shifted_features.loc[
            datasets["train_features"].index,
            datasets["train_features"].columns,
        ]

        pd.testing.assert_series_equal(
            datasets["feature_scaler"]["mean"],
            expected_train_features.mean(),
        )

    def test_build_walk_forward_datasets_fits_scaler_only_on_train(self):
        returns, raw_features = self._mock_builder_inputs()
        fold = self._fold()

        with self._patched_builders(returns, raw_features):
            datasets = build_walk_forward_datasets("config.yaml", fold)

        shifted_features = raw_features.shift(1).dropna()
        train_features = shifted_features.loc[datasets["train_features"].index]
        all_features = shifted_features.loc[
            datasets["train_features"].index
            .append(datasets["validation_features"].index)
            .append(datasets["test_features"].index)
        ]

        pd.testing.assert_series_equal(datasets["feature_scaler"]["mean"], train_features.mean())
        self.assertFalse(datasets["feature_scaler"]["mean"].equals(all_features.mean()))

    def test_build_walk_forward_datasets_indexes_are_chronological_and_non_overlapping(self):
        returns, raw_features = self._mock_builder_inputs()
        fold = self._fold()

        with self._patched_builders(returns, raw_features):
            datasets = build_walk_forward_datasets("config.yaml", fold)

        self.assertLess(datasets["train_returns"].index.max(), datasets["validation_returns"].index.min())
        self.assertLess(datasets["validation_returns"].index.max(), datasets["test_returns"].index.min())
        self.assertTrue(datasets["train_returns"].index.is_monotonic_increasing)
        self.assertTrue(datasets["validation_returns"].index.is_monotonic_increasing)
        self.assertTrue(datasets["test_returns"].index.is_monotonic_increasing)

    def test_build_walk_forward_datasets_respects_explicit_fold_dates(self):
        returns, raw_features = self._mock_builder_inputs()
        fold = self._fold()

        with self._patched_builders(returns, raw_features):
            datasets = build_walk_forward_datasets("config.yaml", fold)

        self.assertGreaterEqual(
            datasets["train_returns"].index.min(),
            pd.Timestamp(fold["train_start"]),
        )
        self.assertLessEqual(
            datasets["train_returns"].index.max(),
            pd.Timestamp(fold["train_end"]),
        )
        self.assertGreaterEqual(
            datasets["validation_returns"].index.min(),
            pd.Timestamp(fold["validation_start"]),
        )
        self.assertLessEqual(
            datasets["validation_returns"].index.max(),
            pd.Timestamp(fold["validation_end"]),
        )
        self.assertGreaterEqual(
            datasets["test_returns"].index.min(),
            pd.Timestamp(fold["test_start"]),
        )
        self.assertLessEqual(
            datasets["test_returns"].index.max(),
            pd.Timestamp(fold["test_end"]),
        )

    def test_build_walk_forward_datasets_uses_v2_features_when_configured(self):
        returns, raw_features = self._mock_builder_inputs()
        fold = self._fold()
        config = {
            "features": {
                "version": "v2",
                "market_asset": "SPY",
                "short_window": 4,
                "long_window": 12,
                "ewma_span": 12,
            }
        }
        build_configured_features_mock = unittest.mock.Mock(return_value=raw_features)

        with patch.multiple(
            "src.data.walk_forward_split",
            load_config=unittest.mock.Mock(return_value=config),
            build_returns_dataset=unittest.mock.Mock(return_value=returns),
            build_configured_features=build_configured_features_mock,
        ):
            datasets = build_walk_forward_datasets("config.yaml", fold)

        build_configured_features_mock.assert_called_once_with(returns, config)
        self.assertTrue(datasets["train_returns"].index.equals(datasets["train_features"].index))

    def test_build_walk_forward_datasets_shifts_v2_raw_features_exactly_once_before_alignment(self):
        returns, raw_features = self._mock_builder_inputs()
        fold = self._fold()
        config = {
            "features": {
                "version": "v2",
                "market_asset": "SPY",
            }
        }

        with patch.multiple(
            "src.data.walk_forward_split",
            load_config=unittest.mock.Mock(return_value=config),
            build_returns_dataset=unittest.mock.Mock(return_value=returns),
            build_configured_features=unittest.mock.Mock(return_value=raw_features),
            normalize_train_validation_test=unittest.mock.Mock(
                side_effect=self._identity_normalize_train_validation_test
            ),
        ):
            datasets = build_walk_forward_datasets("config.yaml", fold)

        features = pd.concat(
            [
                datasets["train_features"],
                datasets["validation_features"],
                datasets["test_features"],
            ]
        )
        returns_by_split = pd.concat(
            [
                datasets["train_returns"],
                datasets["validation_returns"],
                datasets["test_returns"],
            ]
        )
        aligned_date = pd.Timestamp("2024-01-03")
        previous_date = pd.Timestamp("2024-01-02")

        self.assertEqual(
            returns_by_split.loc[aligned_date, "SPY"],
            returns.loc[aligned_date, "SPY"],
        )
        self.assertEqual(
            features.loc[aligned_date, "SPY_ret_1w"],
            raw_features.loc[previous_date, "SPY_ret_1w"],
        )
        self.assertNotEqual(
            features.loc[aligned_date, "SPY_ret_1w"],
            raw_features.loc[aligned_date, "SPY_ret_1w"],
        )

    def _patched_builders(
        self,
        returns: pd.DataFrame,
        raw_features: pd.DataFrame,
        config: dict | None = None,
    ):
        if config is None:
            config = {"features": {"version": "v1"}}

        return patch.multiple(
            "src.data.walk_forward_split",
            load_config=unittest.mock.Mock(return_value=config),
            build_returns_dataset=unittest.mock.Mock(return_value=returns),
            build_configured_features=unittest.mock.Mock(return_value=raw_features),
        )

    @staticmethod
    def _returns_and_features() -> tuple[pd.DataFrame, pd.DataFrame]:
        index = pd.date_range("2024-01-01", periods=8, freq="D")
        returns = pd.DataFrame({"SPY": range(8), "CASH": 0.0}, index=index)
        features = pd.DataFrame({"feature": range(10, 18)}, index=index)

        return returns, features

    @staticmethod
    def _mock_builder_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
        index = pd.date_range("2024-01-01", periods=12, freq="D")
        returns = pd.DataFrame(
            {
                "SPY": [value / 100.0 for value in range(12)],
                "CASH": [0.0] * 12,
            },
            index=index,
        )
        raw_features = pd.DataFrame(
            {
                "SPY_ret_1w": [value * 2.0 for value in range(12)],
                "CASH_ret_1w": [0.0] * 12,
            },
            index=index,
        )

        return returns, raw_features

    @staticmethod
    def _fold() -> dict:
        return {
            "fold_id": "F1",
            "description": "test_fold",
            "train_start": "2024-01-03",
            "train_end": "2024-01-05",
            "validation_start": "2024-01-06",
            "validation_end": "2024-01-08",
            "test_start": "2024-01-09",
            "test_end": "2024-01-11",
        }

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


if __name__ == "__main__":
    unittest.main()
