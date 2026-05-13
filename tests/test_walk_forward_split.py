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

    def _patched_builders(self, returns: pd.DataFrame, raw_features: pd.DataFrame):
        return patch.multiple(
            "src.data.walk_forward_split",
            build_returns_dataset=unittest.mock.Mock(return_value=returns),
            build_features=unittest.mock.Mock(return_value=raw_features),
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


if __name__ == "__main__":
    unittest.main()
