"""Tests for train-only feature normalization."""

import unittest

import numpy as np
import pandas as pd

from src.data.normalize import (
    apply_standard_scaler,
    fit_standard_scaler,
    normalize_train_validation_test,
)


class NormalizeTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0],
                "feature_b": [10.0, 12.0, 14.0, 16.0],
                "constant": [5.0, 5.0, 5.0, 5.0],
            },
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )
        self.validation = pd.DataFrame(
            {
                "feature_a": [5.0, 6.0],
                "feature_b": [18.0, 20.0],
                "constant": [5.0, 5.0],
            },
            index=pd.date_range("2024-01-05", periods=2, freq="D"),
        )
        self.test = pd.DataFrame(
            {
                "feature_a": [7.0, 8.0],
                "feature_b": [22.0, 24.0],
                "constant": [5.0, 5.0],
            },
            index=pd.date_range("2024-01-07", periods=2, freq="D"),
        )

    def test_scaler_mean_and_std_are_computed_only_from_train(self):
        scaler = fit_standard_scaler(self.train)

        pd.testing.assert_series_equal(scaler["mean"], self.train.mean())
        pd.testing.assert_series_equal(
            scaler["std"],
            self.train.std().mask(self.train.std() == 0.0, 1.0),
        )

    def test_normalized_train_has_approximately_zero_mean(self):
        train_norm, _, _, _ = normalize_train_validation_test(
            self.train,
            self.validation,
            self.test,
        )

        self.assertTrue(np.allclose(train_norm.mean(), 0.0))

    def test_validation_and_test_use_train_stats(self):
        scaler = fit_standard_scaler(self.train)
        validation_norm = apply_standard_scaler(self.validation, scaler)
        test_norm = apply_standard_scaler(self.test, scaler)

        expected_validation = (self.validation - self.train.mean()) / scaler["std"]
        expected_test = (self.test - self.train.mean()) / scaler["std"]
        pd.testing.assert_frame_equal(validation_norm, expected_validation)
        pd.testing.assert_frame_equal(test_norm, expected_test)

    def test_constant_column_does_not_create_inf_or_nan(self):
        train_norm, validation_norm, test_norm, _ = normalize_train_validation_test(
            self.train,
            self.validation,
            self.test,
        )

        combined = pd.concat([train_norm, validation_norm, test_norm])
        self.assertFalse(np.isinf(combined.to_numpy()).any())
        self.assertFalse(combined.isna().any().any())

    def test_single_row_train_std_is_replaced_with_one(self):
        train = pd.DataFrame(
            {"feature_a": [1.0], "feature_b": [5.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="D"),
        )
        scaler = fit_standard_scaler(train)
        normalized = apply_standard_scaler(train, scaler)

        self.assertTrue((scaler["std"] == 1.0).all())
        self.assertFalse(np.isinf(normalized.to_numpy()).any())
        self.assertFalse(normalized.isna().any().any())

    def test_output_preserves_index_and_columns(self):
        train_norm, validation_norm, test_norm, _ = normalize_train_validation_test(
            self.train,
            self.validation,
            self.test,
        )

        self.assertTrue(train_norm.index.equals(self.train.index))
        self.assertTrue(validation_norm.index.equals(self.validation.index))
        self.assertTrue(test_norm.index.equals(self.test.index))
        self.assertTrue(train_norm.columns.equals(self.train.columns))
        self.assertTrue(validation_norm.columns.equals(self.validation.columns))
        self.assertTrue(test_norm.columns.equals(self.test.columns))

    def test_column_mismatch_raises_value_error(self):
        scaler = fit_standard_scaler(self.train)
        mismatched = self.validation.rename(columns={"feature_a": "other_feature"})

        with self.assertRaises(ValueError):
            apply_standard_scaler(mismatched, scaler)


if __name__ == "__main__":
    unittest.main()
