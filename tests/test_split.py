"""Tests for chronological data splitting."""

import unittest

import pandas as pd

from src.data.split import chronological_split


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {"value": range(10)},
            index=pd.date_range("2024-01-01", periods=10, freq="D"),
        )

    def test_chronological_split_preserves_order(self):
        train, validation, test = chronological_split(self.data, 0.6, 0.2, 0.2)

        self.assertTrue(train.index.is_monotonic_increasing)
        self.assertTrue(validation.index.is_monotonic_increasing)
        self.assertTrue(test.index.is_monotonic_increasing)
        self.assertLess(train.index[-1], validation.index[0])
        self.assertLess(validation.index[-1], test.index[0])

    def test_ratios_create_expected_lengths(self):
        train, validation, test = chronological_split(self.data, 0.6, 0.2, 0.2)

        self.assertEqual(len(train), 6)
        self.assertEqual(len(validation), 2)
        self.assertEqual(len(test), 2)

    def test_rejects_ratios_that_do_not_sum_to_one(self):
        with self.assertRaises(ValueError):
            chronological_split(self.data, 0.5, 0.2, 0.2)

    def test_rejects_empty_data(self):
        with self.assertRaises(ValueError):
            chronological_split(pd.DataFrame(), 0.6, 0.2, 0.2)

    def test_rejects_splits_that_produce_empty_partition(self):
        short_data = pd.DataFrame({"value": [1, 2]})

        with self.assertRaises(ValueError):
            chronological_split(short_data, 0.6, 0.2, 0.2)


if __name__ == "__main__":
    unittest.main()
