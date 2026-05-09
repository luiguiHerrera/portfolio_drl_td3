"""Tests for allocation risk diagnostics."""

import unittest

import numpy as np
import pandas as pd

from src.backtest.allocation_diagnostics import allocation_diagnostics


class AllocationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
        self.weights = pd.DataFrame(
            {
                "SPY": [0.2, 0.3, 0.4],
                "TLT": [0.2, 0.2, 0.2],
                "GLD": [0.2, 0.1, 0.1],
                "BTC-USD": [0.2, 0.2, 0.1],
                "CASH": [0.2, 0.2, 0.2],
            },
            index=self.index,
        )

    def test_returns_expected_keys_for_weights_only(self):
        diagnostics = allocation_diagnostics(self.weights)
        expected_keys = {
            "average_max_weight",
            "final_max_weight",
            "average_cash_weight",
            "final_cash_weight",
            "average_herfindahl_index",
            "final_herfindahl_index",
            "average_effective_number_of_assets",
            "final_effective_number_of_assets",
            "average_entropy",
            "final_entropy",
        }

        self.assertEqual(set(diagnostics.keys()), expected_keys)

    def test_handles_cash_column(self):
        diagnostics = allocation_diagnostics(self.weights)

        self.assertAlmostEqual(diagnostics["average_cash_weight"], 0.2)
        self.assertAlmostEqual(diagnostics["final_cash_weight"], 0.2)

    def test_returns_zero_cash_weights_when_cash_column_is_absent(self):
        weights = pd.DataFrame(
            np.full((3, 4), 0.25),
            index=self.index,
            columns=["SPY", "TLT", "GLD", "BTC-USD"],
        )
        diagnostics = allocation_diagnostics(weights)

        self.assertEqual(diagnostics["average_cash_weight"], 0.0)
        self.assertEqual(diagnostics["final_cash_weight"], 0.0)

    def test_hhi_and_effective_number_are_correct_for_equal_weights(self):
        weights = pd.DataFrame(
            np.full((3, 5), 0.2),
            index=self.index,
            columns=["SPY", "TLT", "GLD", "BTC-USD", "CASH"],
        )
        diagnostics = allocation_diagnostics(weights)

        self.assertAlmostEqual(diagnostics["average_herfindahl_index"], 0.2)
        self.assertAlmostEqual(diagnostics["final_herfindahl_index"], 0.2)
        self.assertAlmostEqual(diagnostics["average_effective_number_of_assets"], 5.0)
        self.assertAlmostEqual(diagnostics["final_effective_number_of_assets"], 5.0)

    def test_entropy_does_not_produce_nan_with_zero_weights(self):
        weights = pd.DataFrame(
            {
                "SPY": [1.0, 0.0],
                "TLT": [0.0, 1.0],
                "GLD": [0.0, 0.0],
            },
            index=pd.date_range("2024-01-05", periods=2, freq="W-FRI"),
        )
        diagnostics = allocation_diagnostics(weights)

        self.assertFalse(np.isnan(diagnostics["average_entropy"]))
        self.assertFalse(np.isnan(diagnostics["final_entropy"]))

    def test_includes_turnover_diagnostics_when_turnover_provided(self):
        turnover = pd.Series([0.1, 0.2, 0.3], index=self.index)
        diagnostics = allocation_diagnostics(self.weights, turnover=turnover)

        self.assertAlmostEqual(diagnostics["average_turnover"], 0.2)
        self.assertAlmostEqual(diagnostics["final_turnover"], 0.3)

    def test_includes_transaction_cost_diagnostics_when_costs_provided(self):
        transaction_costs = pd.Series([0.001, 0.002, 0.003], index=self.index)
        diagnostics = allocation_diagnostics(
            self.weights,
            transaction_costs=transaction_costs,
        )

        self.assertAlmostEqual(diagnostics["average_transaction_cost"], 0.002)
        self.assertAlmostEqual(diagnostics["final_transaction_cost"], 0.003)

    def test_rejects_empty_weights(self):
        with self.assertRaises(ValueError):
            allocation_diagnostics(pd.DataFrame())

    def test_rejects_negative_weights(self):
        weights = self.weights.copy()
        weights.iloc[0, 0] = -0.1

        with self.assertRaises(ValueError):
            allocation_diagnostics(weights)

    def test_rejects_rows_that_do_not_sum_to_one(self):
        weights = self.weights.copy()
        weights.iloc[0, 0] = 0.9

        with self.assertRaises(ValueError):
            allocation_diagnostics(weights)

    def test_rejects_turnover_index_mismatch(self):
        turnover = pd.Series([0.1, 0.2, 0.3], index=pd.RangeIndex(3))

        with self.assertRaises(ValueError):
            allocation_diagnostics(self.weights, turnover=turnover)

    def test_rejects_transaction_cost_index_mismatch(self):
        transaction_costs = pd.Series([0.001, 0.002, 0.003], index=pd.RangeIndex(3))

        with self.assertRaises(ValueError):
            allocation_diagnostics(self.weights, transaction_costs=transaction_costs)


if __name__ == "__main__":
    unittest.main()
