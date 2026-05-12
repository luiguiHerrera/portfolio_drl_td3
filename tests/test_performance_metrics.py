"""Tests for extended portfolio performance metrics."""

import math
import unittest

import pandas as pd

from src.backtest.performance_metrics import (
    align_return_series,
    calmar_ratio,
    capm_beta_alpha,
    downside_deviation,
    extended_summary_metrics,
    information_ratio,
    sortino_ratio,
    tracking_error,
)


class PerformanceMetricsTests(unittest.TestCase):
    def test_downside_deviation_returns_positive_value_with_negative_returns(self):
        returns = self._series([0.01, -0.02, 0.03, -0.01])

        self.assertGreater(downside_deviation(returns), 0.0)

    def test_downside_deviation_returns_zero_for_returns_above_mar(self):
        returns = self._series([0.01, 0.02, 0.03])

        self.assertEqual(downside_deviation(returns), 0.0)

    def test_downside_deviation_rejects_zero_periods_per_year(self):
        returns = self._series([0.01, -0.02, 0.03])

        with self.assertRaises(ValueError):
            downside_deviation(returns, periods_per_year=0)

    def test_sortino_ratio_finite_case(self):
        returns = self._series([0.01, -0.01, 0.02, 0.00])

        self.assertTrue(math.isfinite(sortino_ratio(returns)))

    def test_sortino_ratio_inf_when_no_downside_and_positive_return(self):
        returns = self._series([0.01, 0.02, 0.03])

        self.assertEqual(sortino_ratio(returns), float("inf"))

    def test_sortino_ratio_rejects_non_numeric_minimum_acceptable_return(self):
        returns = self._series([0.01, -0.01, 0.02])

        with self.assertRaises(TypeError):
            sortino_ratio(returns, minimum_acceptable_return="0.01")

    def test_calmar_ratio_finite_case(self):
        returns = self._series([0.02, -0.01, 0.03, -0.02])

        self.assertTrue(math.isfinite(calmar_ratio(returns)))

    def test_calmar_ratio_inf_when_no_drawdown_and_positive_return(self):
        returns = self._series([0.01, 0.02, 0.03])

        self.assertEqual(calmar_ratio(returns), float("inf"))

    def test_align_return_series_aligns_by_shared_index_and_drops_nans(self):
        strategy = self._series([0.01, None, 0.03, 0.04])
        benchmark = pd.Series(
            [0.02, 0.03, 0.04],
            index=strategy.index[1:].copy(),
        )

        aligned_strategy, aligned_benchmark = align_return_series(strategy, benchmark)

        self.assertEqual(list(aligned_strategy.index), list(strategy.index[2:]))
        self.assertEqual(list(aligned_benchmark.index), list(strategy.index[2:]))

    def test_align_return_series_rejects_no_shared_index(self):
        strategy = self._series([0.01, 0.02])
        benchmark = pd.Series(
            [0.01, 0.02],
            index=pd.date_range("2025-01-01", periods=2, freq="D"),
        )

        with self.assertRaises(ValueError):
            align_return_series(strategy, benchmark)

    def test_tracking_error_computes_positive_value_when_active_returns_vary(self):
        strategy = self._series([0.01, 0.03, -0.01, 0.02])
        benchmark = self._series([0.00, 0.01, 0.01, 0.02])

        self.assertGreater(tracking_error(strategy, benchmark), 0.0)

    def test_tracking_error_rejects_fewer_than_two_aligned_observations(self):
        strategy = self._series([0.01])
        benchmark = self._series([0.00])

        with self.assertRaises(ValueError):
            tracking_error(strategy, benchmark)

    def test_information_ratio_finite_case(self):
        strategy = self._series([0.01, 0.03, -0.01, 0.02])
        benchmark = self._series([0.00, 0.01, 0.01, 0.02])

        self.assertTrue(math.isfinite(information_ratio(strategy, benchmark)))

    def test_capm_beta_alpha_returns_beta_close_to_expected(self):
        market = self._series([-0.02, -0.01, 0.00, 0.01, 0.02])
        strategy = 2.0 * market

        result = capm_beta_alpha(strategy, market)

        self.assertAlmostEqual(result["capm_beta"], 2.0)

    def test_capm_beta_alpha_returns_positive_alpha_after_beta_adjustment(self):
        market = self._series([-0.02, -0.01, 0.00, 0.01, 0.02])
        strategy = market + 0.01

        result = capm_beta_alpha(strategy, market)

        self.assertAlmostEqual(result["capm_beta"], 1.0)
        self.assertGreater(result["capm_alpha"], 0.0)

    def test_capm_beta_alpha_rejects_zero_periods_per_year(self):
        strategy = self._series([0.01, 0.02, 0.03])
        market = self._series([0.00, 0.01, 0.02])

        with self.assertRaises(ValueError):
            capm_beta_alpha(strategy, market, periods_per_year=0)

    def test_capm_beta_alpha_rejects_non_numeric_risk_free_rate(self):
        strategy = self._series([0.01, 0.02, 0.03])
        market = self._series([0.00, 0.01, 0.02])

        with self.assertRaises(TypeError):
            capm_beta_alpha(strategy, market, risk_free_rate="0.02")

    def test_capm_beta_alpha_rejects_zero_market_variance(self):
        strategy = self._series([0.01, 0.02, 0.03])
        market = self._series([0.01, 0.01, 0.01])

        with self.assertRaises(ValueError):
            capm_beta_alpha(strategy, market)

    def test_extended_summary_metrics_includes_only_sortino_calmar_without_inputs(self):
        returns = self._series([0.01, -0.01, 0.02])

        result = extended_summary_metrics(returns)

        self.assertEqual(set(result.keys()), {"sortino_ratio", "calmar_ratio"})

    def test_extended_summary_metrics_rejects_non_integer_periods_per_year(self):
        returns = self._series([0.01, -0.01, 0.02])

        with self.assertRaises(TypeError):
            extended_summary_metrics(returns, periods_per_year=52.0)

    def test_extended_summary_metrics_includes_tracking_and_information_with_benchmark(self):
        returns = self._series([0.01, -0.01, 0.02])
        benchmark = self._series([0.00, 0.01, 0.01])

        result = extended_summary_metrics(returns, benchmark_returns=benchmark)

        self.assertIn("tracking_error", result)
        self.assertIn("information_ratio", result)

    def test_extended_summary_metrics_includes_capm_alpha_beta_with_market(self):
        returns = self._series([0.01, -0.01, 0.02])
        market = self._series([0.00, -0.02, 0.01])

        result = extended_summary_metrics(returns, market_returns=market)

        self.assertIn("capm_alpha", result)
        self.assertIn("capm_beta", result)

    @staticmethod
    def _series(values: list[float]) -> pd.Series:
        return pd.Series(values, index=pd.date_range("2024-01-05", periods=len(values), freq="W-FRI"))


if __name__ == "__main__":
    unittest.main()
