"""Tests for simple dynamic allocation benchmark rules."""

import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.backtest.dynamic_allocation_benchmarks import (
    build_benchmark_timing_audit_summary,
    build_defensive_risk_off_weights,
    build_dynamic_benchmark_suite,
    build_momentum_winner_weights,
    build_risk_adjusted_momentum_winner_weights,
    build_trend_following_spy_cash_weights,
    compute_rolling_momentum,
    compute_rolling_volatility,
    evaluate_weight_strategy,
    save_dynamic_benchmark_suite,
)


class DynamicAllocationBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
        self.returns = pd.DataFrame(
            {
                "SPY": [0.02, 0.03, -0.04, 0.01, 0.02, -0.01, 0.03, 0.02],
                "TLT": [0.00, -0.01, 0.02, 0.01, -0.01, 0.02, 0.00, 0.01],
                "GLD": [0.01, 0.02, 0.03, -0.01, 0.00, 0.04, 0.01, -0.01],
                "BTC-USD": [0.05, -0.03, 0.06, 0.04, -0.02, 0.03, 0.05, -0.01],
                "CASH": [0.0] * 8,
            },
            index=self.dates,
        )

    def test_compute_rolling_momentum_returns_expected_compounded_value(self):
        returns = self.returns[["SPY"]]

        result = compute_rolling_momentum(returns, window=2)

        expected = (1.02 * 1.03) - 1.0
        self.assertAlmostEqual(result.loc[self.dates[1], "SPY"], expected)

    def test_compute_rolling_volatility_returns_expected_rolling_std(self):
        returns = self.returns[["SPY"]]

        result = compute_rolling_volatility(returns, window=2)

        expected = pd.Series([0.02, 0.03]).std()
        self.assertAlmostEqual(result.loc[self.dates[1], "SPY"], expected)

    def test_momentum_winner_allocates_to_asset_with_highest_prior_momentum(self):
        returns = pd.DataFrame(
            {
                "SPY": [0.10, 0.10, -0.50],
                "GLD": [0.00, 0.00, 0.20],
                "CASH": [0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
        )

        weights = build_momentum_winner_weights(returns, window=2)

        self.assertEqual(weights.loc[returns.index[2], "SPY"], 1.0)
        self.assertEqual(weights.loc[returns.index[2], "GLD"], 0.0)

    def test_momentum_winner_uses_one_period_shift_without_same_period_leakage(self):
        returns = pd.DataFrame(
            {
                "SPY": [0.10, 0.10, -0.50],
                "GLD": [0.00, 0.00, 0.20],
                "CASH": [0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
        )

        weights = build_momentum_winner_weights(returns, window=2)
        evaluation = evaluate_weight_strategy(returns, weights, transaction_cost=0.0)

        self.assertAlmostEqual(evaluation["returns"].loc[returns.index[2]], -0.50)

    def test_momentum_winner_does_not_select_future_winner_before_observable(self):
        returns = pd.DataFrame(
            {
                "SPY": [0.10, 0.10, -0.50, 0.00],
                "GLD": [0.00, 0.00, 0.40, 0.00],
                "CASH": [0.00, 0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
        )

        weights = build_momentum_winner_weights(returns, window=2)

        self.assertEqual(weights.loc[returns.index[2], "SPY"], 1.0)
        self.assertEqual(weights.loc[returns.index[2], "GLD"], 0.0)
        self.assertEqual(weights.loc[returns.index[3], "GLD"], 1.0)

    def test_dynamic_benchmark_weights_are_lagged_relative_to_signal(self):
        returns = pd.DataFrame(
            {
                "SPY": [0.10, 0.10, -0.50, 0.00],
                "GLD": [0.00, 0.00, 0.40, 0.00],
                "CASH": [0.00, 0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
        )
        momentum = compute_rolling_momentum(returns[["SPY", "GLD", "CASH"]], window=2)

        weights = build_momentum_winner_weights(returns, window=2)

        self.assertEqual(momentum.loc[returns.index[2]].idxmax(), "GLD")
        self.assertEqual(weights.loc[returns.index[2], "SPY"], 1.0)
        self.assertEqual(weights.loc[returns.index[3], "GLD"], 1.0)

    def test_risk_adjusted_momentum_winner_allocates_by_score(self):
        returns = pd.DataFrame(
            {
                "SPY": [0.30, -0.20, 0.30, 0.00],
                "GLD": [0.04, 0.05, 0.06, 0.00],
                "CASH": [0.00, 0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
        )

        weights = build_risk_adjusted_momentum_winner_weights(
            returns,
            momentum_window=3,
            volatility_window=3,
        )

        self.assertEqual(weights.loc[returns.index[3], "GLD"], 1.0)

    def test_trend_following_allocates_to_spy_when_prior_momentum_positive(self):
        returns = pd.DataFrame(
            {
                "SPY": [0.02, 0.03, 0.00],
                "CASH": [0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
        )

        weights = build_trend_following_spy_cash_weights(returns, window=2)

        self.assertEqual(weights.loc[returns.index[2], "SPY"], 1.0)

    def test_trend_following_allocates_to_cash_when_prior_momentum_non_positive(self):
        returns = pd.DataFrame(
            {
                "SPY": [-0.02, -0.03, 0.00],
                "CASH": [0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
        )

        weights = build_trend_following_spy_cash_weights(returns, window=2)

        self.assertEqual(weights.loc[returns.index[2], "CASH"], 1.0)

    def test_trend_following_reacts_one_period_after_same_period_flip(self):
        returns = pd.DataFrame(
            {
                "SPY": [-0.10, -0.10, 0.50, 0.00],
                "CASH": [0.00, 0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
        )

        weights = build_trend_following_spy_cash_weights(returns, window=2)

        self.assertEqual(weights.loc[returns.index[2], "CASH"], 1.0)
        self.assertEqual(weights.loc[returns.index[3], "SPY"], 1.0)

    def test_defensive_risk_off_allocates_to_best_defensive_asset(self):
        returns = pd.DataFrame(
            {
                "SPY": [-0.02, -0.03, 0.00],
                "TLT": [0.01, 0.01, 0.00],
                "GLD": [0.03, 0.04, 0.00],
                "CASH": [0.00, 0.00, 0.00],
            },
            index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
        )

        weights = build_defensive_risk_off_weights(returns, window=2)

        self.assertEqual(weights.loc[returns.index[2], "GLD"], 1.0)

    def test_weight_rows_sum_to_one(self):
        weights = build_momentum_winner_weights(self.returns, window=2)

        self.assertTrue((weights.sum(axis=1).round(12) == 1.0).all())

    def test_missing_eligible_asset_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_momentum_winner_weights(
                self.returns,
                window=2,
                eligible_assets=["SPY", "MISSING"],
            )

    def test_evaluate_weight_strategy_computes_net_returns_after_transaction_costs(self):
        returns = pd.DataFrame(
            {"SPY": [0.10, 0.00], "GLD": [0.00, 0.04]},
            index=pd.date_range("2024-01-05", periods=2, freq="W-FRI"),
        )
        weights = pd.DataFrame(
            {"SPY": [1.0, 0.0], "GLD": [0.0, 1.0]},
            index=returns.index,
        )

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.01)

        self.assertAlmostEqual(result["returns"].iloc[0], 0.09)
        self.assertAlmostEqual(result["returns"].iloc[1], 0.02)

    def test_evaluate_weight_strategy_computes_turnover_including_first_allocation(self):
        returns = pd.DataFrame(
            {"SPY": [0.10, 0.00], "GLD": [0.00, 0.04]},
            index=pd.date_range("2024-01-05", periods=2, freq="W-FRI"),
        )
        weights = pd.DataFrame(
            {"SPY": [1.0, 0.0], "GLD": [0.0, 1.0]},
            index=returns.index,
        )

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.01)

        self.assertAlmostEqual(result["history"].loc[0, "turnover"], 1.0)
        self.assertAlmostEqual(result["history"].loc[1, "turnover"], 2.0)

    def test_evaluate_weight_strategy_initial_turnover_matches_td3_equal_weight_start(self):
        dates = pd.date_range("2024-01-05", periods=1, freq="W-FRI")
        returns = pd.DataFrame(
            {
                "SPY": [0.00],
                "TLT": [0.00],
                "GLD": [0.00],
                "BTC-USD": [0.00],
                "CASH": [0.00],
            },
            index=dates,
        )
        weights = pd.DataFrame(
            {
                "SPY": [1.0],
                "TLT": [0.0],
                "GLD": [0.0],
                "BTC-USD": [0.0],
                "CASH": [0.0],
            },
            index=dates,
        )

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.01)

        self.assertAlmostEqual(result["history"].loc[0, "turnover"], 1.6)
        self.assertAlmostEqual(result["history"].loc[0, "transaction_cost"], 0.016)

    def test_evaluate_weight_strategy_transaction_cost_reduces_financial_net_return(self):
        returns = pd.DataFrame(
            {"SPY": [0.10], "GLD": [0.00]},
            index=pd.date_range("2024-01-05", periods=1, freq="W-FRI"),
        )
        weights = pd.DataFrame({"SPY": [1.0], "GLD": [0.0]}, index=returns.index)

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.01)
        history = result["history"]

        self.assertAlmostEqual(history.loc[0, "portfolio_return"], 0.10)
        self.assertAlmostEqual(history.loc[0, "transaction_cost"], 0.01)
        self.assertAlmostEqual(history.loc[0, "financial_net_return"], 0.09)

    def test_evaluate_weight_strategy_history_contains_comparison_columns(self):
        weights = pd.DataFrame(1.0, index=self.returns.index, columns=["SPY"])
        returns = self.returns[["SPY"]]

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.0)
        history = result["history"]

        for column in [
            "date",
            "portfolio_return",
            "transaction_cost",
            "financial_net_return",
            "turnover",
            "portfolio_value",
            "drawdown",
            "weight_SPY",
        ]:
            self.assertIn(column, history.columns)

    def test_evaluate_weight_strategy_computes_drawdown(self):
        returns = pd.DataFrame(
            {"SPY": [0.10, -0.20, 0.05]},
            index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
        )
        weights = pd.DataFrame({"SPY": [1.0, 1.0, 1.0]}, index=returns.index)

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.0)

        self.assertAlmostEqual(result["max_drawdown"], -0.20)

    def test_evaluate_weight_strategy_returns_comparable_metric_keys(self):
        weights = pd.DataFrame(1.0, index=self.returns.index, columns=["SPY"])
        returns = self.returns[["SPY"]]

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.0)

        for key in [
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "cumulative_return",
            "max_drawdown",
            "average_turnover",
            "final_value",
        ]:
            self.assertIn(key, result)

    def test_evaluate_weight_strategy_returns_ratio_diagnostic_flags(self):
        weights = pd.DataFrame(1.0, index=self.returns.index, columns=["SPY"])
        returns = self.returns[["SPY"]]

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.0)

        for key in [
            "sortino_ratio_is_finite",
            "sortino_ratio_is_extreme",
            "calmar_ratio_is_finite",
            "calmar_ratio_is_infinite",
            "max_drawdown_is_zero",
        ]:
            self.assertIn(key, result)

    def test_evaluate_weight_strategy_metrics_are_finite_when_expected(self):
        weights = pd.DataFrame(1.0, index=self.returns.index, columns=["SPY"])
        returns = self.returns[["SPY"]]

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.0)

        self.assertTrue(math.isfinite(result["annualized_return"]))
        self.assertTrue(math.isfinite(result["annualized_volatility"]))
        self.assertTrue(math.isfinite(result["sharpe_ratio"]))
        self.assertTrue(math.isfinite(result["sortino_ratio"]))
        self.assertTrue(math.isfinite(result["calmar_ratio"]))

    def test_finite_normal_ratios_have_expected_flags(self):
        returns = pd.DataFrame(
            {"SPY": [0.02, -0.01, 0.03, -0.02, 0.01, 0.02]},
            index=pd.date_range("2024-01-05", periods=6, freq="W-FRI"),
        )
        weights = pd.DataFrame({"SPY": [1.0] * 6}, index=returns.index)

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.0)

        self.assertTrue(result["sortino_ratio_is_finite"])
        self.assertFalse(result["sortino_ratio_is_extreme"])
        self.assertTrue(result["calmar_ratio_is_finite"])
        self.assertFalse(result["calmar_ratio_is_infinite"])
        self.assertFalse(result["max_drawdown_is_zero"])

    def test_zero_volatility_cash_like_returns_do_not_crash(self):
        returns = pd.DataFrame(
            {"CASH": [0.0, 0.0, 0.0, 0.0]},
            index=pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
        )
        weights = pd.DataFrame({"CASH": [1.0, 1.0, 1.0, 1.0]}, index=returns.index)

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.0)

        self.assertEqual(result["annualized_volatility"], 0.0)
        self.assertEqual(result["sharpe_ratio"], 0.0)
        self.assertEqual(result["sortino_ratio"], 0.0)
        self.assertEqual(result["calmar_ratio"], 0.0)

    def test_calmar_is_safe_when_drawdown_is_zero(self):
        returns = pd.DataFrame(
            {"SPY": [0.01, 0.01, 0.01, 0.01]},
            index=pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
        )
        weights = pd.DataFrame({"SPY": [1.0, 1.0, 1.0, 1.0]}, index=returns.index)

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.0)

        self.assertEqual(result["max_drawdown"], 0.0)
        self.assertTrue(math.isinf(result["calmar_ratio"]))

    def test_infinite_calmar_sets_diagnostic_flags(self):
        returns = pd.DataFrame(
            {"SPY": [0.01, 0.01, 0.01, 0.01]},
            index=pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
        )
        weights = pd.DataFrame({"SPY": [1.0, 1.0, 1.0, 1.0]}, index=returns.index)

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.0)

        self.assertTrue(result["calmar_ratio_is_infinite"])
        self.assertFalse(result["calmar_ratio_is_finite"])
        self.assertTrue(result["max_drawdown_is_zero"])

    def test_extreme_sortino_sets_diagnostic_flag(self):
        returns = pd.DataFrame(
            {"SPY": [0.01, 0.01, 0.01, 0.01]},
            index=pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
        )
        weights = pd.DataFrame({"SPY": [1.0, 1.0, 1.0, 1.0]}, index=returns.index)

        result = evaluate_weight_strategy(returns, weights, transaction_cost=0.0)

        self.assertTrue(result["sortino_ratio_is_extreme"])

    def test_build_dynamic_benchmark_suite_returns_expected_names_when_assets_exist(self):
        result = build_dynamic_benchmark_suite(
            self.returns,
            momentum_window=2,
            volatility_window=2,
        )

        self.assertEqual(
            set(result.keys()),
            {
                "momentum_winner_2p",
                "risk_adjusted_momentum_winner_2p_2p",
                "trend_spy_cash_2p",
                "defensive_risk_off_2p",
            },
        )

    def test_dynamic_benchmark_names_reflect_four_period_windows(self):
        result = build_dynamic_benchmark_suite(
            self.returns,
            momentum_window=4,
            volatility_window=4,
        )

        self.assertEqual(
            set(result.keys()),
            {
                "momentum_winner_4p",
                "risk_adjusted_momentum_winner_4p_4p",
                "trend_spy_cash_4p",
                "defensive_risk_off_4p",
            },
        )
        self.assertFalse(any(name.endswith("12p") for name in result))

    def test_risk_adjusted_benchmark_name_reflects_mixed_windows(self):
        result = build_dynamic_benchmark_suite(
            self.returns,
            momentum_window=4,
            volatility_window=3,
        )

        self.assertIn("risk_adjusted_momentum_winner_4p_3p", result)

    def test_save_dynamic_benchmark_suite_writes_history_and_summary_csvs(self):
        suite = build_dynamic_benchmark_suite(
            self.returns,
            momentum_window=2,
            volatility_window=2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_dynamic_benchmark_suite(suite, temp_dir)

            self.assertTrue(Path(paths["summary"]).exists())
            for benchmark_name in suite:
                self.assertTrue(Path(paths[f"{benchmark_name}_history"]).exists())

    def test_saved_summary_contains_comparable_metric_columns(self):
        suite = build_dynamic_benchmark_suite(
            self.returns,
            momentum_window=2,
            volatility_window=2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_dynamic_benchmark_suite(suite, temp_dir)
            summary = pd.read_csv(paths["summary"])

        for column in [
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
        ]:
            self.assertIn(column, summary.columns)

    def test_saved_summary_contains_ratio_diagnostic_flag_columns(self):
        suite = build_dynamic_benchmark_suite(
            self.returns,
            momentum_window=2,
            volatility_window=2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_dynamic_benchmark_suite(suite, temp_dir)
            summary = pd.read_csv(paths["summary"])

        for column in [
            "sortino_ratio_is_finite",
            "sortino_ratio_is_extreme",
            "calmar_ratio_is_finite",
            "calmar_ratio_is_infinite",
            "max_drawdown_is_zero",
        ]:
            self.assertIn(column, summary.columns)

    def test_benchmark_timing_audit_summary_reports_dynamic_rules_comparable(self):
        summary = build_benchmark_timing_audit_summary().set_index("benchmark_name")

        for benchmark_name in [
            "momentum_winner_12p",
            "risk_adjusted_momentum_winner_12p_12p",
            "trend_spy_cash_12p",
            "defensive_risk_off_12p",
        ]:
            self.assertTrue(summary.loc[benchmark_name, "signal_lagged"])
            self.assertTrue(
                summary.loc[benchmark_name, "applies_return_t_after_weight_t"]
            )
            self.assertTrue(summary.loc[benchmark_name, "comparable_with_td3"])

    def test_benchmark_timing_audit_summary_reports_static_gross_references(self):
        summary = build_benchmark_timing_audit_summary().set_index("benchmark_name")

        self.assertFalse(summary.loc["BuyHold_GLD", "comparable_with_td3"])
        self.assertIn(
            "none",
            summary.loc["BuyHold_GLD", "transaction_cost_convention"],
        )

    def test_invalid_inputs_raise_errors(self):
        with self.assertRaises(ValueError):
            compute_rolling_momentum(pd.DataFrame(), window=2)
        with self.assertRaises(TypeError):
            compute_rolling_momentum(pd.DataFrame({"SPY": [0.01, 0.02]}), window=2)
        with self.assertRaises(ValueError):
            compute_rolling_momentum(self.returns, window=1)
        with self.assertRaises(ValueError):
            build_risk_adjusted_momentum_winner_weights(
                self.returns,
                volatility_floor=0.0,
            )
        with self.assertRaises(ValueError):
            evaluate_weight_strategy(self.returns, self.returns, transaction_cost=1.0)
        with self.assertRaises(ValueError):
            evaluate_weight_strategy(self.returns, self.returns, initial_value=0.0)


if __name__ == "__main__":
    unittest.main()
