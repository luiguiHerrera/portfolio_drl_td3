"""Tests for ex-post shadow mandate penalty reports."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.shadow_mandate_penalty_report import (
    build_shadow_mandate_penalty_observations,
    build_shadow_mandate_penalty_report,
    compute_effective_assets_from_weights,
    compute_trailing_volatility,
    infer_weight_columns,
    summarize_shadow_mandate_penalties,
)


class ShadowMandatePenaltyReportTests(unittest.TestCase):
    def test_infer_weight_columns_excludes_bool_and_ok_columns(self):
        data = self._history()
        data["weight_ok"] = [True] * len(data)
        data["weight_flag"] = [False] * len(data)

        result = infer_weight_columns(data)

        self.assertEqual(result, ["weight_SPY", "weight_GLD", "weight_CASH"])

    def test_compute_effective_assets_from_weights_all_in_and_half_half(self):
        weights = pd.DataFrame(
            {
                "weight_SPY": [1.0, 0.5],
                "weight_GLD": [0.0, 0.5],
            }
        )

        result = compute_effective_assets_from_weights(weights)

        self.assertAlmostEqual(result.iloc[0], 1.0)
        self.assertAlmostEqual(result.iloc[1], 2.0)

    def test_compute_trailing_volatility_annualizes_correctly(self):
        returns = pd.Series([0.01, 0.03, 0.02])

        result = compute_trailing_volatility(returns, window=2, periods_per_year=52)

        expected = pd.Series([0.01, 0.03]).std() * np.sqrt(52)
        self.assertAlmostEqual(result.iloc[1], expected)

    def test_build_observations_returns_breach_columns(self):
        result = build_shadow_mandate_penalty_observations(
            self._history(),
            mandate_profile="moderate",
            volatility_window=3,
        )

        for column in [
            "drawdown_breach",
            "volatility_breach",
            "max_weight_breach",
            "effective_assets_breach",
            "turnover_breach",
            "mandate_penalty",
        ]:
            self.assertIn(column, result.columns)

    def test_build_observations_replaces_existing_mandate_columns(self):
        history = self._history()
        history["mandate_penalty"] = 999.0
        history["drawdown_breach"] = 999.0

        result = build_shadow_mandate_penalty_observations(
            history,
            mandate_profile="moderate",
            volatility_window=3,
        )

        self.assertEqual(result.columns.tolist().count("mandate_penalty"), 1)
        self.assertEqual(result.columns.tolist().count("drawdown_breach"), 1)
        self.assertLess(result["mandate_penalty"].mean(), 999.0)

    def test_build_observations_drops_rows_without_trailing_volatility(self):
        history = self._history()

        result = build_shadow_mandate_penalty_observations(
            history,
            volatility_window=3,
        )

        self.assertEqual(len(result), len(history) - 2)

    def test_moderate_mandate_flags_high_max_weight_breach(self):
        result = build_shadow_mandate_penalty_observations(
            self._history(max_weight=0.95),
            mandate_profile="moderate",
            volatility_window=3,
        )

        self.assertTrue((result["max_weight_breach"] > 0.0).all())

    def test_aggressive_mandate_does_not_flag_max_weight_one(self):
        result = build_shadow_mandate_penalty_observations(
            self._history(max_weight=1.0),
            mandate_profile="aggressive",
            volatility_window=3,
        )

        self.assertTrue((result["max_weight_breach"] == 0.0).all())

    def test_conservative_mandate_has_higher_mean_penalty_than_aggressive(self):
        history = self._history(max_weight=0.90)

        conservative = build_shadow_mandate_penalty_observations(
            history,
            mandate_profile="conservative",
            volatility_window=3,
        )
        aggressive = build_shadow_mandate_penalty_observations(
            history,
            mandate_profile="aggressive",
            volatility_window=3,
        )

        self.assertGreater(
            conservative["mandate_penalty"].mean(),
            aggressive["mandate_penalty"].mean(),
        )

    def test_summarize_shadow_mandate_penalties_returns_expected_columns(self):
        observations = build_shadow_mandate_penalty_observations(
            self._history(),
            volatility_window=3,
        )

        result = summarize_shadow_mandate_penalties(observations)

        expected_columns = {
            "n_observations",
            "mean_mandate_penalty",
            "max_mandate_penalty",
            "penalty_positive_rate",
            "mean_drawdown_breach",
            "mean_volatility_breach",
            "mean_max_weight_breach",
            "mean_effective_assets_breach",
            "mean_turnover_breach",
            "drawdown_breach_rate",
            "volatility_breach_rate",
            "max_weight_breach_rate",
            "effective_assets_breach_rate",
            "turnover_breach_rate",
            "mean_max_weight",
            "mean_effective_assets",
            "mean_trailing_volatility",
            "mean_turnover",
        }
        self.assertTrue(expected_columns.issubset(result.columns))

    def test_build_shadow_mandate_penalty_report_handles_multiple_profiles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = self._write_history(temp_dir)

            result = build_shadow_mandate_penalty_report(
                [history_path],
                strategy_names=["strategy"],
                mandate_profiles=["moderate", "aggressive"],
                volatility_window=3,
            )

        self.assertEqual(set(result["summary"]["mandate_profile"]), {"moderate", "aggressive"})

    def test_build_shadow_mandate_penalty_report_saves_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = self._write_history(temp_dir)
            output_dir = Path(temp_dir) / "report"

            result = build_shadow_mandate_penalty_report(
                [history_path],
                output_dir=str(output_dir),
                volatility_window=3,
            )

            self.assertTrue(Path(result["observations_path"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())

    def test_missing_return_column_raises_value_error(self):
        history = self._history().drop(columns=["net_return"])

        with self.assertRaisesRegex(ValueError, "net_return"):
            build_shadow_mandate_penalty_observations(history)

    def test_missing_drawdown_column_raises_value_error(self):
        history = self._history().drop(columns=["drawdown"])

        with self.assertRaisesRegex(ValueError, "drawdown"):
            build_shadow_mandate_penalty_observations(history)

    def test_missing_turnover_column_raises_value_error(self):
        history = self._history().drop(columns=["turnover"])

        with self.assertRaisesRegex(ValueError, "turnover"):
            build_shadow_mandate_penalty_observations(history)

    def test_no_weight_columns_raises_value_error(self):
        history = self._history().drop(
            columns=["weight_SPY", "weight_GLD", "weight_CASH"]
        )

        with self.assertRaisesRegex(ValueError, "No weight_ columns"):
            build_shadow_mandate_penalty_observations(history)

    def test_input_dataframe_is_not_mutated(self):
        history = self._history()
        original = history.copy(deep=True)

        build_shadow_mandate_penalty_observations(history, volatility_window=3)

        pd.testing.assert_frame_equal(history, original)

    def _write_history(self, directory: str) -> str:
        path = Path(directory) / "strategy_history.csv"
        self._history().to_csv(path, index=False)

        return str(path)

    def _history(self, max_weight=0.95):
        dates = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
        other_weight = (1.0 - max_weight) / 2.0

        return pd.DataFrame(
            {
                "date": dates,
                "net_return": [0.01, -0.02, 0.015, -0.01, 0.02, -0.03, 0.01, 0.005],
                "drawdown": [0.0, 0.02, 0.01, 0.03, 0.0, 0.04, 0.02, 0.01],
                "turnover": [0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.2, 0.1],
                "weight_SPY": [max_weight] * 8,
                "weight_GLD": [other_weight] * 8,
                "weight_CASH": [other_weight] * 8,
            }
        )


if __name__ == "__main__":
    unittest.main()
