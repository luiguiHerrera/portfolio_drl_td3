"""Tests for concentration quality diagnostics."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.concentration_quality_diagnostics import (
    add_dominant_asset_state,
    attach_forward_dominant_asset_performance,
    build_concentration_quality_report,
    compute_forward_asset_returns,
    infer_weight_columns,
    summarize_concentration_quality,
    summarize_concentration_quality_by_asset,
)


class ConcentrationQualityDiagnosticsTests(unittest.TestCase):
    def test_infer_weight_columns_excludes_bool_and_ok_columns(self):
        data = pd.DataFrame(
            {
                "weight_SPY": [0.7],
                "weight_GLD": [0.3],
                "weight_ok": [True],
                "weight_flag": [False],
                "not_weight": [1.0],
            }
        )

        result = infer_weight_columns(data)

        self.assertEqual(result, ["weight_SPY", "weight_GLD"])

    def test_add_dominant_asset_state_identifies_dominant_asset(self):
        result = add_dominant_asset_state(self._policy_history())

        self.assertEqual(result.loc[0, "dominant_asset"], "GLD")
        self.assertEqual(result.loc[1, "dominant_asset"], "SPY")
        self.assertAlmostEqual(result.loc[0, "dominant_weight"], 0.9)

    def test_effective_number_of_assets_is_one_for_all_in_and_two_for_half_half(self):
        data = pd.DataFrame(
            {
                "date": ["2024-01-05", "2024-01-12"],
                "weight_SPY": [1.0, 0.5],
                "weight_GLD": [0.0, 0.5],
            }
        )

        result = add_dominant_asset_state(data)

        self.assertAlmostEqual(result.loc[0, "effective_number_of_assets"], 1.0)
        self.assertAlmostEqual(result.loc[1, "effective_number_of_assets"], 2.0)

    def test_compute_forward_asset_returns_horizon_one_uses_next_period(self):
        returns = self._asset_returns()

        result = compute_forward_asset_returns(returns, horizon=1)

        self.assertAlmostEqual(
            result.loc[pd.Timestamp("2024-01-05"), "SPY"],
            returns.loc[pd.Timestamp("2024-01-12"), "SPY"],
        )

    def test_compute_forward_asset_returns_horizon_four_compounds_next_four(self):
        returns = self._asset_returns()

        result = compute_forward_asset_returns(returns, horizon=4)

        expected = (
            (1.0 + returns.loc["2024-01-12":"2024-02-02", "SPY"]).prod()
            - 1.0
        )
        self.assertAlmostEqual(result.loc[pd.Timestamp("2024-01-05"), "SPY"], expected)

    def test_attach_forward_dominant_asset_performance_adds_expected_columns(self):
        result = attach_forward_dominant_asset_performance(
            self._policy_history(),
            self._asset_returns(),
            horizon=1,
        )

        expected_columns = {
            "dominant_forward_return",
            "best_asset_forward_return",
            "worst_asset_forward_return",
            "equal_weight_forward_return",
            "dominant_asset_rank",
            "dominant_is_best_asset",
            "dominant_beats_equal_weight",
            "dominant_forward_excess_vs_equal_weight",
            "dominant_forward_excess_vs_best_asset",
        }
        self.assertTrue(expected_columns.issubset(result.columns))

    def test_dominant_is_best_asset_true_when_future_winner(self):
        result = attach_forward_dominant_asset_performance(
            self._policy_history(),
            self._asset_returns(),
            horizon=1,
        )

        first_row = result.loc[result["date"] == pd.Timestamp("2024-01-05")].iloc[0]
        self.assertTrue(first_row["dominant_is_best_asset"])

    def test_dominant_beats_equal_weight(self):
        result = attach_forward_dominant_asset_performance(
            self._policy_history(),
            self._asset_returns(),
            horizon=1,
        )

        first_row = result.loc[result["date"] == pd.Timestamp("2024-01-05")].iloc[0]
        self.assertTrue(first_row["dominant_beats_equal_weight"])

    def test_dominant_asset_rank(self):
        result = attach_forward_dominant_asset_performance(
            self._policy_history(),
            self._asset_returns(),
            horizon=1,
        )

        first_row = result.loc[result["date"] == pd.Timestamp("2024-01-05")].iloc[0]
        second_row = result.loc[result["date"] == pd.Timestamp("2024-01-12")].iloc[0]
        self.assertEqual(first_row["dominant_asset_rank"], 1)
        self.assertEqual(second_row["dominant_asset_rank"], 3)

    def test_unavailable_forward_returns_are_dropped(self):
        result = attach_forward_dominant_asset_performance(
            self._policy_history(),
            self._asset_returns(),
            horizon=1,
        )

        self.assertNotIn(pd.Timestamp("2024-02-02"), set(result["date"]))

    def test_summarize_concentration_quality_returns_expected_columns(self):
        observations = attach_forward_dominant_asset_performance(
            self._policy_history(),
            self._asset_returns(),
            horizon=1,
        )

        summary = summarize_concentration_quality(observations)

        expected_columns = {
            "n_observations",
            "mean_dominant_weight",
            "mean_effective_number_of_assets",
            "dominant_best_asset_rate",
            "dominant_beats_equal_weight_rate",
            "mean_dominant_forward_return",
            "mean_equal_weight_forward_return",
            "mean_best_asset_forward_return",
            "mean_dominant_forward_excess_vs_equal_weight",
            "mean_dominant_forward_excess_vs_best_asset",
            "mean_dominant_asset_rank",
        }
        self.assertTrue(expected_columns.issubset(summary.columns))

    def test_summarize_concentration_quality_by_asset_groups_by_dominant_asset(self):
        observations = attach_forward_dominant_asset_performance(
            self._policy_history(),
            self._asset_returns(),
            horizon=1,
        )

        by_asset = summarize_concentration_quality_by_asset(observations)

        self.assertEqual(
            set(by_asset["dominant_asset"]),
            {"SPY", "GLD", "CASH"},
        )

    def test_build_concentration_quality_report_handles_multiple_horizons(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy_history(temp_dir)
            returns_path = self._write_asset_returns(temp_dir)

            report = build_concentration_quality_report(
                [policy_path],
                asset_returns_path=returns_path,
                strategy_names=["test_policy"],
                horizons=[1, 4],
            )

        self.assertEqual(set(report["summary"]["horizon"]), {1, 4})
        self.assertEqual(set(report["observations"]["strategy_name"]), {"test_policy"})

    def test_build_concentration_quality_report_saves_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy_history(temp_dir)
            returns_path = self._write_asset_returns(temp_dir)
            output_dir = Path(temp_dir) / "report"

            report = build_concentration_quality_report(
                [policy_path],
                asset_returns_path=returns_path,
                horizons=[1],
                output_dir=str(output_dir),
            )

            self.assertTrue(Path(report["observations_path"]).exists())
            self.assertTrue(Path(report["summary_path"]).exists())
            self.assertTrue(Path(report["by_asset_summary_path"]).exists())

    def test_input_dataframes_are_not_mutated(self):
        policy = self._policy_history()
        returns = self._asset_returns()
        policy_original = policy.copy(deep=True)
        returns_original = returns.copy(deep=True)

        attach_forward_dominant_asset_performance(policy, returns, horizon=1)

        pd.testing.assert_frame_equal(policy, policy_original)
        pd.testing.assert_frame_equal(returns, returns_original)

    def test_missing_date_column_raises_value_error(self):
        policy = self._policy_history().drop(columns=["date"])

        with self.assertRaisesRegex(ValueError, "date"):
            attach_forward_dominant_asset_performance(
                policy,
                self._asset_returns(),
                horizon=1,
            )

    def test_missing_asset_return_columns_raises_value_error(self):
        returns = self._asset_returns().drop(columns=["GLD"])

        with self.assertRaisesRegex(ValueError, "Missing asset return columns"):
            attach_forward_dominant_asset_performance(
                self._policy_history(),
                returns,
                horizon=1,
            )

    def test_invalid_horizon_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "horizon"):
            compute_forward_asset_returns(self._asset_returns(), horizon=0)

    def _policy_history(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [
                    "2024-01-05",
                    "2024-01-12",
                    "2024-01-19",
                    "2024-01-26",
                    "2024-02-02",
                ],
                "portfolio_return": [0.0, 0.01, -0.01, 0.02, 0.0],
                "weight_SPY": [0.1, 0.8, 0.8, 0.0, 0.5],
                "weight_GLD": [0.9, 0.1, 0.1, 0.0, 0.5],
                "weight_CASH": [0.0, 0.1, 0.1, 1.0, 0.0],
                "weight_ok": [True, True, True, True, True],
            }
        )

    def _asset_returns(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
        return pd.DataFrame(
            {
                "SPY": [0.00, 0.01, -0.01, 0.04, 0.01],
                "GLD": [0.00, 0.03, 0.02, 0.01, -0.02],
                "CASH": [0.00, 0.00, 0.00, 0.00, 0.00],
            },
            index=dates,
        )

    def _write_policy_history(self, directory: str) -> str:
        path = Path(directory) / "policy_history.csv"
        self._policy_history().to_csv(path, index=False)

        return str(path)

    def _write_asset_returns(self, directory: str) -> str:
        path = Path(directory) / "asset_returns.csv"
        returns = self._asset_returns().reset_index(names="date")
        returns.to_csv(path, index=False)

        return str(path)


if __name__ == "__main__":
    unittest.main()
