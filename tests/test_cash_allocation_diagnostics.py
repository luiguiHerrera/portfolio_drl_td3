"""Tests for cash allocation diagnostics."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.cash_allocation_diagnostics import (
    add_cash_state_diagnostics,
    attach_forward_cash_opportunity_cost,
    build_cash_allocation_report,
    compute_risk_off_score,
    infer_cash_weight_column,
    summarize_cash_allocation_diagnostics,
)


class CashAllocationDiagnosticsTests(unittest.TestCase):
    def test_infer_cash_weight_column_finds_weight_cash(self):
        self.assertEqual(infer_cash_weight_column(self._policy_history()), "weight_CASH")

    def test_infer_cash_weight_column_raises_when_missing(self):
        data = self._policy_history().drop(columns=["weight_CASH"])

        with self.assertRaisesRegex(ValueError, "weight_CASH"):
            infer_cash_weight_column(data)

    def test_compute_risk_off_score_sums_signal_columns(self):
        data = pd.DataFrame(
            {
                "market_risk_off_regime": [1.0, 0.0],
                "macro_high_vix_regime": [True, False],
            }
        )

        result = compute_risk_off_score(
            data,
            ["market_risk_off_regime", "macro_high_vix_regime"],
        )

        self.assertEqual(result.tolist(), [2.0, 0.0])

    def test_compute_risk_off_score_raises_missing_signal_column(self):
        with self.assertRaisesRegex(ValueError, "Missing risk-off signal"):
            compute_risk_off_score(pd.DataFrame({"a": [1.0]}), ["missing"])

    def test_add_cash_state_diagnostics_flags_cash_above_normal_max(self):
        result = add_cash_state_diagnostics(self._policy_history(), normal_cash_max=0.10)

        self.assertFalse(result.loc[0, "cash_above_normal_max"])
        self.assertTrue(result.loc[1, "cash_above_normal_max"])

    def test_add_cash_state_diagnostics_allows_high_cash_when_risk_off_true(self):
        data = self._policy_history()
        data["risk_off_score"] = [0.0, 1.0, 0.0, 0.0]

        result = add_cash_state_diagnostics(
            data,
            normal_cash_max=0.10,
            risk_off_score_column="risk_off_score",
        )

        self.assertTrue(result.loc[1, "cash_allowed_by_state"])

    def test_unjustified_cash_excess_is_zero_during_risk_off(self):
        data = self._policy_history()
        data["risk_off_score"] = [0.0, 1.0, 0.0, 0.0]

        result = add_cash_state_diagnostics(
            data,
            normal_cash_max=0.10,
            risk_off_score_column="risk_off_score",
        )

        self.assertEqual(result.loc[1, "unjustified_cash_excess"], 0.0)

    def test_unjustified_cash_excess_positive_outside_risk_off(self):
        result = add_cash_state_diagnostics(self._policy_history(), normal_cash_max=0.10)

        self.assertAlmostEqual(result.loc[1, "unjustified_cash_excess"], 0.60)

    def test_attach_forward_cash_opportunity_cost_horizon_one(self):
        cash_state = add_cash_state_diagnostics(self._policy_history(), normal_cash_max=0.10)

        result = attach_forward_cash_opportunity_cost(
            cash_state,
            self._asset_returns(),
            horizon=1,
        )

        first_row = result.loc[result["date"] == pd.Timestamp("2024-01-05")].iloc[0]
        self.assertAlmostEqual(first_row["cash_forward_return"], 0.0)
        self.assertAlmostEqual(
            first_row["equal_weight_forward_return"],
            (0.04 - 0.02 + 0.0) / 3.0,
        )

    def test_attach_forward_cash_opportunity_cost_excludes_cash_from_best_risky(self):
        cash_state = add_cash_state_diagnostics(self._policy_history(), normal_cash_max=0.10)

        result = attach_forward_cash_opportunity_cost(
            cash_state,
            self._asset_returns(),
            horizon=1,
        )

        first_row = result.loc[result["date"] == pd.Timestamp("2024-01-05")].iloc[0]
        self.assertAlmostEqual(first_row["best_risky_asset_forward_return"], 0.04)

    def test_summarize_cash_allocation_diagnostics_returns_expected_columns(self):
        observations = self._cash_observations()

        summary = summarize_cash_allocation_diagnostics(observations)

        expected_columns = {
            "n_observations",
            "mean_cash_weight",
            "max_cash_weight",
            "cash_above_normal_rate",
            "risk_off_rate",
            "cash_allowed_rate",
            "unjustified_cash_rate",
            "mean_unjustified_cash_excess",
            "mean_cash_excess_normal",
            "mean_cash_forward_return",
            "mean_equal_weight_forward_return",
            "mean_best_risky_asset_forward_return",
            "mean_cash_excess_vs_equal_weight",
            "mean_cash_excess_vs_best_risky_asset",
            "cash_underperforms_equal_weight_rate",
            "cash_underperforms_best_risky_asset_rate",
        }
        self.assertTrue(expected_columns.issubset(summary.columns))

    def test_build_cash_allocation_report_handles_multiple_horizons(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy_history(temp_dir)
            returns_path = self._write_asset_returns(temp_dir)

            report = build_cash_allocation_report(
                [policy_path],
                asset_returns_path=returns_path,
                strategy_names=["policy"],
                horizons=[1, 4],
            )

        self.assertEqual(set(report["summary"]["horizon"]), {1, 4})
        self.assertEqual(set(report["summary"]["strategy_name"]), {"policy"})

    def test_build_cash_allocation_report_saves_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = self._write_policy_history(temp_dir)
            returns_path = self._write_asset_returns(temp_dir)
            output_dir = Path(temp_dir) / "cash_report"

            report = build_cash_allocation_report(
                [policy_path],
                asset_returns_path=returns_path,
                horizons=[1],
                output_dir=str(output_dir),
            )

            self.assertTrue(Path(report["observations_path"]).exists())
            self.assertTrue(Path(report["summary_path"]).exists())

    def test_invalid_normal_cash_max_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "normal_cash_max"):
            add_cash_state_diagnostics(self._policy_history(), normal_cash_max=1.5)

    def test_input_dataframes_are_not_mutated(self):
        policy = self._policy_history()
        returns = self._asset_returns()
        policy_original = policy.copy(deep=True)
        returns_original = returns.copy(deep=True)

        cash_state = add_cash_state_diagnostics(policy, normal_cash_max=0.10)
        attach_forward_cash_opportunity_cost(cash_state, returns, horizon=1)

        pd.testing.assert_frame_equal(policy, policy_original)
        pd.testing.assert_frame_equal(returns, returns_original)

    def _cash_observations(self) -> pd.DataFrame:
        cash_state = add_cash_state_diagnostics(self._policy_history(), normal_cash_max=0.10)

        return attach_forward_cash_opportunity_cost(
            cash_state,
            self._asset_returns(),
            horizon=1,
        )

    def _policy_history(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [
                    "2024-01-05",
                    "2024-01-12",
                    "2024-01-19",
                    "2024-01-26",
                ],
                "weight_SPY": [0.8, 0.2, 0.5, 0.4],
                "weight_GLD": [0.1, 0.1, 0.4, 0.5],
                "weight_CASH": [0.1, 0.7, 0.1, 0.1],
            }
        )

    def _asset_returns(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
        return pd.DataFrame(
            {
                "SPY": [0.00, 0.04, -0.01, 0.02, 0.03, 0.01],
                "GLD": [0.00, -0.02, 0.03, 0.01, -0.01, 0.02],
                "CASH": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            },
            index=dates,
        )

    def _write_policy_history(self, directory: str) -> str:
        path = Path(directory) / "policy_history.csv"
        self._policy_history().to_csv(path, index=False)

        return str(path)

    def _write_asset_returns(self, directory: str) -> str:
        path = Path(directory) / "asset_returns.csv"
        self._asset_returns().reset_index(names="date").to_csv(path, index=False)

        return str(path)


if __name__ == "__main__":
    unittest.main()
