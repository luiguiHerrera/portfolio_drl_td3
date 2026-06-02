"""Tests for V5 CASH/risk-off attribution diagnostics."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.analysis.v5_cash_risk_off_attribution import (
    add_cash_risk_off_attribution,
    build_v5_cash_risk_off_report,
    load_policy_history,
    merge_policy_with_v5_regime,
    summarize_by_risk_off_state,
    summarize_cash_risk_off_attribution,
)


class V5CashRiskOffAttributionTests(unittest.TestCase):
    def test_load_policy_history_parses_and_sorts_dates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-12", "2024-01-05"],
                    "weight_CASH": [0.2, 0.1],
                }
            ).to_csv(path, index=False)

            result = load_policy_history(str(path))

        self.assertEqual(result["date"].tolist(), list(pd.to_datetime(["2024-01-05", "2024-01-12"])))

    def test_merge_policy_with_v5_regime_keeps_required_columns(self):
        merged = merge_policy_with_v5_regime(self._policy_history(), self._raw_v5_features())

        for column in (
            "regime_market_drawdown_stress",
            "regime_market_high_vol",
            "correlation_stress",
            "risk_off_score",
            "risk_off_state",
        ):
            self.assertIn(column, merged.columns)

    def test_merge_raises_if_required_v5_columns_are_missing(self):
        features = self._raw_v5_features().drop(columns=["risk_off_state"])

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            merge_policy_with_v5_regime(self._policy_history(), features)

    def test_add_cash_risk_off_attribution_flags_high_cash(self):
        attributed = add_cash_risk_off_attribution(self._merged_policy_features())

        self.assertTrue(attributed.loc[0, "high_cash"])
        self.assertFalse(attributed.loc[1, "high_cash"])

    def test_high_cash_during_risk_off_is_justified(self):
        attributed = add_cash_risk_off_attribution(self._merged_policy_features())

        self.assertTrue(attributed.loc[0, "high_cash_and_risk_off"])
        self.assertEqual(attributed.loc[0, "unjustified_cash_excess"], 0.0)
        self.assertGreater(attributed.loc[0, "risk_off_justified_cash_excess"], 0.0)

    def test_high_cash_outside_risk_off_is_unjustified(self):
        merged = self._merged_policy_features()
        merged.loc[1, "weight_CASH"] = 0.7
        attributed = add_cash_risk_off_attribution(merged)

        self.assertTrue(attributed.loc[1, "high_cash_without_risk_off"])
        self.assertGreater(attributed.loc[1, "unjustified_cash_excess"], 0.0)
        self.assertEqual(attributed.loc[1, "risk_off_justified_cash_excess"], 0.0)

    def test_summarize_computes_share_high_cash_observations_in_risk_off(self):
        merged = self._merged_policy_features()
        merged.loc[1, "weight_CASH"] = 0.7
        merged.loc[2, "weight_CASH"] = 0.0
        attributed = add_cash_risk_off_attribution(merged)

        summary = summarize_cash_risk_off_attribution(attributed)

        self.assertAlmostEqual(
            summary.loc[0, "share_high_cash_observations_in_risk_off"],
            0.5,
        )

    def test_summarize_handles_no_high_cash_without_division_error(self):
        merged = self._merged_policy_features()
        merged["weight_CASH"] = 0.0
        attributed = add_cash_risk_off_attribution(merged)

        summary = summarize_cash_risk_off_attribution(attributed)

        self.assertEqual(summary.loc[0, "share_high_cash_observations_in_risk_off"], 0.0)

    def test_summarize_by_risk_off_state_groups_correctly(self):
        attributed = add_cash_risk_off_attribution(self._merged_policy_features())

        result = summarize_by_risk_off_state(attributed)

        self.assertEqual(set(result["risk_off_state"]), {0.0, 1.0})
        self.assertIn("mean_cash_weight", result.columns)

    def test_build_report_handles_multiple_strategies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = self._write_policy_histories(temp_dir)
            with patch(
                "src.analysis.v5_cash_risk_off_attribution.build_raw_v5_features_for_returns",
                return_value=self._raw_v5_features(),
            ):
                result = build_v5_cash_risk_off_report(
                    policy_history_paths=paths,
                    returns_path="returns.csv",
                    strategy_names=["s1", "s2"],
                )

        self.assertEqual(set(result["summary"]["strategy_name"]), {"s1", "s2"})
        self.assertEqual(set(result["observations"]["strategy_name"]), {"s1", "s2"})

    def test_build_report_saves_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = self._write_policy_histories(temp_dir)
            output_dir = Path(temp_dir) / "report"
            with patch(
                "src.analysis.v5_cash_risk_off_attribution.build_raw_v5_features_for_returns",
                return_value=self._raw_v5_features(),
            ):
                result = build_v5_cash_risk_off_report(
                    policy_history_paths=paths,
                    returns_path="returns.csv",
                    strategy_names=["s1", "s2"],
                    output_dir=str(output_dir),
                )
                self.assertTrue(Path(result["observations_path"]).exists())
                self.assertTrue(Path(result["summary_path"]).exists())
                self.assertTrue(Path(result["by_risk_off_state_path"]).exists())

    def test_input_dataframes_are_not_mutated(self):
        merged = self._merged_policy_features()
        original = merged.copy(deep=True)

        add_cash_risk_off_attribution(merged)

        pd.testing.assert_frame_equal(merged, original)

    def test_invalid_normal_cash_max_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "normal_cash_max"):
            add_cash_risk_off_attribution(self._merged_policy_features(), normal_cash_max=1.5)

    @staticmethod
    def _policy_history() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"]),
                "weight_SPY": [0.2, 0.8, 0.6],
                "weight_CASH": [0.8, 0.05, 0.2],
            }
        )

    @staticmethod
    def _raw_v5_features() -> pd.DataFrame:
        index = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"])
        index.name = "date"

        return pd.DataFrame(
            {
                "regime_market_drawdown_stress": [1.0, 0.0, 0.0],
                "regime_market_high_vol": [0.0, 0.0, 1.0],
                "correlation_stress": [0.0, 0.0, 1.0],
                "risk_off_score": [2.0, 0.0, 2.0],
                "risk_off_state": [1.0, 0.0, 1.0],
            },
            index=index,
        )

    def _merged_policy_features(self) -> pd.DataFrame:
        return merge_policy_with_v5_regime(self._policy_history(), self._raw_v5_features())

    def _write_policy_histories(self, directory: str) -> list[str]:
        paths = []
        for name in ("s1", "s2"):
            path = Path(directory) / f"{name}.csv"
            self._policy_history().to_csv(path, index=False)
            paths.append(str(path))

        return paths


if __name__ == "__main__":
    unittest.main()
