"""Tests for policy behavior diagnostics."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.policy_behavior_diagnostics import (
    add_policy_state_columns,
    build_policy_behavior_report,
    infer_weight_columns,
    summarize_asset_conditional_performance,
    summarize_dominant_asset_distribution,
    summarize_dominant_asset_transitions,
    summarize_holding_periods,
    summarize_policy_concentration,
    summarize_regime_attribution,
)


class PolicyBehaviorDiagnosticsTests(unittest.TestCase):
    def test_infer_weight_columns_detects_numeric_weight_columns(self):
        columns = infer_weight_columns(self._policy_data())

        self.assertEqual(
            columns,
            [
                "weight_SPY",
                "weight_TLT",
                "weight_GLD",
                "weight_BTC-USD",
                "weight_CASH",
            ],
        )

    def test_infer_weight_columns_excludes_boolean_columns_and_ok_columns(self):
        data = self._policy_data()
        data["weight_signal"] = True
        data["weight_SPY_ok"] = True

        columns = infer_weight_columns(data)

        self.assertNotIn("weight_signal", columns)
        self.assertNotIn("weight_SPY_ok", columns)

    def test_infer_weight_columns_raises_value_error_without_valid_weights(self):
        data = pd.DataFrame({"weight_ok": [True], "not_weight": [1.0]})

        with self.assertRaises(ValueError):
            infer_weight_columns(data)

    def test_add_policy_state_columns_adds_dominant_asset_and_weight(self):
        result = add_policy_state_columns(self._policy_data())

        self.assertEqual(result.iloc[0]["dominant_asset"], "SPY")
        self.assertEqual(result.iloc[2]["dominant_asset"], "GLD")
        self.assertEqual(result.iloc[4]["dominant_asset"], "BTC-USD")
        self.assertEqual(result.iloc[6]["dominant_asset"], "CASH")
        self.assertEqual(result.iloc[0]["dominant_weight"], 0.85)

    def test_add_policy_state_columns_computes_herfindahl_and_effective_number(self):
        result = add_policy_state_columns(self._policy_data())
        expected_herfindahl = 0.85**2 + 0.05**2 + 0.05**2 + 0.03**2 + 0.02**2

        self.assertAlmostEqual(result.iloc[0]["herfindahl_index"], expected_herfindahl)
        self.assertAlmostEqual(
            result.iloc[0]["effective_number_of_assets"],
            1.0 / expected_herfindahl,
        )

    def test_add_policy_state_columns_adds_high_concentration_flags(self):
        result = add_policy_state_columns(self._policy_data())

        self.assertTrue(result.iloc[0]["is_highly_concentrated_80"])
        self.assertFalse(result.iloc[0]["is_highly_concentrated_90"])
        self.assertTrue(result.iloc[3]["is_highly_concentrated_90"])

    def test_add_policy_state_columns_adds_weight_sum_without_normalizing(self):
        data = self._policy_data()
        data.loc[data.index[0], "weight_SPY"] = 0.90

        result = add_policy_state_columns(data)

        self.assertAlmostEqual(result.iloc[0]["weight_sum"], 1.05)
        self.assertAlmostEqual(result.iloc[0]["weight_SPY"], 0.90)

    def test_add_policy_state_columns_raises_value_error_on_zero_weight_row(self):
        data = self._policy_data()
        weight_columns = infer_weight_columns(data)
        data.loc[data.index[0], weight_columns] = 0.0

        with self.assertRaises(ValueError):
            add_policy_state_columns(data)

    def test_summarize_policy_concentration_returns_expected_columns_and_values(self):
        result = summarize_policy_concentration(self._policy_data())

        self.assertEqual(result.loc[0, "n_observations"], 8)
        self.assertIn("mean_dominant_weight", result.columns)
        self.assertIn("high_concentration_80_rate", result.columns)
        self.assertAlmostEqual(result.loc[0, "high_concentration_80_rate"], 7 / 8)
        self.assertAlmostEqual(result.loc[0, "mean_weight_sum"], 1.0)

    def test_summarize_dominant_asset_distribution_returns_correct_counts_and_rates(self):
        result = summarize_dominant_asset_distribution(self._policy_data())
        counts = dict(zip(result["dominant_asset"], result["count"]))
        rates = dict(zip(result["dominant_asset"], result["rate"]))

        self.assertEqual(counts["SPY"], 2)
        self.assertEqual(counts["GLD"], 2)
        self.assertEqual(counts["BTC-USD"], 2)
        self.assertEqual(counts["CASH"], 2)
        self.assertAlmostEqual(rates["SPY"], 0.25)

    def test_summarize_dominant_asset_transitions_returns_switch_counts(self):
        result = summarize_dominant_asset_transitions(self._policy_data())
        transitions = {
            (row["from_asset"], row["to_asset"]): row["count"]
            for _, row in result.iterrows()
        }

        self.assertEqual(transitions[("SPY", "GLD")], 1)
        self.assertEqual(transitions[("GLD", "BTC-USD")], 1)
        self.assertEqual(transitions[("BTC-USD", "CASH")], 1)
        self.assertAlmostEqual(result["rate_of_all_switches"].sum(), 1.0)

    def test_summarize_dominant_asset_transitions_returns_empty_frame_without_switches(self):
        data = self._policy_data().iloc[:2]

        result = summarize_dominant_asset_transitions(data)

        self.assertTrue(result.empty)
        self.assertEqual(
            list(result.columns),
            ["from_asset", "to_asset", "count", "rate_of_all_switches"],
        )

    def test_summarize_holding_periods_returns_correct_run_lengths(self):
        result = summarize_holding_periods(self._policy_data())

        self.assertEqual(result["dominant_asset"].tolist(), ["SPY", "GLD", "BTC-USD", "CASH"])
        self.assertEqual(result["holding_period_length"].tolist(), [2, 2, 2, 2])

    def test_summarize_holding_periods_preserves_datetime_index_values(self):
        result = summarize_holding_periods(self._policy_data())

        self.assertEqual(result.loc[0, "start_index"], pd.Timestamp("2024-01-05"))
        self.assertEqual(result.loc[0, "end_index"], pd.Timestamp("2024-01-12"))

    def test_summarize_asset_conditional_performance_computes_metrics(self):
        result = summarize_asset_conditional_performance(
            self._policy_data(),
            return_column="portfolio_return",
        )
        spy = result.loc[result["dominant_asset"] == "SPY"].iloc[0]

        self.assertEqual(spy["n_observations"], 2)
        self.assertAlmostEqual(spy["mean_return"], 0.015)
        self.assertAlmostEqual(spy["cumulative_return"], (1.01 * 1.02) - 1.0)
        self.assertAlmostEqual(spy["hit_rate"], 1.0)
        self.assertGreater(spy["volatility"], 0.0)

    def test_summarize_asset_conditional_performance_raises_key_error_for_missing_return(self):
        with self.assertRaises(KeyError):
            summarize_asset_conditional_performance(
                self._policy_data(),
                return_column="missing_return",
            )

    def test_summarize_regime_attribution_computes_mean_regimes_by_asset(self):
        result = summarize_regime_attribution(
            self._policy_data(),
            regime_columns=[
                "market_high_vol_regime",
                "market_risk_off_regime",
                "macro_high_vix_regime",
                "macro_inflation_pressure_regime",
            ],
        )
        gld = result.loc[result["dominant_asset"] == "GLD"].iloc[0]

        self.assertEqual(gld["n_observations"], 2)
        self.assertAlmostEqual(gld["mean_market_high_vol_regime"], 1.0)
        self.assertAlmostEqual(gld["mean_market_risk_off_regime"], 1.0)

    def test_summarize_regime_attribution_raises_key_error_for_missing_regime(self):
        with self.assertRaises(KeyError):
            summarize_regime_attribution(
                self._policy_data(),
                regime_columns=["missing_regime"],
            )

    def test_summarize_regime_attribution_raises_value_error_for_non_numeric_regime(self):
        data = self._policy_data()
        data["regime_label"] = "risk_off"

        with self.assertRaises(ValueError):
            summarize_regime_attribution(data, regime_columns=["regime_label"])

    def test_build_policy_behavior_report_returns_all_expected_keys(self):
        result = build_policy_behavior_report(
            self._policy_data(),
            return_column="portfolio_return",
            regime_columns=["market_high_vol_regime"],
        )

        expected_keys = [
            "observations",
            "concentration_summary",
            "dominant_asset_distribution",
            "dominant_asset_transitions",
            "holding_periods",
            "conditional_performance",
            "regime_attribution",
            "observations_path",
            "concentration_summary_path",
            "dominant_asset_distribution_path",
            "dominant_asset_transitions_path",
            "holding_periods_path",
            "conditional_performance_path",
            "regime_attribution_path",
        ]
        self.assertEqual(set(result.keys()), set(expected_keys))
        self.assertIsNotNone(result["conditional_performance"])
        self.assertIsNotNone(result["regime_attribution"])

    def test_build_policy_behavior_report_saves_csv_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = build_policy_behavior_report(
                self._policy_data(),
                return_column="portfolio_return",
                regime_columns=["market_high_vol_regime"],
                output_dir=temp_dir,
                report_name="behavior",
            )

            for path_key in [
                "observations_path",
                "concentration_summary_path",
                "dominant_asset_distribution_path",
                "dominant_asset_transitions_path",
                "holding_periods_path",
                "conditional_performance_path",
                "regime_attribution_path",
            ]:
                self.assertTrue(Path(result[path_key]).exists())

    def test_build_policy_behavior_report_returns_none_optional_outputs_when_omitted(self):
        result = build_policy_behavior_report(self._policy_data())

        self.assertIsNone(result["conditional_performance"])
        self.assertIsNone(result["regime_attribution"])
        self.assertIsNone(result["conditional_performance_path"])
        self.assertIsNone(result["regime_attribution_path"])

    def _policy_data(self):
        index = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
        return pd.DataFrame(
            {
                "weight_SPY": [0.85, 0.82, 0.05, 0.02, 0.05, 0.02, 0.05, 0.05],
                "weight_TLT": [0.05, 0.08, 0.05, 0.03, 0.05, 0.03, 0.10, 0.05],
                "weight_GLD": [0.05, 0.05, 0.82, 0.91, 0.05, 0.04, 0.05, 0.05],
                "weight_BTC-USD": [0.03, 0.03, 0.05, 0.02, 0.83, 0.88, 0.05, 0.05],
                "weight_CASH": [0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.75, 0.80],
                "portfolio_return": [0.01, 0.02, -0.01, 0.03, 0.04, -0.02, 0.00, 0.01],
                "market_high_vol_regime": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                "market_risk_off_regime": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                "macro_high_vix_regime": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                "macro_inflation_pressure_regime": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            },
            index=index,
        )


if __name__ == "__main__":
    unittest.main()
