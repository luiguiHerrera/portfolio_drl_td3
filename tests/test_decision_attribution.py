import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.decision_attribution import (
    build_decision_attribution_report,
    compare_td3_to_rule_choices,
    compute_dominant_asset_regret,
    compute_forward_asset_returns,
    dominant_asset_from_policy,
)


class DecisionAttributionTest(unittest.TestCase):
    def test_dominant_asset_extraction_works_from_weight_columns(self):
        policy = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-05", periods=2, freq="W-FRI"),
                "weight_SPY": [0.7, 0.2],
                "weight_GLD": [0.3, 0.8],
            },
        )

        result = dominant_asset_from_policy(policy)

        self.assertEqual(result["dominant_asset"].tolist(), ["SPY", "GLD"])
        self.assertEqual(result["dominant_weight"].tolist(), [0.7, 0.8])

    def test_forward_returns_do_not_use_current_period_return(self):
        returns = pd.DataFrame(
            {"SPY": [0.10, 0.20, -0.10]},
            index=pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
        )

        result = compute_forward_asset_returns(returns, horizons=[1])
        first = result.loc[result["date"] == returns.index[0], "forward_return_SPY"].iloc[0]

        self.assertAlmostEqual(first, 0.20)

    def test_regret_zero_when_td3_chooses_future_best_asset(self):
        dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
        policy = pd.DataFrame(
            {"date": dates, "weight_SPY": [1, 1, 1, 1], "weight_GLD": [0, 0, 0, 0]},
        )
        returns = pd.DataFrame({"SPY": [0, 0.05, 0.04, 0.03], "GLD": [0, 0.01, 0.02, 0.01]}, index=dates)

        result = compute_dominant_asset_regret(policy, returns, horizons=[1])

        self.assertTrue((result["observations"]["regret"] == 0.0).all())
        self.assertEqual(result["summary"]["td3_best_asset_hit_rate"].iloc[0], 1.0)

    def test_regret_positive_when_td3_chooses_worse_asset(self):
        dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
        policy = pd.DataFrame(
            {"date": dates, "weight_SPY": [1, 1, 1, 1], "weight_GLD": [0, 0, 0, 0]},
        )
        returns = pd.DataFrame({"SPY": [0, 0.01, 0.01, 0.01], "GLD": [0, 0.05, 0.04, 0.03]}, index=dates)

        result = compute_dominant_asset_regret(policy, returns, horizons=[1])

        self.assertTrue((result["observations"]["regret"] > 0.0).all())
        self.assertEqual(result["summary"]["td3_best_asset_hit_rate"].iloc[0], 0.0)

    def test_td3_vs_rule_overlap_is_computed_correctly(self):
        returns = self._returns_for_rules()
        policy = pd.DataFrame(
            {
                "date": returns.index,
                "weight_SPY": [1.0] * len(returns),
                "weight_GLD": [0.0] * len(returns),
                "weight_CASH": [0.0] * len(returns),
            },
        )

        result = compare_td3_to_rule_choices(
            policy,
            returns,
            rule_name="momentum_winner_12p",
            horizons=[1],
        )

        self.assertEqual(result["summary"]["overlap_rate"].iloc[0], 1.0)

    def test_td3_vs_rule_forward_return_comparison_works(self):
        returns = self._returns_for_rules()
        policy = pd.DataFrame(
            {
                "date": returns.index,
                "weight_SPY": [0.0] * len(returns),
                "weight_GLD": [1.0] * len(returns),
                "weight_CASH": [0.0] * len(returns),
            },
        )

        result = compare_td3_to_rule_choices(
            policy,
            returns,
            rule_name="momentum_winner_12p",
            horizons=[1],
        )

        self.assertIn("mean_td3_minus_rule", result["summary"].columns)

    def test_handles_missing_cash_for_momentum_rule(self):
        dates = pd.date_range("2024-01-05", periods=16, freq="W-FRI")
        returns = pd.DataFrame({"SPY": [0.01] * 16, "GLD": [0.0] * 16}, index=dates)
        policy = pd.DataFrame({"date": dates, "weight_SPY": [1.0] * 16, "weight_GLD": [0.0] * 16})

        result = compare_td3_to_rule_choices(
            policy,
            returns,
            rule_name="momentum_winner_12p",
            horizons=[1],
        )

        self.assertFalse(result["summary"].empty)

    def test_handles_insufficient_horizon_at_end_of_sample(self):
        dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
        policy = pd.DataFrame(
            {"date": dates, "weight_SPY": [1, 1, 1], "weight_GLD": [0, 0, 0]},
        )
        returns = pd.DataFrame({"SPY": [0.01, 0.02, 0.03], "GLD": [0.0, 0.0, 0.0]}, index=dates)

        result = compute_dominant_asset_regret(policy, returns, horizons=[4])

        self.assertTrue(result["summary"].empty)

    def test_report_function_creates_expected_output_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            returns = self._returns_for_rules()
            returns_path = root / "returns.csv"
            returns.reset_index(names="date").to_csv(returns_path, index=False)
            run_dir = root / "F1_V2_reference_seed_7"
            run_dir.mkdir()
            pd.DataFrame(
                {
                    "date": returns.index,
                    "weight_SPY": [1.0] * len(returns),
                    "weight_GLD": [0.0] * len(returns),
                    "weight_CASH": [0.0] * len(returns),
                },
            ).to_csv(run_dir / "test_policy_history.csv", index=False)

            report = build_decision_attribution_report(
                str(root),
                returns_path=str(returns_path),
                strategies=["V2_reference"],
                horizons=[1],
                rules=["momentum_winner_12p"],
            )

            self.assertTrue(Path(report["dominant_asset_regret_summary_path"]).exists())
            self.assertTrue(Path(report["td3_vs_rule_choice_summary_path"]).exists())
            self.assertTrue(Path(report["dominant_asset_hit_rate_by_horizon_path"]).exists())
            self.assertTrue(Path(report["warnings_path"]).exists())

    def _returns_for_rules(self):
        dates = pd.date_range("2024-01-05", periods=20, freq="W-FRI")
        return pd.DataFrame(
            {
                "SPY": [0.01] * 20,
                "GLD": [0.0] * 20,
                "CASH": [0.0] * 20,
            },
            index=dates,
        )


if __name__ == "__main__":
    unittest.main()
