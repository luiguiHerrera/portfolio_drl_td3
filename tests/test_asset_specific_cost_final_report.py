"""Tests for asset-specific transaction cost final report."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.analysis.asset_specific_cost_final_report import (
    SCORE_COMPARABILITY_NOTE,
    build_asset_specific_cost_final_report,
)


class AssetSpecificCostFinalReportTests(unittest.TestCase):
    def test_build_report_combines_inputs_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v3_dir = self._write_input(
                root,
                "v3",
                base_candidate="V3_real_macro_vintage_clean_no_dxy",
                rows=[
                    ("V3_real_macro_vintage_clean_no_dxy_cap_uncapped", None, 0.30, 0.20),
                    ("V3_real_macro_vintage_clean_no_dxy_cap_0p50", 0.50, 0.70, 0.60),
                ],
            )
            v7_dir = self._write_input(
                root,
                "v7",
                base_candidate="V7_real_macro_vintage_clean_no_dxy_garch",
                rows=[
                    (
                        "V7_real_macro_vintage_clean_no_dxy_garch_cap_uncapped",
                        None,
                        0.40,
                        0.30,
                    ),
                    (
                        "V7_real_macro_vintage_clean_no_dxy_garch_cap_0p50",
                        0.50,
                        0.80,
                        0.65,
                    ),
                ],
            )
            v4_dir = self._write_input(
                root,
                "v4",
                base_candidate="V4_real_garch_current",
                rows=[
                    ("V4_real_garch_current_cap_uncapped", None, 0.20, 0.10),
                    ("V4_real_garch_current_cap_0p50", 0.50, 0.60, 0.55),
                ],
            )
            output_dir = root / "report"

            report = build_asset_specific_cost_final_report(
                v3_dir=str(v3_dir),
                v7_dir=str(v7_dir),
                v4_dir=str(v4_dir),
                output_dir=str(output_dir),
            )

            expected_files = {
                "asset_specific_cost_selected_candidates.csv",
                "asset_specific_cost_main_ranking.csv",
                "asset_specific_cost_summary.md",
                "asset_specific_cost_metadata.json",
            }
            self.assertTrue(expected_files.issubset({p.name for p in output_dir.iterdir()}))
            self.assertEqual(len(report["main_ranking"]), 6)
            self.assertEqual(
                report["main_ranking"].iloc[0]["strategy_name"],
                "V7_real_macro_vintage_clean_no_dxy_garch_cap_0p50",
            )
            self.assertIn(
                "asset_transaction_cost_contribution_BTC-USD",
                pd.read_csv(
                    v7_dir
                    / "per_candidate"
                    / "V7_real_macro_vintage_clean_no_dxy_garch"
                    / "F1_V7_real_macro_vintage_clean_no_dxy_garch_cap_0p50_seed_7"
                    / "test_policy_history.csv"
                ).columns,
            )
            self.assertEqual(
                report["main_ranking"].iloc[0]["transaction_cost_mode"],
                "asset_specific",
            )
            self.assertAlmostEqual(
                float(report["main_ranking"].iloc[0]["average_btc_cost_contribution"]),
                0.001,
            )
            self.assertEqual(
                report["main_ranking"].iloc[0]["score_comparability_note"],
                SCORE_COMPARABILITY_NOTE,
            )
            self.assertIn(
                "not the full original candidate universe",
                report["markdown_summary"],
            )
            self.assertIn(
                "asset_transaction_cost_bps",
                report["metadata"]["cost_assumptions"],
            )

    def test_selected_candidates_identifies_best_lenses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v3_dir = self._write_input(
                root,
                "v3",
                base_candidate="V3_real_macro_vintage_clean_no_dxy",
                rows=[
                    ("V3_real_macro_vintage_clean_no_dxy_cap_uncapped", None, 0.30, 0.20),
                    ("V3_real_macro_vintage_clean_no_dxy_cap_0p50", 0.50, 0.75, 0.70),
                ],
            )
            v7_dir = self._write_input(
                root,
                "v7",
                base_candidate="V7_real_macro_vintage_clean_no_dxy_garch",
                rows=[
                    (
                        "V7_real_macro_vintage_clean_no_dxy_garch_cap_uncapped",
                        None,
                        0.40,
                        0.30,
                    ),
                    (
                        "V7_real_macro_vintage_clean_no_dxy_garch_cap_0p50",
                        0.50,
                        0.60,
                        0.50,
                    ),
                ],
            )
            v4_dir = self._write_input(
                root,
                "v4",
                base_candidate="V4_real_garch_current",
                rows=[
                    ("V4_real_garch_current_cap_uncapped", None, 0.20, 0.10),
                    ("V4_real_garch_current_cap_0p50", 0.50, 0.65, 0.55),
                ],
            )

            report = build_asset_specific_cost_final_report(
                v3_dir=str(v3_dir),
                v7_dir=str(v7_dir),
                v4_dir=str(v4_dir),
                output_dir=str(root / "report"),
            )
            selected = report["selected_candidates"].set_index("selection")

            self.assertEqual(
                selected.loc["best_by_mandate_aware_score", "strategy_name"],
                "V3_real_macro_vintage_clean_no_dxy_cap_0p50",
            )
            self.assertEqual(
                selected.loc["best_by_robust_score", "strategy_name"],
                "V3_real_macro_vintage_clean_no_dxy_cap_0p50",
            )

    def _write_input(self, root, label, base_candidate, rows):
        directory = root / label
        directory.mkdir()
        candidate_output_dir = directory / "per_candidate" / base_candidate
        configs_dir = candidate_output_dir / "configs"
        configs_dir.mkdir(parents=True)
        records = []
        for candidate_name, cap, robust, mandate in rows:
            run_dir = candidate_output_dir / f"F1_{candidate_name}_seed_7"
            run_dir.mkdir(parents=True)
            self._write_history(run_dir / "test_policy_history.csv")
            self._write_config(configs_dir / f"F1_{candidate_name}_seed_7.yaml")
            records.append(
                {
                    "candidate_name": candidate_name,
                    "base_candidate": base_candidate,
                    "max_weight_cap": cap,
                    "split": "test",
                    "n_folds": 1,
                    "n_seeds": 1,
                    "episodes": 1,
                    "cumulative_return": 0.1,
                    "annualized_return": 0.1,
                    "annualized_volatility": 0.1,
                    "sharpe": robust,
                    "sortino": robust,
                    "calmar": robust,
                    "robust_score": robust,
                    "mandate_aware_score": mandate,
                    "max_drawdown": -0.1 - (0.01 if cap is None else 0.0),
                    "worst_max_drawdown": -0.2,
                    "average_turnover": 0.2 if cap is None else 0.1,
                    "mean_transaction_cost": 0.001,
                    "average_effective_number_of_assets": 1.0 if cap is None else 3.0,
                    "average_max_weight": 0.95 if cap is None else 0.5,
                    "mean_cash_weight": 0.1,
                    "cash_above_10_rate": 0.0,
                    "concentration_classification": "not_concentrated",
                    "suspicious_or_lazy_concentration_candidate": False,
                    "justified_concentration_candidate": False,
                    "decision_label": "test",
                    "candidate_output_dir": str(candidate_output_dir),
                    "cap_label": "uncapped" if cap is None else f"{cap:.2f}",
                }
            )
        pd.DataFrame(records).to_csv(directory / "cap_sensitivity_all_results.csv", index=False)
        (directory / "cap_sensitivity_metadata.json").write_text(
            json.dumps({"transaction_cost": 0.001}),
            encoding="utf-8",
        )
        return directory

    def _write_history(self, path):
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-05", periods=2, freq="W-FRI"),
                "financial_net_return": [0.01, 0.02],
                "asset_transaction_cost_contribution_BTC-USD": [0.001, 0.001],
                "weight_BTC-USD": [0.25, 0.35],
            }
        ).to_csv(path, index=False)

    def _write_config(self, path):
        config = {
            "environment": {
                "transaction_cost_mode": "asset_specific",
                "asset_transaction_cost_bps": {
                    "SPY": 2.0,
                    "TLT": 2.0,
                    "GLD": 2.0,
                    "BTC-USD": 10.0,
                    "CASH": 0.0,
                },
            },
        }
        path.write_text(yaml.safe_dump(config), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
