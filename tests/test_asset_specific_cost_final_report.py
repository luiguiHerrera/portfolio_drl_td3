"""Tests for asset-specific transaction cost final report."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

import src.analysis.asset_specific_cost_final_report as report_module
from src.analysis.asset_specific_cost_final_report import (
    SCORE_COMPARABILITY_NOTE,
    build_asset_specific_cost_final_report,
    build_official_asset_specific_cost_full_report,
    collect_run_records,
    parse_run_folder_name,
    read_csv_with_retry,
    validate_full_coverage,
)


class AssetSpecificCostFinalReportTests(unittest.TestCase):
    def test_parse_run_folder_names(self):
        result = parse_run_folder_name("F4_V6_financial_state_cap_uncapped_seed_505")

        self.assertEqual(result["fold"], "F4")
        self.assertEqual(result["candidate"], "V6_financial_state")
        self.assertEqual(result["cap"], "uncapped")
        self.assertEqual(result["seed"], 505)
        self.assertTrue(pd.isna(result["max_weight_cap"]))

        capped = parse_run_folder_name("F1_V2_reference_full_cap_0p50_seed_101")
        self.assertEqual(capped["cap_label"], "0.50")
        self.assertEqual(capped["max_weight_cap"], 0.50)

    def test_parse_run_folder_rejects_bad_name(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized run folder"):
            parse_run_folder_name("configs")

    def test_collect_records_ignores_configs_and_detects_missing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            candidate_dir = source / "per_candidate" / "V2_reference_full"
            (candidate_dir / "configs").mkdir(parents=True)
            run_dir = candidate_dir / "F1_V2_reference_full_cap_uncapped_seed_7"
            run_dir.mkdir()
            self._write_history(run_dir / "test_policy_history.csv")
            self._write_config(run_dir / "test_metrics_table.csv")

            with self.assertRaises(FileNotFoundError):
                collect_run_records(
                    {"source": source},
                    read_log=[],
                    ignored_dirs=[],
                )

    def test_validate_full_coverage_detects_missing_histories(self):
        with mock.patch.object(report_module, "EXPECTED_FULL_CANDIDATES", ["A"]), \
            mock.patch.object(report_module, "EXPECTED_CAP_LABELS", ["uncapped"]), \
            mock.patch.object(report_module, "EXPECTED_FOLDS", ["F1"]), \
            mock.patch.object(report_module, "EXPECTED_SEEDS", [7]):
            with self.assertRaisesRegex(ValueError, "Expected 1 histories"):
                validate_full_coverage([])

    def test_safe_csv_reader_retries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.csv"
            path.write_text("a\n1\n", encoding="utf-8")
            calls = {"n": 0}
            original = pd.read_csv

            def flaky_read_csv(*args, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise TimeoutError("temporary")
                return original(*args, **kwargs)

            with mock.patch("pandas.read_csv", side_effect=flaky_read_csv):
                frame = read_csv_with_retry(path, retries=2, sleep_seconds=0.0)

            self.assertEqual(calls["n"], 2)
            self.assertEqual(frame["a"].iloc[0], 1)

    def test_official_builder_combines_sources_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v2_v6 = root / "v2_v6"
            v7 = root / "v7"
            v8 = root / "v8"
            self._write_official_run_tree(
                v2_v6,
                "V2_reference_full",
                ["uncapped", "0p50"],
            )
            self._write_official_run_tree(
                v7,
                "V7_real_macro_vintage_clean_no_dxy_garch",
                ["uncapped", "0p50"],
            )
            self._write_official_run_tree(
                v8,
                "V8_ewma_garch_vol_current",
                ["uncapped", "0p50"],
            )
            output_dir = root / "out"

            with mock.patch.object(
                report_module,
                "EXPECTED_FULL_CANDIDATES",
                [
                    "V2_reference_full",
                    "V7_real_macro_vintage_clean_no_dxy_garch",
                    "V8_ewma_garch_vol_current",
                ],
            ), mock.patch.object(
                report_module,
                "EXPECTED_CAP_LABELS",
                ["uncapped", "0p50"],
            ), mock.patch.object(
                report_module,
                "EXPECTED_FOLDS",
                ["F1"],
            ), mock.patch.object(
                report_module,
                "EXPECTED_SEEDS",
                [7],
            ), mock.patch.object(
                report_module,
                "EXPECTED_HISTORIES_PER_CANDIDATE_CAP",
                1,
            ):
                report = build_official_asset_specific_cost_full_report(
                    v2_v6_dir=str(v2_v6),
                    v7_dir=str(v7),
                    v8_dir=str(v8),
                    output_dir=str(output_dir),
                )

            self.assertEqual(len(report["all_candidate_caps"]), 6)
            self.assertEqual(report["metadata"]["found_histories"], 6)
            self.assertEqual(
                report["metadata"]["score_scope"],
                "combined_asset_specific_full_universe",
            )
            expected_files = {
                "asset_specific_cost_all_candidate_caps.csv",
                "asset_specific_cost_selected_candidates.csv",
                "asset_specific_cost_main_ranking.csv",
                "asset_specific_cost_best_by_metric.csv",
                "asset_specific_cost_metadata.json",
                "asset_specific_cost_summary.md",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output_dir.iterdir()}))

    def test_official_builder_rejects_scalar_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v2_v6 = root / "v2_v6"
            v7 = root / "v7"
            v8 = root / "v8"
            self._write_official_run_tree(v2_v6, "V2_reference_full", ["uncapped"])
            self._write_official_run_tree(
                v7,
                "V7_real_macro_vintage_clean_no_dxy_garch",
                ["uncapped"],
            )
            self._write_official_run_tree(v8, "V8_ewma_garch_vol_current", ["uncapped"])
            history_path = (
                v2_v6
                / "per_candidate"
                / "V2_reference_full"
                / "F1_V2_reference_full_cap_uncapped_seed_7"
                / "test_policy_history.csv"
            )
            history = pd.read_csv(history_path)
            history["transaction_cost_mode"] = "scalar"
            history.to_csv(history_path, index=False)

            with mock.patch.object(
                report_module,
                "EXPECTED_FULL_CANDIDATES",
                [
                    "V2_reference_full",
                    "V7_real_macro_vintage_clean_no_dxy_garch",
                    "V8_ewma_garch_vol_current",
                ],
            ), mock.patch.object(
                report_module,
                "EXPECTED_CAP_LABELS",
                ["uncapped"],
            ), mock.patch.object(
                report_module,
                "EXPECTED_FOLDS",
                ["F1"],
            ), mock.patch.object(
                report_module,
                "EXPECTED_SEEDS",
                [7],
            ), mock.patch.object(
                report_module,
                "EXPECTED_HISTORIES_PER_CANDIDATE_CAP",
                1,
            ):
                with self.assertRaisesRegex(ValueError, "non asset-specific"):
                    build_official_asset_specific_cost_full_report(
                        v2_v6_dir=str(v2_v6),
                        v7_dir=str(v7),
                        v8_dir=str(v8),
                        output_dir=str(root / "out"),
                    )

    def test_official_builder_rejects_missing_asset_contribution_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v2_v6 = root / "v2_v6"
            v7 = root / "v7"
            v8 = root / "v8"
            self._write_official_run_tree(v2_v6, "V2_reference_full", ["uncapped"])
            self._write_official_run_tree(
                v7,
                "V7_real_macro_vintage_clean_no_dxy_garch",
                ["uncapped"],
            )
            self._write_official_run_tree(v8, "V8_ewma_garch_vol_current", ["uncapped"])
            history_path = (
                v2_v6
                / "per_candidate"
                / "V2_reference_full"
                / "F1_V2_reference_full_cap_uncapped_seed_7"
                / "test_policy_history.csv"
            )
            history = pd.read_csv(history_path).drop(
                columns=["asset_transaction_cost_contribution_BTC-USD"],
            )
            history.to_csv(history_path, index=False)

            with mock.patch.object(
                report_module,
                "EXPECTED_FULL_CANDIDATES",
                [
                    "V2_reference_full",
                    "V7_real_macro_vintage_clean_no_dxy_garch",
                    "V8_ewma_garch_vol_current",
                ],
            ), mock.patch.object(
                report_module,
                "EXPECTED_CAP_LABELS",
                ["uncapped"],
            ), mock.patch.object(
                report_module,
                "EXPECTED_FOLDS",
                ["F1"],
            ), mock.patch.object(
                report_module,
                "EXPECTED_SEEDS",
                [7],
            ), mock.patch.object(
                report_module,
                "EXPECTED_HISTORIES_PER_CANDIDATE_CAP",
                1,
            ):
                with self.assertRaisesRegex(ValueError, "missing required history columns"):
                    build_official_asset_specific_cost_full_report(
                        v2_v6_dir=str(v2_v6),
                        v7_dir=str(v7),
                        v8_dir=str(v8),
                        output_dir=str(root / "out"),
                    )

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
        if path.suffix == ".csv":
            pd.DataFrame(
                [
                    {
                        "Unnamed: 0": "agent",
                        "cumulative_return": 0.10,
                        "annualized_return": 0.10,
                        "annualized_volatility": 0.10,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.10,
                        "sortino_ratio": 1.0,
                        "calmar_ratio": 1.0,
                    }
                ]
            ).to_csv(path, index=False)
            return
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

    def _write_official_run_tree(self, root, candidate, caps):
        candidate_dir = root / "per_candidate" / candidate
        (candidate_dir / "configs").mkdir(parents=True)
        for cap in caps:
            run_name = f"F1_{candidate}_cap_{cap}_seed_7"
            run_dir = candidate_dir / run_name
            run_dir.mkdir(parents=True)
            self._write_official_history(run_dir / "test_policy_history.csv")
            pd.DataFrame(
                [
                    {
                        "Unnamed: 0": "agent",
                        "cumulative_return": 0.10 if cap == "uncapped" else 0.20,
                        "annualized_return": 0.10 if cap == "uncapped" else 0.20,
                        "annualized_volatility": 0.10,
                        "sharpe_ratio": 0.5 if cap == "uncapped" else 1.0,
                        "max_drawdown": -0.12 if cap == "uncapped" else -0.08,
                        "sortino_ratio": 0.5 if cap == "uncapped" else 1.0,
                        "calmar_ratio": 0.5 if cap == "uncapped" else 1.0,
                    }
                ]
            ).to_csv(run_dir / "test_metrics_table.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "average_turnover": 0.20 if cap == "uncapped" else 0.10,
                        "average_transaction_cost": 0.001,
                        "average_effective_number_of_assets": 1.1 if cap == "uncapped" else 3.0,
                        "average_max_weight": 0.95 if cap == "uncapped" else 0.50,
                        "average_cash_weight": 0.10,
                    }
                ]
            ).to_csv(run_dir / "test_diagnostics.csv", index=False)

    def _write_official_history(self, path):
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-05", periods=4, freq="W-FRI"),
                "financial_net_return": [0.01, 0.02, -0.005, 0.015],
                "transaction_cost_mode": ["asset_specific"] * 4,
                "transaction_cost": [0.001] * 4,
                "turnover": [0.1] * 4,
                "asset_turnover_SPY": [0.01] * 4,
                "asset_turnover_TLT": [0.01] * 4,
                "asset_turnover_GLD": [0.01] * 4,
                "asset_turnover_BTC-USD": [0.01] * 4,
                "asset_turnover_CASH": [0.01] * 4,
                "asset_transaction_cost_contribution_SPY": [0.000002] * 4,
                "asset_transaction_cost_contribution_TLT": [0.000002] * 4,
                "asset_transaction_cost_contribution_GLD": [0.000002] * 4,
                "asset_transaction_cost_contribution_BTC-USD": [0.000010] * 4,
                "asset_transaction_cost_contribution_CASH": [0.0] * 4,
                "weight_SPY": [0.2] * 4,
                "weight_TLT": [0.2] * 4,
                "weight_GLD": [0.2] * 4,
                "weight_BTC-USD": [0.2] * 4,
                "weight_CASH": [0.2] * 4,
                "cash_weight": [0.2] * 4,
            }
        ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
