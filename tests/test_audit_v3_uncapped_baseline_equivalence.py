"""Tests for V3 uncapped baseline equivalence audit."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.audit_v3_uncapped_baseline_equivalence import (
    audit_v3_uncapped_baseline_equivalence,
    build_metadata_comparison,
    build_metric_comparison,
    check_uncapped_row_has_no_cap,
    load_audit_inputs,
)


class V3UncappedBaselineEquivalenceAuditTests(unittest.TestCase):
    def test_metadata_comparison_detects_mismatched_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            protocol_dir, cap_dir, candidate_dir = self._write_inputs(temp_dir)
            metadata_path = candidate_dir / "max_weight_cap_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["candidate"] = "V3_wrong_candidate"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            inputs = load_audit_inputs(protocol_dir, cap_dir)

            comparison = build_metadata_comparison(inputs)
            row = comparison.set_index("field").loc["candidate"]

        self.assertEqual(row["status"], "fail")

    def test_metadata_comparison_detects_mismatched_cap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            protocol_dir, cap_dir, _ = self._write_inputs(temp_dir)
            all_results_path = cap_dir / "cap_sensitivity_all_results.csv"
            all_results = pd.read_csv(all_results_path)
            all_results.loc[
                all_results["candidate_name"] == "V3_real_macro_current_cap_uncapped",
                "max_weight_cap",
            ] = 0.5
            all_results.to_csv(all_results_path, index=False)
            inputs = load_audit_inputs(protocol_dir, cap_dir)

            comparison = build_metadata_comparison(inputs)
            row = comparison.set_index("field").loc["uncapped_max_weight_cap"]

        self.assertEqual(row["status"], "fail")

    def test_metric_comparison_reports_deltas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            protocol_dir, cap_dir, _ = self._write_inputs(temp_dir)
            inputs = load_audit_inputs(protocol_dir, cap_dir)

            comparison = build_metric_comparison(inputs)
            sharpe_row = comparison.set_index("metric").loc["sharpe"]

        self.assertAlmostEqual(sharpe_row["protocol_value"], 0.2)
        self.assertAlmostEqual(sharpe_row["cap_uncapped_value"], 0.3)
        self.assertAlmostEqual(sharpe_row["delta_cap_minus_protocol"], 0.1)

    def test_check_fails_if_uncapped_row_has_cap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            protocol_dir, cap_dir, _ = self._write_inputs(temp_dir)
            all_results_path = cap_dir / "cap_sensitivity_all_results.csv"
            all_results = pd.read_csv(all_results_path)
            all_results.loc[
                all_results["candidate_name"] == "V3_real_macro_current_cap_uncapped",
                "max_weight_cap",
            ] = 0.6
            all_results.to_csv(all_results_path, index=False)
            inputs = load_audit_inputs(protocol_dir, cap_dir)

            check = check_uncapped_row_has_no_cap(inputs)

        self.assertEqual(check["status"], "fail")
        self.assertIn("0.6", check["details"])

    def test_summary_markdown_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            protocol_dir, cap_dir, _ = self._write_inputs(temp_dir)
            output_dir = Path(temp_dir) / "audit"

            result = audit_v3_uncapped_baseline_equivalence(
                protocol_dir=str(protocol_dir),
                cap_sensitivity_dir=str(cap_dir),
                output_dir=str(output_dir),
            )

            summary_path = Path(result["paths"]["summary"])
            summary_exists = summary_path.exists()

        self.assertTrue(summary_exists)
        self.assertIn("V3 Uncapped Baseline Equivalence Audit", result["summary_markdown"])

    def _write_inputs(self, temp_dir: str) -> tuple[Path, Path, Path]:
        root = Path(temp_dir)
        protocol_dir = root / "protocol"
        cap_dir = root / "cap"
        candidate_dir = cap_dir / "per_candidate" / "V3_real_macro_current"
        protocol_dir.mkdir()
        candidate_dir.mkdir(parents=True)

        protocol_metadata = {
            "returns_path": "data/processed/returns_weekly_latest.csv",
            "candidates": ["V3_real_macro_current"],
            "episodes": 60,
            "seeds": [7],
            "folds": [{"fold_id": "F1"}],
            "actual_folds": [{"fold": "F1", "n_test": 2}],
            "transaction_cost_rate": 0.001,
            "base_config_path": "configs/empirical_long_history.yaml",
            "run_configs": {
                "F1_V3_real_macro_current_seed_7": {
                    "reward": {"lambda_return": 1.0},
                    "features": {
                        "version": "v3",
                        "macro_path": "data/processed/macro_weekly_latest.csv",
                    },
                }
            },
        }
        cap_metadata = {
            "returns_path": "data/processed/returns_weekly_latest.csv",
            "candidates": ["V3_real_macro_current"],
            "episodes": 60,
            "seeds": [7],
            "candidate_output_dirs": {
                "V3_real_macro_current": str(candidate_dir),
            },
        }
        max_weight_metadata = {
            "candidate": "V3_real_macro_current",
            "episodes": 60,
            "seeds": [7],
            "folds": [{"fold_id": "F1"}],
            "actual_folds": [{"fold": "F1", "n_test": 2}],
            "transaction_cost": 0.001,
            "base_config_path": "configs/empirical_long_history.yaml",
            "run_configs": {
                "F1_V3_real_macro_current_cap_uncapped_seed_7": {
                    "reward": {"lambda_return": 1.0},
                    "features": {
                        "version": "v3",
                        "macro_path": "data/processed/macro_weekly_latest.csv",
                    },
                    "max_weight_cap": None,
                }
            },
        }
        (protocol_dir / "protocol_pure_td3_revalidation_metadata.json").write_text(
            json.dumps(protocol_metadata),
            encoding="utf-8",
        )
        (cap_dir / "cap_sensitivity_metadata.json").write_text(
            json.dumps(cap_metadata),
            encoding="utf-8",
        )
        (candidate_dir / "max_weight_cap_metadata.json").write_text(
            json.dumps(max_weight_metadata),
            encoding="utf-8",
        )

        self._protocol_overall().to_csv(
            protocol_dir / "overall_aggregate_by_strategy_split.csv",
            index=False,
        )
        self._protocol_robust().to_csv(protocol_dir / "robust_score_ranking.csv", index=False)
        self._protocol_seed_fold().to_csv(
            protocol_dir / "seed_fold_strategy_results.csv",
            index=False,
        )
        self._cap_all_results().to_csv(cap_dir / "cap_sensitivity_all_results.csv", index=False)
        self._cap_overall().to_csv(
            candidate_dir / "overall_aggregate_by_strategy_split.csv",
            index=False,
        )
        self._cap_robust().to_csv(candidate_dir / "robust_score_ranking.csv", index=False)
        self._cap_seed_fold().to_csv(
            candidate_dir / "seed_fold_strategy_results.csv",
            index=False,
        )
        return protocol_dir, cap_dir, candidate_dir

    def _protocol_overall(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strategy": "V3_real_macro_current",
                    "split": "test",
                    "mean_sharpe": 0.2,
                    "mean_annualized_return": 0.05,
                    "mean_annualized_volatility": 0.1,
                    "mean_cumulative_return": 0.04,
                    "mean_max_drawdown": -0.2,
                    "worst_max_drawdown": -0.3,
                    "mean_average_turnover": 0.3,
                    "mean_average_effective_number_of_assets": 1.1,
                    "mean_average_max_weight": 0.9,
                    "mean_cash_weight": 0.1,
                    "cash_above_10_rate": 0.2,
                    "mean_transaction_cost": 0.0003,
                }
            ]
        )

    def _protocol_robust(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strategy": "V3_real_macro_current",
                    "robust_score": 0.4,
                    "median_run_dsr_n25": 0.1,
                    "date_averaged_dsr_n25": 0.2,
                    "sharpe": 0.2,
                    "max_drawdown": -0.2,
                    "worst_drawdown": -0.3,
                    "turnover": 0.3,
                    "effective_assets": 1.1,
                }
            ]
        )

    def _protocol_seed_fold(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                self._seed_row("V3_real_macro_current", 0.2, 0.05, -0.2, 0.3),
            ]
        )

    def _cap_all_results(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "candidate_name": "V3_real_macro_current_cap_uncapped",
                    "split": "test",
                    "max_weight_cap": pd.NA,
                    "sharpe": 0.3,
                    "annualized_return": 0.06,
                    "annualized_volatility": 0.11,
                    "cumulative_return": 0.05,
                    "max_drawdown": -0.21,
                    "worst_max_drawdown": -0.31,
                    "average_turnover": 0.4,
                    "average_effective_number_of_assets": 1.2,
                    "average_max_weight": 0.85,
                    "mean_cash_weight": 0.1,
                    "cash_above_10_rate": 0.2,
                    "mean_transaction_cost": 0.0004,
                }
            ]
        )

    def _cap_overall(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strategy": "V3_real_macro_current_cap_uncapped",
                    "split": "test",
                    "mean_sharpe": 0.3,
                    "mean_annualized_return": 0.06,
                    "mean_max_drawdown": -0.21,
                }
            ]
        )

    def _cap_robust(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strategy": "V3_real_macro_current_cap_uncapped",
                    "robust_score": 0.5,
                    "median_run_dsr_n25": 0.15,
                    "date_averaged_dsr_n25": 0.25,
                    "sharpe": 0.3,
                    "max_drawdown": -0.21,
                    "worst_drawdown": -0.31,
                    "turnover": 0.4,
                    "effective_assets": 1.2,
                }
            ]
        )

    def _cap_seed_fold(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                self._seed_row("V3_real_macro_current_cap_uncapped", 0.3, 0.06, -0.21, 0.4),
            ]
        )

    def _seed_row(
        self,
        strategy: str,
        sharpe: float,
        annualized_return: float,
        max_drawdown: float,
        turnover: float,
    ) -> dict:
        return {
            "strategy": strategy,
            "fold": "F1",
            "seed": 7,
            "split": "test",
            "sharpe_ratio": sharpe,
            "annualized_return": annualized_return,
            "annualized_volatility": 0.1,
            "cumulative_return": annualized_return,
            "max_drawdown": max_drawdown,
            "average_turnover": turnover,
            "average_effective_number_of_assets": 1.1,
            "average_max_weight": 0.9,
            "transaction_cost": turnover * 0.001,
        }


if __name__ == "__main__":
    unittest.main()
