"""Tests for the protocol TD3 and benchmark comparison runner."""

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.experiments import run_protocol_td3_comparison as td3_runner_module
from src.experiments.run_protocol_td3_comparison import run_protocol_td3_comparison


class ProtocolTD3ComparisonRunnerTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2023-01-06", periods=60, freq="W-FRI")
        self.returns = pd.DataFrame(
            {
                "date": self.dates,
                "SPY": [0.01 + 0.001 * ((i % 4) - 1.5) for i in range(60)],
                "TLT": [0.003 + 0.001 * ((i % 3) - 1.0) for i in range(60)],
                "GLD": [0.004 + 0.002 * ((i % 5) - 2.0) for i in range(60)],
                "BTC-USD": [0.012 + 0.010 * ((i % 6) - 2.5) for i in range(60)],
                "CASH": [0.0] * 60,
            }
        )
        self.td3_results = [
            {
                "strategy_name": "V6_financial_state",
                "candidate_name": "V6_financial_state",
                "feature_version": "v6",
                "cumulative_return": 0.12,
                "annualized_return": 0.10,
                "annualized_volatility": 0.18,
                "sharpe": 0.56,
                "sortino": 0.80,
                "calmar": 0.70,
                "max_drawdown": -0.14,
                "average_turnover": 0.30,
                "total_transaction_cost": 0.02,
                "average_max_weight": 0.72,
                "average_effective_number_of_assets": 1.80,
                "mean_cash_weight": 0.04,
                "cash_above_10pct": 0.05,
                "robust_score": 0.42,
                "median_run_dsr_n25": 0.20,
                "dsr_method": "median_run",
            }
        ]

    def test_smoke_runner_creates_expected_output_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_smoke(temp_dir)

            for key in [
                "protocol_comparison_metrics",
                "protocol_comparison_summary",
                "protocol_comparison_diagnostics",
                "protocol_model_selection_table",
                "benchmark_metrics_table",
                "td3_candidate_metrics_table",
                "metadata",
            ]:
                self.assertTrue(Path(result["paths"][key]).exists())
            self.assertTrue(Path(result["paths"]["histories_dir"]).exists())

    def test_benchmark_suite_is_included(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_smoke(temp_dir)
            strategies = set(result["combined_metrics"]["strategy_name"])

        self.assertIn("BuyHold_SPY", strategies)
        self.assertIn("Equal_Weight", strategies)
        self.assertIn("rolling_risk_parity_inverse_vol_12p", strategies)
        self.assertIn("rolling_markowitz_long_only_52p", strategies)

    def test_td3_candidate_rows_are_included(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_smoke(temp_dir)
            td3_rows = result["combined_metrics"][
                result["combined_metrics"]["strategy_type"] == "td3"
            ]

        self.assertEqual(len(td3_rows), 1)
        self.assertEqual(td3_rows.iloc[0]["strategy_name"], "V6_financial_state")
        self.assertEqual(td3_rows.iloc[0]["feature_version"], "v6")

    def test_output_tables_include_strategy_type_and_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_smoke(temp_dir)

            for frame_key in [
                "combined_metrics",
                "comparison_summary",
                "diagnostics",
                "model_selection",
            ]:
                frame = result[frame_key]
                self.assertIn("strategy_name", frame.columns)
                self.assertIn("strategy_type", frame.columns)

    def test_model_selection_table_contains_conservative_flags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_smoke(temp_dir)
            table = result["model_selection"]

        for column in [
            "beats_equal_weight",
            "beats_equal_weight_risky",
            "beats_risk_parity",
            "beats_markowitz_long_only",
            "drawdown_not_worse_than_equal_weight",
            "turnover_acceptable",
            "robust_score_rank",
            "final_protocol_rank",
        ]:
            self.assertIn(column, table.columns)

    def test_metadata_json_contains_protocol_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_smoke(temp_dir)
            metadata = json.loads(Path(result["paths"]["metadata"]).read_text())

        self.assertTrue(metadata["smoke_mode"])
        self.assertIn("information through t-1", metadata["timing_convention"])
        self.assertIn("sum(abs", metadata["turnover_convention"])
        self.assertIn("median_run", metadata["DSR_method_policy"])

    def test_smoke_mode_does_not_include_training_orchestration(self):
        source = inspect.getsource(td3_runner_module)

        self.assertNotIn("train_td3", source)
        self.assertNotIn("TD3Agent", source)

    def test_csv_td3_results_path_still_works(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            td3_results_path = Path(temp_dir) / "td3_results.csv"
            pd.DataFrame(self.td3_results).to_csv(td3_results_path, index=False)

            result = run_protocol_td3_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
                td3_results_path=str(td3_results_path),
                smoke=True,
            )

            td3_rows = result["td3_candidate_metrics"]
            metadata = result["metadata"]

        self.assertEqual(len(td3_rows), 1)
        self.assertEqual(td3_rows.iloc[0]["strategy_name"], "V6_financial_state")
        self.assertEqual(metadata["td3_results_path_type"], "csv")

    def test_directory_td3_results_path_ingests_test_drl_rows_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            experiment_dir = self._write_synthetic_experiment_dir(temp_dir)

            result = run_protocol_td3_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
                td3_results_path=str(experiment_dir),
                smoke=True,
            )

            td3_rows = result["td3_candidate_metrics"]
            strategies = set(td3_rows["strategy_name"])

        self.assertEqual(strategies, {"V2_reference_full", "V6_financial_state"})
        self.assertNotIn("Equal_Weight", strategies)
        self.assertTrue((td3_rows["strategy_type"] == "td3").all())

    def test_directory_ingestion_merges_robust_score_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            experiment_dir = self._write_synthetic_experiment_dir(temp_dir)

            result = run_protocol_td3_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
                td3_results_path=str(experiment_dir),
                smoke=True,
            )

            v6 = result["td3_candidate_metrics"].set_index("strategy_name").loc[
                "V6_financial_state"
            ]

        self.assertAlmostEqual(float(v6["robust_score"]), 0.44)
        self.assertAlmostEqual(float(v6["pooled_dsr_n25"]), 0.62)
        self.assertAlmostEqual(float(v6["median_run_dsr_n25"]), 0.08)
        self.assertEqual(v6["dsr_method"], "median_run")

    def test_directory_ingestion_infers_feature_versions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            experiment_dir = self._write_synthetic_experiment_dir(temp_dir)

            result = run_protocol_td3_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
                td3_results_path=str(experiment_dir),
                smoke=True,
            )

            versions = result["td3_candidate_metrics"].set_index("strategy_name")[
                "feature_version"
            ].to_dict()

        self.assertEqual(versions["V2_reference_full"], "v2")
        self.assertEqual(versions["V6_financial_state"], "v6")

    def test_directory_metadata_records_path_type_and_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            experiment_dir = self._write_synthetic_experiment_dir(temp_dir)

            result = run_protocol_td3_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
                td3_results_path=str(experiment_dir),
                smoke=True,
            )
            metadata = json.loads(Path(result["paths"]["metadata"]).read_text())

        self.assertEqual(metadata["td3_results_path_type"], "directory")
        self.assertEqual(metadata["td3_results_path"], str(experiment_dir))
        self.assertEqual(len(metadata["td3_ingestion_source_files"]), 2)
        self.assertIn("mean_transaction_cost", metadata["td3_transaction_cost_note"])

    def test_directory_runner_writes_non_empty_td3_candidate_metrics_table(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            experiment_dir = self._write_synthetic_experiment_dir(temp_dir)

            result = run_protocol_td3_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
                td3_results_path=str(experiment_dir),
                smoke=True,
            )
            td3_table = pd.read_csv(result["paths"]["td3_candidate_metrics_table"])

        self.assertFalse(td3_table.empty)
        self.assertIn("V6_financial_state", set(td3_table["strategy_name"]))
        self.assertTrue(td3_table["total_transaction_cost"].isna().all())
        self.assertIn("average_transaction_cost", td3_table.columns)

    def _run_smoke(self, temp_dir: str) -> dict:
        returns_path = self._write_returns(temp_dir)
        output_dir = Path(temp_dir) / "outputs"
        return run_protocol_td3_comparison(
            returns_path=str(returns_path),
            output_dir=str(output_dir),
            candidates=[
                {
                    "name": "V6_financial_state",
                    "feature_version": "v6",
                    "config_path": "configs/config.yaml",
                    "seeds": [7],
                    "episodes": 1,
                }
            ],
            td3_results=self.td3_results,
            smoke=True,
            transaction_cost=0.01,
        )

    def _write_returns(self, temp_dir: str) -> Path:
        path = Path(temp_dir) / "returns.csv"
        self.returns.to_csv(path, index=False)
        return path

    def _write_synthetic_experiment_dir(self, temp_dir: str) -> Path:
        experiment_dir = Path(temp_dir) / "experiment"
        experiment_dir.mkdir()
        aggregate = pd.DataFrame(
            [
                {
                    "strategy": "V2_reference_full",
                    "split": "validation",
                    "strategy_type": "drl",
                    "mean_cumulative_return": 0.01,
                    "mean_annualized_return": 0.01,
                    "mean_annualized_volatility": 0.10,
                    "mean_sharpe": 0.10,
                    "mean_sortino": 0.20,
                    "mean_calmar": 0.30,
                    "mean_max_drawdown": -0.20,
                    "mean_average_turnover": 0.40,
                    "mean_average_max_weight": 0.90,
                    "mean_average_effective_number_of_assets": 1.20,
                    "mean_cash_weight": 0.10,
                    "cash_above_10_rate": 0.20,
                    "mean_transaction_cost": 0.0004,
                },
                {
                    "strategy": "V2_reference_full",
                    "split": "test",
                    "strategy_type": "drl",
                    "mean_cumulative_return": 0.02,
                    "mean_annualized_return": 0.02,
                    "mean_annualized_volatility": 0.11,
                    "mean_sharpe": 0.21,
                    "mean_sortino": 0.31,
                    "mean_calmar": 0.41,
                    "mean_max_drawdown": -0.19,
                    "mean_average_turnover": 0.41,
                    "mean_average_max_weight": 0.91,
                    "mean_average_effective_number_of_assets": 1.21,
                    "mean_cash_weight": 0.11,
                    "cash_above_10_rate": 0.21,
                    "mean_transaction_cost": 0.0005,
                },
                {
                    "strategy": "V6_financial_state",
                    "split": "test",
                    "strategy_type": "drl",
                    "mean_cumulative_return": 0.12,
                    "mean_annualized_return": 0.10,
                    "mean_annualized_volatility": 0.18,
                    "mean_sharpe": 0.56,
                    "mean_sortino": 0.80,
                    "mean_calmar": 0.70,
                    "mean_max_drawdown": -0.14,
                    "mean_average_turnover": 0.30,
                    "mean_average_max_weight": 0.72,
                    "mean_average_effective_number_of_assets": 1.80,
                    "mean_cash_weight": 0.04,
                    "cash_above_10_rate": 0.05,
                    "mean_transaction_cost": 0.0006,
                },
                {
                    "strategy": "Equal_Weight",
                    "split": "test",
                    "strategy_type": "benchmark",
                    "mean_cumulative_return": 0.20,
                    "mean_sharpe": 0.90,
                },
            ]
        )
        aggregate.to_csv(
            experiment_dir / "overall_aggregate_by_strategy_split.csv",
            index=False,
        )
        robust = pd.DataFrame(
            [
                {
                    "strategy": "V6_financial_state",
                    "type": "drl",
                    "robust_score": 0.44,
                    "pooled_dsr_n10": 0.75,
                    "pooled_dsr_n25": 0.62,
                    "pooled_dsr_n50": 0.51,
                    "mean_run_dsr_n25": 0.18,
                    "median_run_dsr_n25": 0.08,
                    "date_averaged_dsr_n25": 0.24,
                    "dsr_method": "median_run",
                },
                {
                    "strategy": "V2_reference_full",
                    "type": "drl",
                    "robust_score": 0.35,
                    "pooled_dsr_n10": 0.40,
                    "pooled_dsr_n25": 0.30,
                    "pooled_dsr_n50": 0.20,
                    "mean_run_dsr_n25": 0.12,
                    "median_run_dsr_n25": 0.05,
                    "date_averaged_dsr_n25": 0.10,
                    "dsr_method": "median_run",
                },
                {
                    "strategy": "Equal_Weight",
                    "type": "benchmark",
                    "robust_score": 0.90,
                },
            ]
        )
        robust.to_csv(experiment_dir / "robust_score_ranking.csv", index=False)
        return experiment_dir


if __name__ == "__main__":
    unittest.main()
