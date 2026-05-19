"""Tests for CSV output utilities for basic experiment results."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.experiments.save_experiment_outputs import save_basic_experiment_outputs


class SaveExperimentOutputsTests(unittest.TestCase):
    def setUp(self):
        self.experiment_result = {
            "training_summary": {
                "total_episodes": 2,
                "final_episode": 2,
                "final_portfolio_value": 101000.0,
            },
            "validation_metrics_table": self._metrics_table(),
            "test_metrics_table": self._metrics_table(),
            "validation_comparison_summary": {
                "best_policy_by_sharpe": "agent",
                "best_sharpe_ratio": 1.2,
                "best_individual_buyhold_by_sharpe": "buy_hold_SPY",
                "agent_vs_best_individual_buyhold_sharpe_diff": 0.1,
            },
            "test_comparison_summary": {
                "best_policy_by_sharpe": "buy_and_hold",
                "best_sharpe_ratio": 0.8,
                "best_individual_buyhold_by_sharpe": "buy_hold_SPY",
                "agent_vs_best_individual_buyhold_sharpe_diff": -0.2,
            },
            "validation_diagnostics": self._diagnostics(final_portfolio_value=102000.0),
            "test_diagnostics": self._diagnostics(final_portfolio_value=99000.0),
            "raw_result": {"agent": object(), "replay_buffer": object()},
        }

    def test_returns_expected_path_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(self.experiment_result, temp_dir)

        self.assertEqual(
            set(paths.keys()),
            {
                "output_dir",
                "training_summary",
                "validation_metrics_table",
                "test_metrics_table",
                "validation_comparison_summary",
                "test_comparison_summary",
                "validation_diagnostics",
                "test_diagnostics",
            },
        )

    def test_does_not_fail_when_policy_history_is_absent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(self.experiment_result, temp_dir)

        self.assertNotIn("validation_policy_history", paths)
        self.assertNotIn("test_policy_history", paths)

    def test_creates_experiment_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(self.experiment_result, temp_dir)

            self.assertTrue(Path(paths["output_dir"]).is_dir())

    def test_writes_all_expected_csv_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(self.experiment_result, temp_dir)

            for key, path in paths.items():
                if key == "output_dir":
                    continue
                self.assertTrue(Path(path).is_file())

    def test_does_not_save_raw_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(self.experiment_result, temp_dir)

            output_files = {path.name for path in Path(paths["output_dir"]).iterdir()}
            self.assertNotIn("raw_result.csv", output_files)

    def test_saved_metrics_tables_preserve_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(self.experiment_result, temp_dir)

            validation_metrics = pd.read_csv(paths["validation_metrics_table"], index_col=0)
            test_metrics = pd.read_csv(paths["test_metrics_table"], index_col=0)

        expected_index = [
            "agent",
            "equal_weight_gross",
            "equal_weight_rebalanced_net",
            "buy_and_hold",
            "buy_hold_SPY",
        ]
        self.assertEqual(list(validation_metrics.index), expected_index)
        self.assertEqual(list(test_metrics.index), expected_index)

    def test_saved_test_comparison_summary_includes_individual_buyhold_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(self.experiment_result, temp_dir)

            summary = pd.read_csv(paths["test_comparison_summary"])

        self.assertIn("best_individual_buyhold_by_sharpe", summary.columns)
        self.assertIn("agent_vs_best_individual_buyhold_sharpe_diff", summary.columns)

    def test_saved_diagnostics_flatten_final_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(self.experiment_result, temp_dir)

            diagnostics = pd.read_csv(paths["validation_diagnostics"])

        expected_weight_columns = {
            "final_weight_SPY",
            "final_weight_TLT",
            "final_weight_GLD",
            "final_weight_BTC-USD",
            "final_weight_CASH",
        }
        self.assertTrue(expected_weight_columns.issubset(set(diagnostics.columns)))
        self.assertNotIn("final_weights", diagnostics.columns)

    def test_saved_diagnostics_include_allocation_risk_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(self.experiment_result, temp_dir)

            diagnostics = pd.read_csv(paths["validation_diagnostics"])

        expected_columns = {
            "average_max_weight",
            "final_max_weight",
            "average_cash_weight",
            "final_cash_weight",
            "average_herfindahl_index",
            "final_herfindahl_index",
            "average_effective_number_of_assets",
            "final_effective_number_of_assets",
            "average_entropy",
            "final_entropy",
            "average_turnover",
            "final_turnover",
            "average_transaction_cost",
            "final_transaction_cost",
            "max_weight",
            "cash_weight",
            "final_weight_SPY",
            "final_weight_TLT",
            "final_weight_GLD",
            "final_weight_BTC-USD",
            "final_weight_CASH",
        }
        self.assertTrue(expected_columns.issubset(set(diagnostics.columns)))

    def test_saves_policy_history_csvs_when_present(self):
        experiment_result = dict(self.experiment_result)
        experiment_result["validation_policy_history"] = self._policy_history()
        experiment_result["test_policy_history"] = self._policy_history()

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(experiment_result, temp_dir)

            self.assertTrue(Path(paths["validation_policy_history"]).is_file())
            self.assertTrue(Path(paths["test_policy_history"]).is_file())
            validation_history = pd.read_csv(paths["validation_policy_history"])
            test_history = pd.read_csv(paths["test_policy_history"])

        self.assertIn("date", validation_history.columns)
        self.assertIn("weight_SPY", validation_history.columns)
        self.assertEqual(len(validation_history), 2)
        self.assertEqual(len(test_history), 2)

    def test_saved_policy_history_csv_includes_mandate_columns_when_present(self):
        experiment_result = dict(self.experiment_result)
        experiment_result["validation_policy_history"] = self._policy_history_with_mandate()
        experiment_result["test_policy_history"] = self._policy_history_with_mandate()

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(experiment_result, temp_dir)
            test_history = pd.read_csv(paths["test_policy_history"])

        expected_columns = {
            "mandate_penalty",
            "mandate_drawdown_breach",
            "mandate_volatility_breach",
            "mandate_max_weight_breach",
            "mandate_effective_assets_breach",
            "mandate_turnover_breach",
        }
        self.assertTrue(expected_columns.issubset(test_history.columns))

    def test_saved_policy_history_csv_includes_turnover_reward_columns_when_present(self):
        experiment_result = dict(self.experiment_result)
        experiment_result["validation_policy_history"] = (
            self._policy_history_with_turnover_reward()
        )
        experiment_result["test_policy_history"] = self._policy_history_with_turnover_reward()

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(experiment_result, temp_dir)
            test_history = pd.read_csv(paths["test_policy_history"])

        expected_columns = {
            "turnover_penalty",
            "turnover_penalty_mode",
            "turnover_free_band",
            "turnover_excess",
        }
        self.assertTrue(expected_columns.issubset(test_history.columns))

    def test_works_with_custom_output_dir_and_experiment_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_basic_experiment_outputs(
                self.experiment_result,
                output_dir=temp_dir,
                experiment_name="custom_experiment",
            )

            self.assertEqual(Path(paths["output_dir"]).name, "custom_experiment")
            self.assertTrue(Path(paths["training_summary"]).is_file())

    def test_missing_required_experiment_result_key_raises_key_error(self):
        experiment_result = dict(self.experiment_result)
        del experiment_result["training_summary"]

        with self.assertRaises(KeyError):
            save_basic_experiment_outputs(experiment_result)

    def test_non_dict_experiment_result_raises_type_error(self):
        with self.assertRaises(TypeError):
            save_basic_experiment_outputs([])

    def test_validation_metrics_table_with_wrong_type_raises_type_error(self):
        experiment_result = dict(self.experiment_result)
        experiment_result["validation_metrics_table"] = {}

        with self.assertRaises(TypeError):
            save_basic_experiment_outputs(experiment_result)

    def test_training_summary_with_wrong_type_raises_type_error(self):
        experiment_result = dict(self.experiment_result)
        experiment_result["training_summary"] = []

        with self.assertRaises(TypeError):
            save_basic_experiment_outputs(experiment_result)

    def test_diagnostics_missing_final_weights_raises_key_error(self):
        experiment_result = dict(self.experiment_result)
        experiment_result["validation_diagnostics"] = {
            "final_portfolio_value": 102000.0,
            "average_turnover": 0.1,
        }

        with self.assertRaises(KeyError):
            save_basic_experiment_outputs(experiment_result)

    def test_diagnostics_final_weights_not_dict_raises_type_error(self):
        experiment_result = dict(self.experiment_result)
        experiment_result["validation_diagnostics"] = {
            "final_portfolio_value": 102000.0,
            "average_turnover": 0.1,
            "final_weights": [],
        }

        with self.assertRaises(TypeError):
            save_basic_experiment_outputs(experiment_result)

    @staticmethod
    def _metrics_table() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cumulative_return": [0.01, 0.02, 0.015, 0.018, 0.03],
                "annualized_return": [0.10, 0.12, 0.11, 0.09, 0.13],
                "annualized_volatility": [0.05, 0.06, 0.055, 0.07, 0.08],
                "sharpe_ratio": [1.0, 0.8, 0.9, 0.7, 1.1],
                "max_drawdown": [-0.02, -0.03, -0.025, -0.04, -0.05],
            },
            index=[
                "agent",
                "equal_weight_gross",
                "equal_weight_rebalanced_net",
                "buy_and_hold",
                "buy_hold_SPY",
            ],
        )

    @staticmethod
    def _diagnostics(final_portfolio_value: float) -> dict:
        return {
            "final_portfolio_value": final_portfolio_value,
            "average_max_weight": 0.35,
            "final_max_weight": 0.40,
            "average_cash_weight": 0.18,
            "final_cash_weight": 0.20,
            "average_herfindahl_index": 0.25,
            "final_herfindahl_index": 0.28,
            "average_effective_number_of_assets": 4.0,
            "final_effective_number_of_assets": 3.57,
            "average_entropy": 1.50,
            "final_entropy": 1.45,
            "average_turnover": 0.10,
            "final_turnover": 0.12,
            "average_transaction_cost": 0.001,
            "final_transaction_cost": 0.0012,
            "max_weight": 0.40,
            "cash_weight": 0.20,
            "final_weights": {
                "SPY": 0.2,
                "TLT": 0.2,
                "GLD": 0.2,
                "BTC-USD": 0.2,
                "CASH": 0.2,
            },
        }

    @staticmethod
    def _policy_history() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-05", periods=2, freq="W-FRI"),
                "portfolio_return": [0.01, 0.02],
                "portfolio_value": [101000.0, 103020.0],
                "drawdown": [0.0, 0.0],
                "turnover": [0.1, 0.2],
                "transaction_cost": [0.0001, 0.0002],
                "max_weight": [0.3, 0.4],
                "cash_weight": [0.2, 0.1],
                "weight_SPY": [0.2, 0.4],
                "weight_TLT": [0.2, 0.2],
                "weight_GLD": [0.2, 0.1],
                "weight_BTC-USD": [0.2, 0.2],
                "weight_CASH": [0.2, 0.1],
            }
        )

    @staticmethod
    def _policy_history_with_mandate() -> pd.DataFrame:
        history = SaveExperimentOutputsTests._policy_history()
        history["mandate_penalty"] = [0.1, 0.2]
        history["mandate_drawdown_breach"] = [0.0, 0.0]
        history["mandate_volatility_breach"] = [0.0, 0.01]
        history["mandate_max_weight_breach"] = [0.0, 0.05]
        history["mandate_effective_assets_breach"] = [0.2, 0.3]
        history["mandate_turnover_breach"] = [0.0, 0.0]

        return history

    @staticmethod
    def _policy_history_with_turnover_reward() -> pd.DataFrame:
        history = SaveExperimentOutputsTests._policy_history()
        history["turnover_penalty"] = [0.01, 0.02]
        history["turnover_penalty_mode"] = ["linear", "linear"]
        history["turnover_free_band"] = [0.0, 0.0]
        history["turnover_excess"] = [0.1, 0.2]

        return history


if __name__ == "__main__":
    unittest.main()
