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
            },
            "test_comparison_summary": {
                "best_policy_by_sharpe": "buy_and_hold",
                "best_sharpe_ratio": 0.8,
            },
            "validation_diagnostics": {
                "final_portfolio_value": 102000.0,
                "average_turnover": 0.1,
                "final_weights": {
                    "SPY": 0.2,
                    "TLT": 0.2,
                    "GLD": 0.2,
                    "BTC-USD": 0.2,
                    "CASH": 0.2,
                },
            },
            "test_diagnostics": {
                "final_portfolio_value": 99000.0,
                "average_turnover": 0.2,
                "final_weights": {
                    "SPY": 0.3,
                    "TLT": 0.2,
                    "GLD": 0.2,
                    "BTC-USD": 0.1,
                    "CASH": 0.2,
                },
            },
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
        ]
        self.assertEqual(list(validation_metrics.index), expected_index)
        self.assertEqual(list(test_metrics.index), expected_index)

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
                "cumulative_return": [0.01, 0.02, 0.015, 0.018],
                "annualized_return": [0.10, 0.12, 0.11, 0.09],
                "annualized_volatility": [0.05, 0.06, 0.055, 0.07],
                "sharpe_ratio": [1.0, 0.8, 0.9, 0.7],
                "max_drawdown": [-0.02, -0.03, -0.025, -0.04],
            },
            index=[
                "agent",
                "equal_weight_gross",
                "equal_weight_rebalanced_net",
                "buy_and_hold",
            ],
        )


if __name__ == "__main__":
    unittest.main()
