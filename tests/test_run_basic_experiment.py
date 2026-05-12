"""Tests for the minimal in-memory experiment runner."""

import unittest
from unittest.mock import patch

import pandas as pd

from src.experiments.run_basic_experiment import (
    run_basic_experiment,
    summarize_metrics_table,
)


class RunBasicExperimentTests(unittest.TestCase):
    def setUp(self):
        self.validation_metrics_table = self._metrics_table()
        self.test_metrics_table = self._metrics_table()
        self.validation_diagnostics = {
            "final_portfolio_value": 101000.0,
            "average_turnover": 0.1,
            "average_transaction_cost": 0.0001,
            "final_weights": {"SPY": 0.2, "TLT": 0.2, "GLD": 0.2, "BTC-USD": 0.2, "CASH": 0.2},
            "max_weight": 0.2,
            "cash_weight": 0.2,
        }
        self.test_diagnostics = {
            "final_portfolio_value": 99000.0,
            "average_turnover": 0.2,
            "average_transaction_cost": 0.0002,
            "final_weights": {"SPY": 0.3, "TLT": 0.2, "GLD": 0.2, "BTC-USD": 0.1, "CASH": 0.2},
            "max_weight": 0.3,
            "cash_weight": 0.2,
        }
        self.train_td3_result = {
            "episode_logs": [
                {
                    "episode": 1,
                    "final_portfolio_value": 100500.0,
                    "total_reward": 0.005,
                    "average_turnover": 0.15,
                    "average_transaction_cost": 0.00015,
                    "max_weight": 0.25,
                    "cash_weight": 0.2,
                },
                {
                    "episode": 2,
                    "final_portfolio_value": 101500.0,
                    "total_reward": 0.015,
                    "average_turnover": 0.12,
                    "average_transaction_cost": 0.00012,
                    "max_weight": 0.3,
                    "cash_weight": 0.1,
                },
            ],
            "validation_comparison": {
                "metrics_table": self.validation_metrics_table,
            },
            "test_comparison": {
                "metrics_table": self.test_metrics_table,
            },
            "validation_evaluation": {
                "diagnostics": self.validation_diagnostics,
            },
            "test_evaluation": {
                "diagnostics": self.test_diagnostics,
            },
        }

    def test_run_basic_experiment_returns_expected_top_level_keys(self):
        result = self._run_experiment()

        self.assertEqual(
            set(result.keys()),
            {
                "training_summary",
                "validation_metrics_table",
                "test_metrics_table",
                "validation_comparison_summary",
                "test_comparison_summary",
                "validation_diagnostics",
                "test_diagnostics",
                "raw_result",
            },
        )

    def test_train_td3_is_called_once_with_config_path(self):
        config_path = "configs/config.yaml"

        with patch(
            "src.experiments.run_basic_experiment.train_td3",
            return_value=self.train_td3_result,
        ) as train_td3_mock:
            run_basic_experiment(config_path)

        train_td3_mock.assert_called_once_with(config_path)

    def test_training_summary_uses_final_episode(self):
        result = self._run_experiment()
        training_summary = result["training_summary"]

        self.assertEqual(training_summary["total_episodes"], 2)
        self.assertEqual(training_summary["final_episode"], 2)
        self.assertEqual(training_summary["final_portfolio_value"], 101500.0)
        self.assertEqual(training_summary["final_total_reward"], 0.015)
        self.assertEqual(training_summary["final_average_turnover"], 0.12)
        self.assertEqual(training_summary["final_average_transaction_cost"], 0.00012)
        self.assertEqual(training_summary["final_max_weight"], 0.3)
        self.assertEqual(training_summary["final_cash_weight"], 0.1)

    def test_metrics_tables_are_dataframes(self):
        result = self._run_experiment()

        self.assertIsInstance(result["validation_metrics_table"], pd.DataFrame)
        self.assertIsInstance(result["test_metrics_table"], pd.DataFrame)

    def test_summarize_metrics_table_returns_expected_keys(self):
        summary = summarize_metrics_table(self.validation_metrics_table)

        self.assertEqual(
            set(summary.keys()),
            {
                "best_policy_by_sharpe",
                "best_sharpe_ratio",
                "agent_rank_by_sharpe",
                "agent_sharpe_ratio",
                "agent_cumulative_return",
                "agent_max_drawdown",
                "agent_vs_equal_weight_rebalanced_net_sharpe_diff",
                "agent_vs_buy_and_hold_sharpe_diff",
                "best_individual_buyhold_by_sharpe",
                "best_individual_buyhold_sharpe_ratio",
                "best_individual_buyhold_cumulative_return",
                "agent_vs_best_individual_buyhold_sharpe_diff",
                "agent_vs_equal_weight_rebalanced_net_cumulative_return_diff",
                "agent_vs_buy_and_hold_cumulative_return_diff",
                "agent_vs_best_individual_buyhold_cumulative_return_diff",
            },
        )

    def test_summarize_metrics_table_identifies_best_policy_by_sharpe(self):
        summary = summarize_metrics_table(self.validation_metrics_table)

        self.assertEqual(summary["best_policy_by_sharpe"], "equal_weight_rebalanced_net")
        self.assertEqual(summary["best_sharpe_ratio"], 1.5)

    def test_summarize_metrics_table_ranks_agent_by_sharpe(self):
        summary = summarize_metrics_table(self.validation_metrics_table)

        self.assertEqual(summary["agent_rank_by_sharpe"], 3)

    def test_summarize_metrics_table_computes_agent_differences(self):
        summary = summarize_metrics_table(self.validation_metrics_table)

        self.assertAlmostEqual(
            summary["agent_vs_equal_weight_rebalanced_net_sharpe_diff"],
            1.0 - 1.5,
        )
        self.assertAlmostEqual(
            summary["agent_vs_buy_and_hold_sharpe_diff"],
            1.0 - 0.4,
        )
        self.assertAlmostEqual(
            summary["agent_vs_equal_weight_rebalanced_net_cumulative_return_diff"],
            0.03 - 0.04,
        )
        self.assertAlmostEqual(
            summary["agent_vs_buy_and_hold_cumulative_return_diff"],
            0.03 - 0.01,
        )

    def test_summarize_metrics_table_identifies_best_individual_buyhold_by_sharpe(self):
        summary = summarize_metrics_table(self.validation_metrics_table)

        self.assertEqual(summary["best_individual_buyhold_by_sharpe"], "buy_hold_SPY")
        self.assertEqual(summary["best_individual_buyhold_sharpe_ratio"], 1.2)
        self.assertEqual(summary["best_individual_buyhold_cumulative_return"], 0.06)

    def test_summarize_metrics_table_computes_agent_vs_best_individual_buyhold_diffs(self):
        summary = summarize_metrics_table(self.validation_metrics_table)

        self.assertAlmostEqual(
            summary["agent_vs_best_individual_buyhold_sharpe_diff"],
            1.0 - 1.2,
        )
        self.assertAlmostEqual(
            summary["agent_vs_best_individual_buyhold_cumulative_return_diff"],
            0.03 - 0.06,
        )

    def test_run_basic_experiment_returns_comparison_summaries(self):
        result = self._run_experiment()

        self.assertIn(
            "best_individual_buyhold_by_sharpe",
            result["validation_comparison_summary"],
        )
        self.assertIn(
            "agent_vs_best_individual_buyhold_sharpe_diff",
            result["test_comparison_summary"],
        )
        self.assertEqual(
            result["validation_comparison_summary"],
            summarize_metrics_table(self.validation_metrics_table),
        )
        self.assertEqual(
            result["test_comparison_summary"],
            summarize_metrics_table(self.test_metrics_table),
        )

    def test_summarize_metrics_table_raises_key_error_for_missing_metric_column(self):
        metrics_table = self.validation_metrics_table.drop(columns=["sharpe_ratio"])

        with self.assertRaises(KeyError):
            summarize_metrics_table(metrics_table)

    def test_summarize_metrics_table_raises_key_error_for_missing_policy_row(self):
        metrics_table = self.validation_metrics_table.drop(index=["buy_and_hold"])

        with self.assertRaises(KeyError):
            summarize_metrics_table(metrics_table)

    def test_diagnostics_are_passed_through_unchanged(self):
        result = self._run_experiment()

        self.assertIs(result["validation_diagnostics"], self.validation_diagnostics)
        self.assertIs(result["test_diagnostics"], self.test_diagnostics)

    def test_raw_result_is_original_train_td3_result_object(self):
        result = self._run_experiment()

        self.assertIs(result["raw_result"], self.train_td3_result)

    def _run_experiment(self):
        with patch(
            "src.experiments.run_basic_experiment.train_td3",
            return_value=self.train_td3_result,
        ):
            return run_basic_experiment("configs/config.yaml")

    @staticmethod
    def _metrics_table() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cumulative_return": [0.03, 0.02, 0.04, 0.01, 0.06, 0.005],
                "annualized_return": [0.10, 0.08, 0.12, 0.05, 0.16, 0.02],
                "annualized_volatility": [0.05, 0.06, 0.04, 0.07, 0.08, 0.01],
                "sharpe_ratio": [1.0, 0.8, 1.5, 0.4, 1.2, 0.1],
                "max_drawdown": [-0.02, -0.03, -0.01, -0.04, -0.05, 0.0],
            },
            index=[
                "agent",
                "equal_weight_gross",
                "equal_weight_rebalanced_net",
                "buy_and_hold",
                "buy_hold_SPY",
                "buy_hold_CASH",
            ],
        )


if __name__ == "__main__":
    unittest.main()
