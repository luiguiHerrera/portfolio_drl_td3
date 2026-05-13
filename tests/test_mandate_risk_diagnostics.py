"""Tests for mandate risk diagnostics."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.mandate_risk_diagnostics import (
    add_mandate_flags,
    build_mandate_risk_report,
    load_metrics_and_diagnostics,
    summarize_dominant_assets,
    summarize_mandate_risk,
)


class MandateRiskDiagnosticsTests(unittest.TestCase):
    def test_load_metrics_and_diagnostics_combines_one_pair(self):
        with self._temporary_pair() as paths:
            result = load_metrics_and_diagnostics(
                [paths["metrics"]],
                [paths["diagnostics"]],
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "cumulative_return"], 0.10)
        self.assertEqual(result.loc[0, "average_max_weight"], 0.50)
        self.assertEqual(result.loc[0, "metrics_path"], paths["metrics"])
        self.assertEqual(result.loc[0, "diagnostics_path"], paths["diagnostics"])

    def test_load_metrics_and_diagnostics_combines_multiple_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = self._write_pair(Path(temp_dir), "first", cumulative_return=0.10)
            second = self._write_pair(Path(temp_dir), "second", cumulative_return=0.20)

            result = load_metrics_and_diagnostics(
                [first["metrics"], second["metrics"]],
                [first["diagnostics"], second["diagnostics"]],
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(result["metrics_path"].tolist(), [first["metrics"], second["metrics"]])
        self.assertEqual(result["cumulative_return"].tolist(), [0.10, 0.20])

    def test_empty_path_lists_raise_value_error(self):
        with self.assertRaises(ValueError):
            load_metrics_and_diagnostics([], [])

    def test_different_path_lengths_raise_value_error(self):
        with self.assertRaises(ValueError):
            load_metrics_and_diagnostics(["metrics.csv"], [])

    def test_missing_file_raises_file_not_found_error(self):
        with self.assertRaises(FileNotFoundError):
            load_metrics_and_diagnostics(["missing_metrics.csv"], ["missing_diag.csv"])

    def test_missing_agent_row_raises_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metrics_path = temp_path / "metrics.csv"
            diagnostics_path = temp_path / "diagnostics.csv"
            self._metrics_dataframe(index=["equal_weight"]).to_csv(metrics_path)
            self._diagnostics_dataframe().to_csv(diagnostics_path, index=False)

            with self.assertRaises(ValueError):
                load_metrics_and_diagnostics([str(metrics_path)], [str(diagnostics_path)])

    def test_empty_diagnostics_csv_raises_value_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metrics_path = temp_path / "metrics.csv"
            diagnostics_path = temp_path / "diagnostics.csv"
            self._metrics_dataframe().to_csv(metrics_path)
            self._diagnostics_dataframe().iloc[0:0].to_csv(diagnostics_path, index=False)

            with self.assertRaises(ValueError):
                load_metrics_and_diagnostics([str(metrics_path)], [str(diagnostics_path)])

    def test_add_mandate_flags_computes_expected_boolean_columns(self):
        result = add_mandate_flags(self._mandate_observations())

        for column in [
            "drawdown_ok",
            "volatility_ok",
            "final_weight_ok",
            "average_weight_ok",
            "effective_assets_ok",
            "turnover_ok",
            "mandate_pass",
        ]:
            self.assertIn(column, result.columns)
        self.assertTrue(result.loc[0, "mandate_pass"])
        self.assertFalse(result.loc[1, "mandate_pass"])

    def test_drawdown_logic_is_correct(self):
        result = add_mandate_flags(self._mandate_observations())

        self.assertTrue(result.loc[0, "drawdown_ok"])
        self.assertFalse(result.loc[1, "drawdown_ok"])

    def test_volatility_logic_is_correct(self):
        data = self._mandate_observations()
        data.loc[1, "max_drawdown"] = -0.10
        data.loc[1, "annualized_volatility"] = 0.30

        result = add_mandate_flags(data)

        self.assertTrue(result.loc[0, "volatility_ok"])
        self.assertFalse(result.loc[1, "volatility_ok"])

    def test_concentration_final_weight_logic_is_correct(self):
        result = add_mandate_flags(self._mandate_observations())

        self.assertTrue(result.loc[0, "final_weight_ok"])
        self.assertFalse(result.loc[2, "final_weight_ok"])

    def test_effective_assets_logic_is_correct(self):
        result = add_mandate_flags(self._mandate_observations())

        self.assertTrue(result.loc[0, "effective_assets_ok"])
        self.assertFalse(result.loc[3, "effective_assets_ok"])

    def test_turnover_logic_is_correct(self):
        data = self._mandate_observations()
        data.loc[1, "max_drawdown"] = -0.10
        data.loc[1, "average_turnover"] = 0.90

        result = add_mandate_flags(data)

        self.assertTrue(result.loc[0, "turnover_ok"])
        self.assertFalse(result.loc[1, "turnover_ok"])

    def test_summarize_mandate_risk_returns_expected_summary_columns(self):
        result = summarize_mandate_risk(self._mandate_observations())

        expected_columns = [
            "n_observations",
            "max_drawdown_limit",
            "max_volatility_limit",
            "max_weight_limit",
            "min_effective_assets",
            "max_turnover_limit",
            "mean_cumulative_return",
            "mean_annualized_return",
            "mean_annualized_volatility",
            "mean_sharpe_ratio",
            "mean_sortino_ratio",
            "mean_calmar_ratio",
            "mean_max_drawdown",
            "worst_max_drawdown",
            "mean_information_ratio_vs_equal_weight_rebalanced_net",
            "mean_capm_alpha_vs_SPY",
            "mean_average_max_weight",
            "mean_final_max_weight",
            "max_final_max_weight",
            "mean_average_effective_number_of_assets",
            "min_average_effective_number_of_assets",
            "mean_final_effective_number_of_assets",
            "min_final_effective_number_of_assets",
            "mean_average_herfindahl_index",
            "mean_final_herfindahl_index",
            "mean_average_turnover",
            "mean_final_turnover",
            "mean_average_transaction_cost",
            "mean_final_transaction_cost",
            "drawdown_pass_rate",
            "volatility_pass_rate",
            "final_weight_pass_rate",
            "average_weight_pass_rate",
            "effective_assets_pass_rate",
            "turnover_pass_rate",
            "mandate_pass_rate",
        ]
        for column in expected_columns:
            self.assertIn(column, result.columns)

    def test_mandate_pass_rate_is_computed_correctly(self):
        result = summarize_mandate_risk(self._mandate_observations())

        self.assertEqual(result.loc[0, "n_observations"], 4)
        self.assertEqual(result.loc[0, "mandate_pass_rate"], 0.25)

    def test_summarize_dominant_assets_identifies_assets_correctly(self):
        result = summarize_dominant_assets(self._mandate_observations())
        counts = dict(zip(result["dominant_asset"], result["count"]))

        self.assertEqual(result.loc[0, "dominant_asset"], "SPY")
        self.assertEqual(counts["SPY"], 2)
        self.assertEqual(counts["GLD"], 1)
        self.assertEqual(counts["CASH"], 1)

    def test_summarize_dominant_assets_ignores_final_weight_ok_flag(self):
        diagnostics = pd.DataFrame(
            [
                {
                    "final_weight_SPY": 0.20,
                    "final_weight_GLD": 0.70,
                    "final_weight_BTC-USD": 0.10,
                    "final_weight_ok": True,
                }
            ]
        )

        result = summarize_dominant_assets(diagnostics)

        self.assertEqual(result.loc[0, "dominant_asset"], "GLD")
        self.assertNotIn("ok", result["dominant_asset"].tolist())

    def test_summarize_dominant_assets_never_returns_ok_asset(self):
        diagnostics = add_mandate_flags(self._mandate_observations())

        result = summarize_dominant_assets(diagnostics)

        self.assertNotIn("ok", result["dominant_asset"].tolist())

    def test_summarize_dominant_assets_ignores_boolean_final_weight_columns(self):
        diagnostics = pd.DataFrame(
            [
                {
                    "final_weight_SPY": 0.20,
                    "final_weight_GLD": 0.70,
                    "final_weight_BTC-USD": 0.10,
                    "final_weight_signal": True,
                }
            ]
        )

        result = summarize_dominant_assets(diagnostics)

        self.assertEqual(result.loc[0, "dominant_asset"], "GLD")
        self.assertNotIn("signal", result["dominant_asset"].tolist())

    def test_dominant_asset_rates_sum_to_one(self):
        result = summarize_dominant_assets(self._mandate_observations())

        self.assertAlmostEqual(result["rate"].sum(), 1.0)

    def test_missing_final_weight_columns_raise_value_error(self):
        with self.assertRaises(ValueError):
            summarize_dominant_assets(pd.DataFrame({"average_max_weight": [0.5]}))

    def test_invalid_mandate_limits_raise_value_error(self):
        data = self._mandate_observations()

        invalid_kwargs = [
            {"max_drawdown_limit": 0.01},
            {"max_volatility_limit": -0.01},
            {"max_weight_limit": -0.01},
            {"max_weight_limit": 1.01},
            {"min_effective_assets": 0.99},
            {"max_turnover_limit": -0.01},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    add_mandate_flags(data, **kwargs)

    def test_build_mandate_risk_report_saves_csvs_when_output_dir_is_provided(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = self._write_pair(Path(temp_dir), "report")
            output_dir = Path(temp_dir) / "out"

            result = build_mandate_risk_report(
                [paths["metrics"]],
                [paths["diagnostics"]],
                output_dir=str(output_dir),
                report_name="mandate",
            )

            self.assertTrue(Path(result["observations_path"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())
            self.assertTrue(Path(result["dominant_assets_path"]).exists())

    def test_build_mandate_risk_report_returns_none_paths_without_output_dir(self):
        with self._temporary_pair() as paths:
            result = build_mandate_risk_report(
                [paths["metrics"]],
                [paths["diagnostics"]],
                output_dir=None,
            )

        self.assertIsNone(result["observations_path"])
        self.assertIsNone(result["summary_path"])
        self.assertIsNone(result["dominant_assets_path"])

    def _temporary_pair(self, **overrides):
        temp_dir = tempfile.TemporaryDirectory()
        paths = self._write_pair(Path(temp_dir.name), "pair", **overrides)

        class TemporaryPair:
            def __enter__(self_inner):
                return paths

            def __exit__(self_inner, exc_type, exc_value, traceback):
                temp_dir.cleanup()

        return TemporaryPair()

    def _write_pair(self, directory: Path, stem: str, **metric_overrides):
        metrics_path = directory / f"{stem}_metrics.csv"
        diagnostics_path = directory / f"{stem}_diagnostics.csv"

        self._metrics_dataframe(**metric_overrides).to_csv(metrics_path)
        self._diagnostics_dataframe().to_csv(diagnostics_path, index=False)

        return {
            "metrics": str(metrics_path),
            "diagnostics": str(diagnostics_path),
        }

    def _metrics_dataframe(self, index=None, **overrides):
        if index is None:
            index = ["agent", "equal_weight"]
        rows = []
        for _ in index:
            row = {
                "cumulative_return": 0.10,
                "annualized_return": 0.08,
                "annualized_volatility": 0.12,
                "sharpe_ratio": 0.70,
                "max_drawdown": -0.10,
                "sortino_ratio": 1.10,
                "calmar_ratio": 0.80,
                "information_ratio_vs_equal_weight_rebalanced_net": 0.20,
                "capm_alpha_vs_SPY": 0.03,
            }
            row.update(overrides)
            rows.append(row)

        return pd.DataFrame(rows, index=index)

    def _diagnostics_dataframe(self):
        return pd.DataFrame(
            [
                {
                    "average_max_weight": 0.50,
                    "final_max_weight": 0.55,
                    "average_effective_number_of_assets": 2.00,
                    "final_effective_number_of_assets": 1.90,
                    "average_herfindahl_index": 0.50,
                    "final_herfindahl_index": 0.53,
                    "average_turnover": 0.20,
                    "final_turnover": 0.15,
                    "average_transaction_cost": 0.001,
                    "final_transaction_cost": 0.0005,
                    "final_weight_SPY": 0.55,
                    "final_weight_GLD": 0.30,
                    "final_weight_CASH": 0.15,
                }
            ]
        )

    def _mandate_observations(self):
        data = pd.DataFrame(
            [
                self._observation(
                    max_drawdown=-0.15,
                    annualized_volatility=0.15,
                    average_max_weight=0.50,
                    final_max_weight=0.50,
                    average_effective_number_of_assets=2.00,
                    average_turnover=0.20,
                    final_weight_SPY=0.50,
                    final_weight_GLD=0.30,
                    final_weight_CASH=0.20,
                ),
                self._observation(
                    max_drawdown=-0.25,
                    final_weight_SPY=0.20,
                    final_weight_GLD=0.35,
                    final_weight_CASH=0.45,
                ),
                self._observation(
                    average_max_weight=0.90,
                    final_max_weight=0.90,
                    final_weight_SPY=0.90,
                    final_weight_GLD=0.05,
                    final_weight_CASH=0.05,
                ),
                self._observation(
                    average_effective_number_of_assets=1.00,
                    final_effective_number_of_assets=1.05,
                    final_weight_SPY=0.25,
                    final_weight_GLD=0.50,
                    final_weight_CASH=0.25,
                ),
            ]
        )
        return data

    def _observation(self, **overrides):
        row = {
            "cumulative_return": 0.10,
            "annualized_return": 0.08,
            "annualized_volatility": 0.15,
            "sharpe_ratio": 0.70,
            "max_drawdown": -0.15,
            "sortino_ratio": 1.10,
            "calmar_ratio": 0.80,
            "information_ratio_vs_equal_weight_rebalanced_net": 0.20,
            "capm_alpha_vs_SPY": 0.03,
            "average_max_weight": 0.50,
            "final_max_weight": 0.50,
            "average_effective_number_of_assets": 2.00,
            "final_effective_number_of_assets": 1.90,
            "average_herfindahl_index": 0.50,
            "final_herfindahl_index": 0.53,
            "average_turnover": 0.20,
            "final_turnover": 0.15,
            "average_transaction_cost": 0.001,
            "final_transaction_cost": 0.0005,
            "final_weight_SPY": 0.50,
            "final_weight_GLD": 0.30,
            "final_weight_CASH": 0.20,
        }
        row.update(overrides)
        return row


if __name__ == "__main__":
    unittest.main()
