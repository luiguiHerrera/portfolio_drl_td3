"""Tests for regime analysis reporting."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.regime_analysis_report import (
    REGIME_DEFINITIONS,
    build_regime_analysis_report,
    build_regime_pairwise_comparisons,
    build_regime_strategy_metrics,
    calculate_max_drawdown,
    calculate_return_metrics,
    slice_returns,
)


class RegimeAnalysisReportTests(unittest.TestCase):
    def test_regime_slicing_works_on_synthetic_dated_returns(self):
        returns = self._series("2022-01-07", [0.01, 0.02, -0.01, 0.03])

        sliced = slice_returns(returns, "2022-01-10", "2022-01-28")

        self.assertEqual(len(sliced), 3)
        self.assertEqual(sliced.index.min(), pd.Timestamp("2022-01-14"))
        self.assertEqual(sliced.index.max(), pd.Timestamp("2022-01-28"))

    def test_metrics_calculate_correctly(self):
        returns = self._series("2022-01-07", [0.01, 0.02, -0.01])

        metrics = calculate_return_metrics(returns)

        self.assertEqual(metrics["n_observations"], 3)
        self.assertAlmostEqual(metrics["cumulative_return"], (1.01 * 1.02 * 0.99) - 1.0)
        self.assertAlmostEqual(metrics["hit_rate"], 2 / 3)
        self.assertEqual(metrics["worst_period_return"], -0.01)

    def test_max_drawdown_calculation_works(self):
        returns = self._series("2022-01-07", [0.10, -0.20, 0.05])

        max_drawdown = calculate_max_drawdown(returns)

        self.assertAlmostEqual(max_drawdown, -0.20)

    def test_pairwise_comparisons_work(self):
        histories = {
            "left": self._series("2022-01-07", ([0.02, -0.002] * 26)),
            "right": self._series("2022-01-07", ([0.01, -0.004] * 26)),
        }
        metrics = build_regime_strategy_metrics(histories)

        pairwise = build_regime_pairwise_comparisons(metrics, [("left", "right")])
        available = pairwise[pairwise["comparison_available"] == True]

        self.assertFalse(available.empty)
        self.assertTrue(bool(available.iloc[0]["left_beats_right_by_sharpe"]))

    def test_missing_histories_are_reported_as_warnings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            final_dir = self._write_synthetic_report(temp_dir, include_v3_history=False)
            output_dir = Path(temp_dir) / "out"

            result = build_regime_analysis_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
            )

        self.assertTrue(any("history" in warning.lower() for warning in result["warnings"]))

    def test_summary_markdown_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            final_dir = self._write_synthetic_report(temp_dir)
            output_dir = Path(temp_dir) / "out"

            result = build_regime_analysis_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
            )

            self.assertTrue(Path(result["paths"]["summary"]).exists())
            self.assertIn("Regime Analysis Report", result["summary"])

    def test_metadata_records_regime_definitions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            final_dir = self._write_synthetic_report(temp_dir)
            output_dir = Path(temp_dir) / "out"

            result = build_regime_analysis_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
            )
            metadata = json.loads(Path(result["paths"]["metadata"]).read_text())

        self.assertEqual(len(metadata["regimes"]), len(REGIME_DEFINITIONS))
        self.assertEqual(metadata["history_policy"].split()[0], "TD3")

    def test_v3_clean_no_dxy_history_resolves_from_optional_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_dir = root / "final"
            output_dir = root / "out"
            base_cap_dir = root / "base_cap"
            clean_cap_dir = root / "clean_cap"
            bench_dir = root / "bench"
            final_dir.mkdir()
            base_cap_dir.mkdir()
            (bench_dir / "benchmarks" / "histories").mkdir(parents=True)

            selected = pd.DataFrame(
                [
                    {
                        "strategy_name": "V3_real_macro_vintage_clean_no_dxy_cap_0.50",
                        "base_candidate": "V3_real_macro_vintage_clean_no_dxy",
                        "selected_cap": 0.50,
                        "strategy_group": "td3_best_constrained",
                    }
                ]
            )
            selected.to_csv(final_dir / "final_constrained_td3_selected_candidates.csv", index=False)
            metadata = {
                "cap_sensitivity_dir": str(base_cap_dir),
                "benchmark_comparison_dir": str(bench_dir),
            }
            (final_dir / "final_constrained_td3_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            clean_history_dir = (
                clean_cap_dir
                / "per_candidate"
                / "V3_real_macro_vintage_clean_no_dxy"
                / "F1_V3_real_macro_vintage_clean_no_dxy_cap_0p50_seed_7"
            )
            clean_history_dir.mkdir(parents=True)
            self._history_frame("2022-01-07", [0.01] * 52).to_csv(
                clean_history_dir / "test_policy_history.csv",
                index=False,
            )
            self._history_frame("2022-01-07", [0.005] * 52).to_csv(
                bench_dir / "benchmarks" / "histories" / "BuyHold_GLD_history.csv",
                index=False,
            )

            result = build_regime_analysis_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
                v3_clean_no_dxy_cap_sensitivity_dir=str(clean_cap_dir),
            )

        strategy_name = "V3_real_macro_vintage_clean_no_dxy_cap_0.50"
        self.assertIn(strategy_name, result["histories"])
        self.assertEqual(
            result["history_sources"][strategy_name]["source_dir"],
            str(clean_cap_dir),
        )
        self.assertIn(
            "V3_real_macro_vintage_clean_no_dxy",
            result["history_sources"][strategy_name]["history_files_sample"][0],
        )
        self.assertFalse(any(strategy_name in warning for warning in result["warnings"]))

    def test_v7_clean_no_dxy_garch_history_resolves_from_optional_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_dir = root / "final"
            output_dir = root / "out"
            base_cap_dir = root / "base_cap"
            clean_garch_cap_dir = root / "clean_garch_cap"
            bench_dir = root / "bench"
            final_dir.mkdir()
            base_cap_dir.mkdir()
            (bench_dir / "benchmarks" / "histories").mkdir(parents=True)

            strategy_name = "V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50"
            selected = pd.DataFrame(
                [
                    {
                        "strategy_name": strategy_name,
                        "base_candidate": "V7_real_macro_vintage_clean_no_dxy_garch",
                        "selected_cap": 0.50,
                        "strategy_group": "td3_evaluated_constrained",
                    }
                ]
            )
            selected.to_csv(final_dir / "final_constrained_td3_selected_candidates.csv", index=False)
            metadata = {
                "cap_sensitivity_dir": str(base_cap_dir),
                "benchmark_comparison_dir": str(bench_dir),
            }
            (final_dir / "final_constrained_td3_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            clean_garch_history_dir = (
                clean_garch_cap_dir
                / "per_candidate"
                / "V7_real_macro_vintage_clean_no_dxy_garch"
                / "F1_V7_real_macro_vintage_clean_no_dxy_garch_cap_0p50_seed_7"
            )
            clean_garch_history_dir.mkdir(parents=True)
            self._history_frame("2022-01-07", [0.01] * 52).to_csv(
                clean_garch_history_dir / "test_policy_history.csv",
                index=False,
            )
            self._history_frame("2022-01-07", [0.005] * 52).to_csv(
                bench_dir / "benchmarks" / "histories" / "BuyHold_GLD_history.csv",
                index=False,
            )

            result = build_regime_analysis_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
                v7_clean_no_dxy_garch_cap_sensitivity_dir=str(clean_garch_cap_dir),
            )

        self.assertIn(strategy_name, result["histories"])
        self.assertEqual(
            result["history_sources"][strategy_name]["source_dir"],
            str(clean_garch_cap_dir),
        )
        self.assertIn(
            "V7_real_macro_vintage_clean_no_dxy_garch",
            result["history_sources"][strategy_name]["history_files_sample"][0],
        )
        pairwise = result["pairwise"]
        self.assertTrue(
            (
                pairwise["left_strategy"]
                == "V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50"
            ).any()
        )
        self.assertFalse(any(strategy_name in warning for warning in result["warnings"]))

    def test_asset_specific_report_resolves_and_rejects_scalar_histories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_dir = root / "asset_report"
            source_dir = root / "asset_source"
            benchmark_dir = root / "benchmarks"
            output_dir = root / "out"
            final_dir.mkdir()
            history_dir = (
                source_dir
                / "per_candidate"
                / "V5_no_volatility_block"
                / "F1_V5_no_volatility_block_cap_0p50_seed_7"
            )
            history_dir.mkdir(parents=True)
            (benchmark_dir / "histories").mkdir(parents=True)
            self._asset_specific_history_frame("2022-01-07", [0.01] * 52).to_csv(
                history_dir / "test_policy_history.csv",
                index=False,
            )
            for benchmark in ["trend_spy_cash_12p", "BuyHold_GLD", "Equal_Weight"]:
                self._asset_specific_history_frame("2022-01-07", [0.005] * 52).to_csv(
                    benchmark_dir / "histories" / f"{benchmark}_history.csv",
                    index=False,
                )
            pd.DataFrame(
                [
                    {
                        "candidate_name": "V5_no_volatility_block_cap_0p50",
                        "base_candidate": "V5_no_volatility_block",
                        "max_weight_cap": 0.50,
                    }
                ]
            ).to_csv(final_dir / "asset_specific_cost_selected_candidates.csv", index=False)
            (final_dir / "asset_specific_cost_metadata.json").write_text(
                json.dumps(
                    {
                        "runner": "src.analysis.asset_specific_cost_final_report",
                        "source_dirs": {"v2_v6": str(source_dir)},
                    }
                ),
                encoding="utf-8",
            )

            result = build_regime_analysis_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
                benchmark_dir=str(benchmark_dir),
            )

            self.assertIn("V5_no_volatility_block_cap_0p50", result["histories"])
            self.assertIn("trend_spy_cash_12p", result["histories"])
            self.assertFalse(result["pairwise"].empty)

            scalar_path = benchmark_dir / "histories" / "BuyHold_GLD_history.csv"
            pd.read_csv(scalar_path).drop(columns=["transaction_cost_mode"]).to_csv(
                scalar_path,
                index=False,
            )
            with self.assertRaisesRegex(ValueError, "missing transaction_cost_mode"):
                build_regime_analysis_report(
                    final_report_dir=str(final_dir),
                    output_dir=str(output_dir),
                    benchmark_dir=str(benchmark_dir),
                )

    def _write_synthetic_report(
        self,
        temp_dir: str,
        include_v3_history: bool = True,
    ) -> Path:
        root = Path(temp_dir)
        final_dir = root / "final"
        cap_dir = root / "cap"
        bench_dir = root / "bench"
        final_dir.mkdir()
        (bench_dir / "benchmarks" / "histories").mkdir(parents=True)
        selected = pd.DataFrame(
            [
                {
                    "strategy_name": "V3_cap_0.60",
                    "base_candidate": "V3_real_macro_current",
                    "selected_cap": 0.60,
                    "strategy_group": "td3_best_constrained",
                }
            ]
        )
        selected.to_csv(final_dir / "final_constrained_td3_selected_candidates.csv", index=False)
        metadata = {
            "v3_cap_sensitivity_dir": str(cap_dir),
            "benchmark_comparison_dir": str(bench_dir),
        }
        (final_dir / "final_constrained_td3_metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        if include_v3_history:
            history_dir = (
                cap_dir
                / "per_candidate"
                / "V3_real_macro_current"
                / "F1_V3_real_macro_current_cap_0p60_seed_7"
            )
            history_dir.mkdir(parents=True)
            self._history_frame("2022-01-07", [0.01] * 52).to_csv(
                history_dir / "test_policy_history.csv",
                index=False,
            )
        self._history_frame("2022-01-07", [0.005] * 52).to_csv(
            bench_dir / "benchmarks" / "histories" / "BuyHold_GLD_history.csv",
            index=False,
        )
        return final_dir

    @staticmethod
    def _series(start: str, values: list[float]) -> pd.Series:
        index = pd.date_range(start, periods=len(values), freq="W-FRI")
        return pd.Series(values, index=index, name="return")

    @staticmethod
    def _history_frame(start: str, values: list[float]) -> pd.DataFrame:
        index = pd.date_range(start, periods=len(values), freq="W-FRI")
        return pd.DataFrame({"date": index, "financial_net_return": values})

    @staticmethod
    def _asset_specific_history_frame(start: str, values: list[float]) -> pd.DataFrame:
        index = pd.date_range(start, periods=len(values), freq="W-FRI")
        frame = pd.DataFrame(
            {
                "date": index,
                "financial_net_return": values,
                "portfolio_return": values,
                "transaction_cost_mode": ["asset_specific"] * len(values),
                "transaction_cost": [0.0001] * len(values),
                "turnover": [0.1] * len(values),
            }
        )
        for asset in ["SPY", "TLT", "GLD", "BTC-USD", "CASH"]:
            frame[f"asset_turnover_{asset}"] = 0.0
            frame[f"asset_transaction_cost_contribution_{asset}"] = 0.0
        return frame


if __name__ == "__main__":
    unittest.main()
