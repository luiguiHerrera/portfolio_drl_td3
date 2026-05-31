"""Tests for statistical validation report."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.statistical_validation_report import (
    align_return_pair,
    build_statistical_validation_report,
    calculate_max_drawdown,
    compute_return_metrics,
    load_date_averaged_return_series,
    paired_block_bootstrap_deltas,
)


class StatisticalValidationReportTests(unittest.TestCase):
    def test_bootstrap_ci_metrics_work_on_synthetic_returns(self):
        returns = pd.Series([0.01, 0.02, -0.005, 0.015] * 20)

        metrics = compute_return_metrics(returns)

        self.assertIn("annualized_return", metrics)
        self.assertIn("sharpe", metrics)
        self.assertGreater(metrics["cumulative_return"], 0.0)

    def test_paired_bootstrap_detects_positive_delta(self):
        candidate = pd.Series([0.02, 0.015, 0.01, 0.005] * 20)
        benchmark = pd.Series([0.005, 0.004, 0.003, 0.002] * 20)

        deltas = paired_block_bootstrap_deltas(
            candidate,
            benchmark,
            n_bootstrap=100,
            block_size=4,
            random_seed=7,
        )

        probability = sum(delta["cumulative_return"] > 0 for delta in deltas) / len(deltas)
        self.assertGreater(probability, 0.95)

    def test_paired_bootstrap_detects_uncertain_delta(self):
        candidate = pd.Series([0.01, -0.01] * 40)
        benchmark = pd.Series([0.009, -0.009] * 40)

        deltas = paired_block_bootstrap_deltas(
            candidate,
            benchmark,
            n_bootstrap=100,
            block_size=2,
            random_seed=11,
        )

        probability = sum(delta["sharpe"] > 0 for delta in deltas) / len(deltas)
        self.assertLess(probability, 0.95)

    def test_max_drawdown_calculation(self):
        returns = pd.Series([0.10, -0.20, 0.05])

        self.assertAlmostEqual(calculate_max_drawdown(returns), -0.20)

    def test_date_averaged_histories_average_duplicate_dates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path_a = Path(temp_dir) / "a.csv"
            path_b = Path(temp_dir) / "b.csv"
            dates = pd.date_range("2024-01-05", periods=2, freq="W-FRI")
            pd.DataFrame(
                {
                    "date": dates,
                    "financial_net_return": [0.01, 0.03],
                }
            ).to_csv(path_a, index=False)
            pd.DataFrame(
                {
                    "date": dates,
                    "financial_net_return": [0.03, 0.01],
                }
            ).to_csv(path_b, index=False)

            series = load_date_averaged_return_series([path_a, path_b])

        self.assertAlmostEqual(series.iloc[0], 0.02)
        self.assertAlmostEqual(series.iloc[1], 0.02)

    def test_align_return_pair_uses_common_dates(self):
        dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
        candidate = pd.Series([0.01, 0.02, 0.03], index=dates)
        benchmark = pd.Series([0.01, 0.02], index=dates[1:])

        aligned = align_return_pair(candidate, benchmark)

        self.assertEqual(len(aligned), 2)
        self.assertEqual(list(aligned.columns), ["candidate", "benchmark"])

    def test_report_writes_outputs_and_warns_on_missing_histories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            final_dir = Path(temp_dir) / "final"
            output_dir = Path(temp_dir) / "out"
            cap_dir = Path(temp_dir) / "cap"
            bench_dir = Path(temp_dir) / "bench"
            final_dir.mkdir()
            cap_dir.mkdir()
            bench_history_dir = bench_dir / "benchmarks" / "histories"
            bench_history_dir.mkdir(parents=True)

            selected = pd.DataFrame(
                [
                    {
                        "strategy_name": "V3_cap_0.60",
                        "base_candidate": "V3_real_macro_current",
                        "source": "seeded_cap_sensitivity",
                        "selected_cap": 0.60,
                    }
                ]
            )
            selected.to_csv(final_dir / "final_constrained_td3_selected_candidates.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "strategy_name": "BuyHold_GLD",
                        "strategy_type": "benchmark",
                    }
                ]
            ).to_csv(final_dir / "final_constrained_td3_mandate_ranking.csv", index=False)
            metadata = {
                "v3_cap_sensitivity_dir": str(cap_dir),
                "benchmark_comparison_dir": str(bench_dir),
            }
            (final_dir / "final_constrained_td3_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            self._write_history(bench_history_dir / "BuyHold_GLD_history.csv")

            report = build_statistical_validation_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
                n_bootstrap=20,
                block_size=2,
            )

            self.assertTrue((output_dir / "statistical_validation_metric_ci.csv").exists())
            self.assertTrue((output_dir / "statistical_validation_pairwise_bootstrap.csv").exists())
            self.assertTrue((output_dir / "statistical_validation_summary.md").exists())
            self.assertTrue(any("No TD3 test policy histories" in warning for warning in report["warnings"]))

    def test_report_creates_pairwise_table_when_histories_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            final_dir = Path(temp_dir) / "final"
            output_dir = Path(temp_dir) / "out"
            cap_dir = Path(temp_dir) / "cap"
            bench_dir = Path(temp_dir) / "bench"
            final_dir.mkdir()
            cap_dir.mkdir()
            td3_dir = (
                cap_dir
                / "per_candidate"
                / "V3_real_macro_current"
                / "F1_V3_real_macro_current_cap_0p60_seed_7"
            )
            td3_dir.mkdir(parents=True)
            bench_history_dir = bench_dir / "benchmarks" / "histories"
            bench_history_dir.mkdir(parents=True)
            self._write_history(td3_dir / "test_policy_history.csv", value=0.02)
            self._write_history(bench_history_dir / "BuyHold_GLD_history.csv", value=0.005)
            self._write_history(bench_history_dir / "trend_spy_cash_12p_history.csv", value=0.004)

            pd.DataFrame(
                [
                    {
                        "strategy_name": "V3_cap_0.60",
                        "base_candidate": "V3_real_macro_current",
                        "source": "seeded_cap_sensitivity",
                        "selected_cap": 0.60,
                    }
                ]
            ).to_csv(final_dir / "final_constrained_td3_selected_candidates.csv", index=False)
            pd.DataFrame(
                [
                    {"strategy_name": "BuyHold_GLD", "strategy_type": "benchmark"},
                    {"strategy_name": "trend_spy_cash_12p", "strategy_type": "benchmark"},
                ]
            ).to_csv(final_dir / "final_constrained_td3_mandate_ranking.csv", index=False)
            (final_dir / "final_constrained_td3_metadata.json").write_text(
                json.dumps(
                    {
                        "v3_cap_sensitivity_dir": str(cap_dir),
                        "benchmark_comparison_dir": str(bench_dir),
                    }
                ),
                encoding="utf-8",
            )

            report = build_statistical_validation_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
                n_bootstrap=20,
                block_size=2,
            )

        self.assertFalse(report["pairwise_bootstrap"].empty)
        self.assertIn("V3_cap_0.60", set(report["pairwise_bootstrap"]["candidate"]))

    def test_v3_clean_no_dxy_history_resolves_from_optional_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            final_dir = Path(temp_dir) / "final"
            output_dir = Path(temp_dir) / "out"
            base_cap_dir = Path(temp_dir) / "base_cap"
            clean_cap_dir = Path(temp_dir) / "clean_cap"
            bench_dir = Path(temp_dir) / "bench"
            final_dir.mkdir()
            base_cap_dir.mkdir()
            clean_history_dir = (
                clean_cap_dir
                / "per_candidate"
                / "V3_real_macro_vintage_clean_no_dxy"
                / "F1_V3_real_macro_vintage_clean_no_dxy_cap_0p50_seed_7"
            )
            clean_history_dir.mkdir(parents=True)
            bench_history_dir = bench_dir / "benchmarks" / "histories"
            bench_history_dir.mkdir(parents=True)
            self._write_history(clean_history_dir / "test_policy_history.csv", value=0.02)
            self._write_history(bench_history_dir / "BuyHold_GLD_history.csv", value=0.005)

            pd.DataFrame(
                [
                    {
                        "strategy_name": "V3_real_macro_vintage_clean_no_dxy_cap_0.50",
                        "base_candidate": "V3_real_macro_vintage_clean_no_dxy",
                        "source": "v3_clean_no_dxy_cap_sensitivity",
                        "selected_cap": 0.50,
                    }
                ]
            ).to_csv(final_dir / "final_constrained_td3_selected_candidates.csv", index=False)
            pd.DataFrame(
                [
                    {"strategy_name": "BuyHold_GLD", "strategy_type": "benchmark"},
                ]
            ).to_csv(final_dir / "final_constrained_td3_mandate_ranking.csv", index=False)
            (final_dir / "final_constrained_td3_metadata.json").write_text(
                json.dumps(
                    {
                        "cap_sensitivity_dir": str(base_cap_dir),
                        "benchmark_comparison_dir": str(bench_dir),
                    }
                ),
                encoding="utf-8",
            )

            report = build_statistical_validation_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
                n_bootstrap=20,
                block_size=2,
                v3_clean_no_dxy_cap_sensitivity_dir=str(clean_cap_dir),
            )

        records = report["history_records"].set_index("strategy_name")
        strategy_name = "V3_real_macro_vintage_clean_no_dxy_cap_0.50"
        self.assertTrue(bool(records.loc[strategy_name, "history_found"]))
        self.assertIn(
            "V3_real_macro_vintage_clean_no_dxy",
            records.loc[strategy_name, "source"],
        )
        self.assertFalse(
            any(strategy_name in warning for warning in report["warnings"]),
        )

    def test_v7_clean_no_dxy_garch_history_resolves_from_optional_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            final_dir = Path(temp_dir) / "final"
            output_dir = Path(temp_dir) / "out"
            base_cap_dir = Path(temp_dir) / "base_cap"
            clean_garch_cap_dir = Path(temp_dir) / "clean_garch_cap"
            bench_dir = Path(temp_dir) / "bench"
            final_dir.mkdir()
            base_cap_dir.mkdir()
            clean_garch_history_dir = (
                clean_garch_cap_dir
                / "per_candidate"
                / "V7_real_macro_vintage_clean_no_dxy_garch"
                / "F1_V7_real_macro_vintage_clean_no_dxy_garch_cap_0p50_seed_7"
            )
            clean_garch_history_dir.mkdir(parents=True)
            bench_history_dir = bench_dir / "benchmarks" / "histories"
            bench_history_dir.mkdir(parents=True)
            self._write_history(clean_garch_history_dir / "test_policy_history.csv", value=0.02)
            self._write_history(bench_history_dir / "BuyHold_GLD_history.csv", value=0.005)

            strategy_name = "V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50"
            pd.DataFrame(
                [
                    {
                        "strategy_name": strategy_name,
                        "base_candidate": "V7_real_macro_vintage_clean_no_dxy_garch",
                        "source": "v7_clean_no_dxy_garch_cap_sensitivity",
                        "selected_cap": 0.50,
                    }
                ]
            ).to_csv(final_dir / "final_constrained_td3_selected_candidates.csv", index=False)
            pd.DataFrame(
                [
                    {"strategy_name": "BuyHold_GLD", "strategy_type": "benchmark"},
                ]
            ).to_csv(final_dir / "final_constrained_td3_mandate_ranking.csv", index=False)
            (final_dir / "final_constrained_td3_metadata.json").write_text(
                json.dumps(
                    {
                        "cap_sensitivity_dir": str(base_cap_dir),
                        "benchmark_comparison_dir": str(bench_dir),
                    }
                ),
                encoding="utf-8",
            )

            report = build_statistical_validation_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
                n_bootstrap=20,
                block_size=2,
                v7_clean_no_dxy_garch_cap_sensitivity_dir=str(clean_garch_cap_dir),
            )

        records = report["history_records"].set_index("strategy_name")
        self.assertTrue(bool(records.loc[strategy_name, "history_found"]))
        self.assertIn(
            "V7_real_macro_vintage_clean_no_dxy_garch",
            records.loc[strategy_name, "source"],
        )
        self.assertIn(strategy_name, set(report["pairwise_bootstrap"]["candidate"]))
        self.assertFalse(
            any(strategy_name in warning for warning in report["warnings"]),
        )

    def test_asset_specific_report_resolves_histories_and_rejects_scalar(self):
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
            self._write_asset_specific_history(
                history_dir / "test_policy_history.csv",
                value=0.02,
            )
            for benchmark in ["trend_spy_cash_12p", "BuyHold_GLD", "Equal_Weight"]:
                self._write_asset_specific_history(
                    benchmark_dir / "histories" / f"{benchmark}_history.csv",
                    value=0.005,
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

            report = build_statistical_validation_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
                benchmark_dir=str(benchmark_dir),
                n_bootstrap=20,
                block_size=2,
            )

            records = report["history_records"].set_index("strategy_name")
            self.assertTrue(bool(records.loc["V5_no_volatility_block_cap_0p50", "history_found"]))
            self.assertFalse(report["pairwise_bootstrap"].empty)

            scalar_path = benchmark_dir / "histories" / "BuyHold_GLD_history.csv"
            pd.read_csv(scalar_path).drop(columns=["transaction_cost_mode"]).to_csv(
                scalar_path,
                index=False,
            )
            with self.assertRaisesRegex(ValueError, "missing transaction_cost_mode"):
                build_statistical_validation_report(
                    final_report_dir=str(final_dir),
                    output_dir=str(output_dir),
                    benchmark_dir=str(benchmark_dir),
                    n_bootstrap=20,
                    block_size=2,
                )

    @staticmethod
    def _write_history(path: Path, value: float = 0.01) -> None:
        dates = pd.date_range("2024-01-05", periods=20, freq="W-FRI")
        pd.DataFrame(
            {
                "date": dates,
                "financial_net_return": [value] * len(dates),
                "portfolio_return": [value] * len(dates),
            }
        ).to_csv(path, index=False)

    @staticmethod
    def _write_asset_specific_history(path: Path, value: float = 0.01) -> None:
        dates = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
        frame = pd.DataFrame(
            {
                "date": dates,
                "financial_net_return": [value] * len(dates),
                "portfolio_return": [value] * len(dates),
                "transaction_cost_mode": ["asset_specific"] * len(dates),
                "transaction_cost": [0.0001] * len(dates),
                "turnover": [0.1] * len(dates),
            }
        )
        for asset in ["SPY", "TLT", "GLD", "BTC-USD", "CASH"]:
            frame[f"asset_turnover_{asset}"] = 0.0
            frame[f"asset_transaction_cost_contribution_{asset}"] = 0.0
        frame.to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
