"""Tests for White Reality Check reporting."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.statistical_validation_report import load_date_averaged_return_series
from src.analysis.white_reality_check_report import (
    align_differential_matrix,
    build_white_reality_check_report,
    run_white_reality_check_for_benchmark,
    white_reality_check_statistic,
)


class WhiteRealityCheckReportTests(unittest.TestCase):
    def test_v3_and_v7_clean_candidate_paths_resolve(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_dir = root / "final"
            output_dir = root / "out"
            v3_dir = root / "v3_clean"
            v7_dir = root / "v7_clean_garch"
            bench_dir = root / "bench"
            final_dir.mkdir()
            (bench_dir / "benchmarks" / "histories").mkdir(parents=True)

            self._write_td3_history(
                v3_dir,
                "V3_real_macro_vintage_clean_no_dxy",
                "V3_real_macro_vintage_clean_no_dxy_cap_0.50",
                value=0.02,
            )
            self._write_td3_history(
                v7_dir,
                "V7_real_macro_vintage_clean_no_dxy_garch",
                "V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50",
                value=0.018,
            )
            self._write_history(
                bench_dir / "benchmarks" / "histories" / "BuyHold_GLD_history.csv",
                value=0.005,
            )

            pd.DataFrame(
                [
                    {
                        "strategy_name": "V3_real_macro_vintage_clean_no_dxy_cap_0.50",
                        "base_candidate": "V3_real_macro_vintage_clean_no_dxy",
                        "source": "v3_clean_no_dxy_cap_sensitivity",
                        "selected_cap": 0.50,
                    },
                    {
                        "strategy_name": "V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50",
                        "base_candidate": "V7_real_macro_vintage_clean_no_dxy_garch",
                        "source": "v7_clean_no_dxy_garch_cap_sensitivity",
                        "selected_cap": 0.50,
                    },
                ]
            ).to_csv(final_dir / "final_constrained_td3_selected_candidates.csv", index=False)
            pd.DataFrame(
                [{"strategy_name": "BuyHold_GLD", "strategy_type": "benchmark"}]
            ).to_csv(final_dir / "final_constrained_td3_mandate_ranking.csv", index=False)
            (final_dir / "final_constrained_td3_metadata.json").write_text(
                json.dumps({"benchmark_comparison_dir": str(bench_dir)}),
                encoding="utf-8",
            )

            result = build_white_reality_check_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
                benchmarks=["BuyHold_GLD"],
                n_bootstrap=20,
                block_length=4,
                seed=7,
                v3_clean_no_dxy_cap_sensitivity_dir=str(v3_dir),
                v7_clean_no_dxy_garch_cap_sensitivity_dir=str(v7_dir),
            )

        records = result["history_records"].set_index("strategy_name")
        self.assertTrue(
            bool(records.loc["V3_real_macro_vintage_clean_no_dxy_cap_0.50", "history_found"])
        )
        self.assertTrue(
            bool(
                records.loc[
                    "V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50",
                    "history_found",
                ]
            )
        )
        self.assertFalse(result["summary"].empty)

    def test_zero_differentials_have_high_p_value(self):
        matrix = pd.DataFrame(
            {
                "candidate_a": [0.0] * 40,
                "candidate_b": [0.0] * 40,
            }
        )

        result = white_reality_check_statistic(
            matrix,
            n_bootstrap=50,
            block_length=4,
            seed=11,
        )

        self.assertAlmostEqual(result["observed_statistic"], 0.0)
        self.assertGreater(result["p_value"], 0.90)

    def test_positive_differential_has_lower_p_value(self):
        matrix = pd.DataFrame(
            {
                "candidate_a": [0.01] * 40,
                "candidate_b": [0.0] * 40,
            }
        )

        result = white_reality_check_statistic(
            matrix,
            n_bootstrap=50,
            block_length=4,
            seed=11,
        )

        self.assertGreater(result["observed_statistic"], 0.0)
        self.assertLess(result["p_value"], 0.10)

    def test_date_alignment_uses_overlap_only(self):
        candidate = pd.Series(
            [0.02, 0.03, 0.04],
            index=pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"]),
        )
        benchmark = pd.Series(
            [0.01, 0.01],
            index=pd.to_datetime(["2024-01-12", "2024-01-26"]),
        )
        matrix = align_differential_matrix({"candidate": candidate - benchmark})

        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix.index[0], pd.Timestamp("2024-01-12"))

    def test_date_averaging_duplicates_before_testing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path_a = Path(temp_dir) / "a.csv"
            path_b = Path(temp_dir) / "b.csv"
            dates = pd.date_range("2024-01-05", periods=2, freq="W-FRI")
            pd.DataFrame({"date": dates, "financial_net_return": [0.01, 0.03]}).to_csv(
                path_a,
                index=False,
            )
            pd.DataFrame({"date": dates, "financial_net_return": [0.03, 0.01]}).to_csv(
                path_b,
                index=False,
            )

            series = load_date_averaged_return_series([path_a, path_b])

        self.assertAlmostEqual(series.iloc[0], 0.02)
        self.assertAlmostEqual(series.iloc[1], 0.02)

    def test_outputs_are_created_and_summary_text_is_cautious(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_dir, cap_dir, bench_dir = self._write_minimal_report(root)
            output_dir = root / "out"

            result = build_white_reality_check_report(
                final_report_dir=str(final_dir),
                output_dir=str(output_dir),
                benchmarks=["BuyHold_GLD"],
                n_bootstrap=20,
                block_length=4,
                seed=123,
                v3_clean_no_dxy_cap_sensitivity_dir=str(cap_dir),
            )

            self.assertTrue((output_dir / "white_reality_check_summary.csv").exists())
            self.assertTrue(
                (output_dir / "white_reality_check_candidate_differentials.csv").exists()
            )
            self.assertTrue(
                (output_dir / "white_reality_check_bootstrap_distribution.csv").exists()
            )
            self.assertTrue((output_dir / "white_reality_check_metadata.json").exists())
            self.assertTrue((output_dir / "white_reality_check_summary.md").exists())
            self.assertIn("White Reality Check style", result["markdown"])
            self.assertIn("not an SPA test", result["markdown"])
            self.assertIn("does not imply market dominance", result["markdown"])
            metadata = json.loads((output_dir / "white_reality_check_metadata.json").read_text())
            self.assertEqual(metadata["return_column_used"], "financial_net_return preferred, portfolio_return fallback")

    def test_missing_candidate_histories_warn_without_crashing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_dir, cap_dir, bench_dir = self._write_minimal_report(
                root,
                include_candidate_history=False,
            )

            result = build_white_reality_check_report(
                final_report_dir=str(final_dir),
                output_dir=str(root / "out"),
                benchmarks=["BuyHold_GLD"],
                n_bootstrap=10,
                block_length=2,
                seed=7,
                v3_clean_no_dxy_cap_sensitivity_dir=str(cap_dir),
            )

        self.assertTrue(any("No TD3 test policy histories" in warning for warning in result["warnings"]))

    def test_run_for_benchmark_reports_positive_best_candidate(self):
        dates = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
        histories = {
            "candidate_good": pd.Series([0.02] * 40, index=dates),
            "candidate_flat": pd.Series([0.005] * 40, index=dates),
            "BuyHold_GLD": pd.Series([0.005] * 40, index=dates),
        }

        result = run_white_reality_check_for_benchmark(
            histories=histories,
            candidate_names=["candidate_good", "candidate_flat"],
            benchmark_name="BuyHold_GLD",
            n_bootstrap=20,
            block_length=4,
            seed=3,
        )

        self.assertEqual(result["summary"]["best_candidate_by_mean_diff"], "candidate_good")
        self.assertGreater(result["summary"]["observed_statistic"], 0.0)

    def _write_minimal_report(
        self,
        root: Path,
        include_candidate_history: bool = True,
    ) -> tuple[Path, Path, Path]:
        final_dir = root / "final"
        cap_dir = root / "cap"
        bench_dir = root / "bench"
        final_dir.mkdir()
        (bench_dir / "benchmarks" / "histories").mkdir(parents=True)
        if include_candidate_history:
            self._write_td3_history(
                cap_dir,
                "V3_real_macro_vintage_clean_no_dxy",
                "V3_real_macro_vintage_clean_no_dxy_cap_0.50",
                value=0.02,
            )
        self._write_history(
            bench_dir / "benchmarks" / "histories" / "BuyHold_GLD_history.csv",
            value=0.005,
        )
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
            [{"strategy_name": "BuyHold_GLD", "strategy_type": "benchmark"}]
        ).to_csv(final_dir / "final_constrained_td3_mandate_ranking.csv", index=False)
        (final_dir / "final_constrained_td3_metadata.json").write_text(
            json.dumps({"benchmark_comparison_dir": str(bench_dir)}),
            encoding="utf-8",
        )
        return final_dir, cap_dir, bench_dir

    def _write_td3_history(
        self,
        cap_dir: Path,
        base_candidate: str,
        strategy_name: str,
        value: float,
    ) -> None:
        cap_label = strategy_name.split("_cap_")[-1].replace(".", "p")
        history_dir = (
            cap_dir
            / "per_candidate"
            / base_candidate
            / f"F1_{base_candidate}_cap_{cap_label}_seed_7"
        )
        history_dir.mkdir(parents=True)
        self._write_history(history_dir / "test_policy_history.csv", value=value)

    @staticmethod
    def _write_history(path: Path, value: float) -> None:
        dates = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
        pd.DataFrame(
            {
                "date": dates,
                "financial_net_return": [value] * len(dates),
                "portfolio_return": [value] * len(dates),
            }
        ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
