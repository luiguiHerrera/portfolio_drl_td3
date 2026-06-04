import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.asset_specific_constraint_pareto_report import (
    build_constraint_pareto_report,
    build_constraint_pass_fail_matrix,
    build_feasible_strategy_rankings,
    load_and_prepare_universe,
    pareto_non_dominated_mask,
)


class AssetSpecificConstraintParetoReportTest(unittest.TestCase):
    def test_constraint_filtering_and_feasible_ranking(self):
        strategies = _strategy_frame()
        matrix = build_constraint_pass_fail_matrix(strategies)
        feasible = build_feasible_strategy_rankings(matrix)

        conservative = feasible[feasible["profile"] == "conservative"]
        moderate = feasible[feasible["profile"] == "moderate"]

        self.assertEqual(conservative.iloc[0]["strategy_name"], "V4_real_garch_current_cap_0p50")
        self.assertEqual(moderate.iloc[0]["strategy_name"], "V5_no_volatility_block_cap_0p50")
        v5_conservative = matrix[
            (matrix["profile"] == "conservative")
            & (matrix["strategy_name"] == "V5_no_volatility_block_cap_0p50")
        ].iloc[0]
        self.assertFalse(bool(v5_conservative["feasible"]))
        self.assertIn("max_drawdown", v5_conservative["failed_constraints"])
        self.assertNotIn("average_max_weight", v5_conservative["failed_constraints"])

    def test_hard_filter_does_not_fail_on_average_max_weight(self):
        strategy = pd.DataFrame(
            [
                _row(
                    "high_weight_but_diversified",
                    "td3",
                    sharpe=0.7,
                    calmar=2.0,
                    sortino=1.4,
                    annualized_return=0.07,
                    annualized_volatility=0.09,
                    max_drawdown=-0.08,
                    average_turnover=0.04,
                    mean_transaction_cost=0.00001,
                    average_effective_number_of_assets=3.2,
                    average_max_weight=0.95,
                )
            ]
        )
        matrix = build_constraint_pass_fail_matrix(strategy)
        conservative = matrix[matrix["profile"] == "conservative"].iloc[0]

        self.assertTrue(bool(conservative["feasible"]))
        self.assertNotIn("average_max_weight_pass", matrix.columns)

    def test_pareto_frontier_logic(self):
        data = pd.DataFrame(
            {
                "sharpe": [1.0, 0.8, 1.1],
                "drawdown_severity": [0.10, 0.20, 0.11],
                "average_turnover": [0.10, 0.30, 0.11],
            }
        )
        mask = pareto_non_dominated_mask(
            data,
            {
                "sharpe": "max",
                "drawdown_severity": "min",
                "average_turnover": "min",
            },
        )
        self.assertEqual(mask.tolist(), [True, False, True])

    def test_benchmark_history_enrichment_for_missing_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            combined = root / "combined.csv"
            benchmark_dir = root / "benchmarks"
            histories = benchmark_dir / "histories"
            histories.mkdir(parents=True)
            data = _strategy_frame()
            data.loc[data["strategy_name"] == "trend_spy_cash_12p", "average_max_weight"] = pd.NA
            data.loc[
                data["strategy_name"] == "trend_spy_cash_12p",
                "average_effective_number_of_assets",
            ] = pd.NA
            data.to_csv(combined, index=False)
            _write_benchmark_history(histories / "trend_spy_cash_12p_history.csv")

            loaded, notes = load_and_prepare_universe(combined, benchmark_dir)

            trend = loaded[loaded["strategy_name"] == "trend_spy_cash_12p"].iloc[0]
            self.assertAlmostEqual(trend["average_max_weight"], 0.6)
            self.assertGreater(trend["average_effective_number_of_assets"], 1.0)
            self.assertTrue(notes)

    def test_missing_required_diagnostics_fail_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            combined = root / "combined.csv"
            benchmark_dir = root / "benchmarks"
            data = _strategy_frame()
            data.loc[data["strategy_name"] == "V5_no_volatility_block_cap_0p50", "average_turnover"] = pd.NA
            data.to_csv(combined, index=False)

            with self.assertRaisesRegex(ValueError, "Missing required diagnostics"):
                load_and_prepare_universe(combined, benchmark_dir)

    def test_report_outputs_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            combined = root / "combined.csv"
            benchmark_dir = root / "benchmarks"
            stat_dir = root / "stat"
            wrc_dir = root / "wrc"
            output_dir = root / "out"
            benchmark_dir.mkdir()
            stat_dir.mkdir()
            wrc_dir.mkdir()
            _strategy_frame().to_csv(combined, index=False)
            _write_statistical_outputs(stat_dir, wrc_dir)

            result = build_constraint_pareto_report(
                combined_ranking_path=str(combined),
                benchmark_dir=str(benchmark_dir),
                statistical_validation_dir=str(stat_dir),
                white_reality_check_dir=str(wrc_dir),
                output_dir=str(output_dir),
            )

            expected = [
                "feasible_strategy_rankings.csv",
                "constraint_pass_fail_matrix.csv",
                "standard_metric_rankings.csv",
                "pareto_frontier.csv",
                "pareto_dominated_strategies.csv",
                "constraint_pareto_summary.md",
                "constraint_pareto_metadata.json",
            ]
            for filename in expected:
                self.assertTrue((output_dir / filename).exists(), filename)
            self.assertIn("does not use custom", result["summary"])
            metadata = json.loads((output_dir / "constraint_pareto_metadata.json").read_text())
            self.assertEqual(metadata["mandate_profile_source"], "src/risk/mandate_profiles.py")
            self.assertIn("canonical_mandate_profiles", metadata)
            self.assertIn("not official", metadata["max_weight_mandate_note"])
            self.assertNotIn("max_weight_limit", metadata["canonical_mandate_profiles"]["conservative"])

    def test_corrected_combined_report_metadata_validates_counts_and_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            combined = root / "final_corrected_zero_cash_combined_ranking.csv"
            benchmark_dir = root / "benchmarks"
            stat_dir = root / "stat"
            wrc_dir = root / "wrc"
            output_dir = root / "out"
            benchmark_dir.mkdir()
            stat_dir.mkdir()
            wrc_dir.mkdir()
            rows = []
            for idx in range(5):
                rows.append(
                    _row(
                        f"V{idx}_candidate_cap_0p50",
                        "TD3",
                        sharpe=0.8,
                        calmar=1.0,
                        sortino=1.2,
                        annualized_return=0.08,
                        annualized_volatility=0.12,
                        max_drawdown=-0.12,
                        average_turnover=0.08,
                        mean_transaction_cost=0.00002,
                        average_effective_number_of_assets=2.5,
                        average_max_weight=0.5,
                    )
                )
            for idx in range(14):
                rows.append(
                    _row(
                        f"benchmark_{idx:02d}",
                        "benchmark",
                        sharpe=0.6,
                        calmar=0.8,
                        sortino=1.0,
                        annualized_return=0.06,
                        annualized_volatility=0.10,
                        max_drawdown=-0.10,
                        average_turnover=0.02,
                        mean_transaction_cost=0.00001,
                        average_effective_number_of_assets=2.0,
                        average_max_weight=0.6,
                    )
                )
            pd.DataFrame(rows).to_csv(combined, index=False)
            (root / "final_corrected_zero_cash_benchmark_comparison_metadata.json").write_text(
                json.dumps(
                    {
                        "transaction_cost_mode": "asset_specific",
                        "asset_transaction_cost_bps": {
                            "SPY": 2.0,
                            "TLT": 2.0,
                            "GLD": 2.0,
                            "BTC-USD": 10.0,
                            "CASH": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            _write_statistical_outputs(stat_dir, wrc_dir)

            result = build_constraint_pareto_report(
                combined_ranking_path=str(combined),
                benchmark_dir=str(benchmark_dir),
                statistical_validation_dir=str(stat_dir),
                white_reality_check_dir=str(wrc_dir),
                output_dir=str(output_dir),
            )

            validation = result["metadata"]["input_validation"]
            self.assertEqual(validation["selected_td3_count"], 5)
            self.assertEqual(validation["benchmark_count"], 14)
            self.assertEqual(validation["cash_bps"], 0.0)


def _strategy_frame() -> pd.DataFrame:
    rows = [
        _row(
            "V5_no_volatility_block_cap_0p50",
            "td3",
            sharpe=1.05,
            calmar=3.1,
            sortino=2.7,
            annualized_return=0.12,
            annualized_volatility=0.14,
            max_drawdown=-0.11,
            average_turnover=0.08,
            mean_transaction_cost=0.00003,
            average_effective_number_of_assets=3.16,
            average_max_weight=0.95,
        ),
        _row(
            "V4_real_garch_current_cap_0p50",
            "td3",
            sharpe=0.96,
            calmar=3.0,
            sortino=2.6,
            annualized_return=0.11,
            annualized_volatility=0.09,
            max_drawdown=-0.09,
            average_turnover=0.04,
            mean_transaction_cost=0.00003,
            average_effective_number_of_assets=3.16,
            average_max_weight=0.90,
        ),
        _row(
            "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
            "td3",
            sharpe=0.95,
            calmar=2.8,
            sortino=2.5,
            annualized_return=0.09,
            annualized_volatility=0.10,
            max_drawdown=-0.08,
            average_turnover=0.06,
            mean_transaction_cost=0.00002,
            average_effective_number_of_assets=1.95,
            average_max_weight=0.70,
        ),
        _row(
            "trend_spy_cash_12p",
            "benchmark",
            sharpe=0.88,
            calmar=0.55,
            sortino=1.2,
            annualized_return=0.10,
            annualized_volatility=0.18,
            max_drawdown=-0.18,
            average_turnover=0.12,
            mean_transaction_cost=0.00002,
            average_effective_number_of_assets=1.0,
            average_max_weight=1.0,
        ),
    ]
    data = pd.DataFrame(rows)
    data["drawdown_severity"] = data["max_drawdown"].abs()
    return data


def _row(
    strategy_name,
    strategy_type,
    *,
    sharpe,
    calmar,
    sortino,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    average_turnover,
    mean_transaction_cost,
    average_effective_number_of_assets,
    average_max_weight,
):
    return {
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "strategy_group": "td3_selected" if strategy_type == "td3" else "benchmark",
        "transaction_cost_mode": "asset_specific",
        "sharpe": sharpe,
        "calmar": calmar,
        "sortino": sortino,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "average_turnover": average_turnover,
        "mean_transaction_cost": mean_transaction_cost,
        "average_effective_number_of_assets": average_effective_number_of_assets,
        "average_max_weight": average_max_weight,
    }


def _write_benchmark_history(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "date": "2024-01-05",
                "transaction_cost_mode": "asset_specific",
                "turnover": 0.2,
                "transaction_cost": 0.00002,
                "weight_SPY": 0.6,
                "weight_TLT": 0.4,
                "weight_GLD": 0.0,
                "weight_BTC-USD": 0.0,
                "weight_CASH": 0.0,
            },
            {
                "date": "2024-01-12",
                "transaction_cost_mode": "asset_specific",
                "turnover": 0.1,
                "transaction_cost": 0.00001,
                "weight_SPY": 0.4,
                "weight_TLT": 0.6,
                "weight_GLD": 0.0,
                "weight_BTC-USD": 0.0,
                "weight_CASH": 0.0,
            },
        ]
    ).to_csv(path, index=False)


def _write_statistical_outputs(stat_dir: Path, wrc_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "candidate": "V5_no_volatility_block_cap_0p50",
                "benchmark": "trend_spy_cash_12p",
                "metric": "sharpe",
                "probability_candidate_beats": 0.60,
            }
        ]
    ).to_csv(stat_dir / "statistical_validation_pairwise_bootstrap.csv", index=False)
    pd.DataFrame(
        [
            {
                "benchmark": "trend_spy_cash_12p",
                "p_value": 0.52,
                "best_candidate_by_mean_diff": "V5_no_volatility_block_cap_0p50",
            }
        ]
    ).to_csv(wrc_dir / "white_reality_check_summary.csv", index=False)


if __name__ == "__main__":
    unittest.main()
