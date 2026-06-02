"""Tests for transaction cost sensitivity reporting."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.transaction_cost_sensitivity_report import (
    KEY_V3_CLEAN,
    CostScenario,
    apply_cost_scenario,
    build_transaction_cost_sensitivity_report,
    build_winners_table,
    estimate_cost_drag,
)


class TransactionCostSensitivityReportTests(unittest.TestCase):
    def test_cost_adjustment_reduces_returns_when_turnover_positive(self):
        frame = pd.DataFrame(
            {
                "portfolio_return": [0.02, 0.02],
                "financial_net_return": [0.019, 0.019],
                "turnover": [1.0, 1.0],
            }
        )
        scenario = CostScenario("stress", etf_stock_bps=5.0, btc_bps=30.0, blended_bps=10.0)

        adjusted = apply_cost_scenario(frame, scenario)

        self.assertLess(adjusted["returns"].iloc[0], frame["portfolio_return"].iloc[0])
        self.assertGreater(adjusted["cost_drag"].iloc[0], 0.0)

    def test_zero_turnover_creates_zero_additional_drag(self):
        frame = pd.DataFrame(
            {
                "portfolio_return": [0.02, 0.02],
                "financial_net_return": [0.02, 0.02],
                "turnover": [0.0, 0.0],
            }
        )
        scenario = CostScenario("ibkr_proxy", etf_stock_bps=2.0, btc_bps=10.0, blended_bps=5.0)

        adjusted = apply_cost_scenario(frame, scenario)

        self.assertAlmostEqual(float(adjusted["cost_drag"].sum()), 0.0)

    def test_btc_bps_creates_larger_drag_than_etf_when_asset_weights_available(self):
        etf_frame = pd.DataFrame(
            {
                "portfolio_return": [0.0, 0.0],
                "financial_net_return": [0.0, 0.0],
                "turnover": [1.0, 1.0],
                "weight_SPY": [1.0, 0.0],
                "weight_BTC-USD": [0.0, 0.0],
                "weight_CASH": [0.0, 1.0],
            }
        )
        btc_frame = pd.DataFrame(
            {
                "portfolio_return": [0.0, 0.0],
                "financial_net_return": [0.0, 0.0],
                "turnover": [1.0, 1.0],
                "weight_SPY": [0.0, 0.0],
                "weight_BTC-USD": [1.0, 0.0],
                "weight_CASH": [0.0, 1.0],
            }
        )
        scenario = CostScenario("ibkr_proxy", etf_stock_bps=2.0, btc_bps=10.0, blended_bps=5.0)

        etf_drag, *_ = estimate_cost_drag(etf_frame, scenario)
        btc_drag, *_ = estimate_cost_drag(btc_frame, scenario)

        self.assertGreater(float(btc_drag.sum()), float(etf_drag.sum()))

    def test_blended_proxy_used_when_asset_weights_missing(self):
        frame = pd.DataFrame(
            {
                "portfolio_return": [0.01],
                "financial_net_return": [0.01],
                "turnover": [2.0],
            }
        )
        scenario = CostScenario("ibkr_proxy", etf_stock_bps=2.0, btc_bps=10.0, blended_bps=5.0)

        drag, method, asset_level_available, blended_used = estimate_cost_drag(frame, scenario)

        self.assertEqual(method, "blended_total_turnover_proxy")
        self.assertFalse(asset_level_available)
        self.assertTrue(blended_used)
        self.assertAlmostEqual(float(drag.iloc[0]), 2.0 * 0.0005)

    def test_existing_scenario_leaves_returns_unchanged(self):
        frame = pd.DataFrame(
            {
                "portfolio_return": [0.02, 0.02],
                "financial_net_return": [0.019, 0.018],
                "turnover": [1.0, 1.0],
            }
        )
        scenario = CostScenario("existing", 0.0, 0.0, 0.0, use_existing_net_returns=True)

        adjusted = apply_cost_scenario(frame, scenario)

        self.assertEqual(list(adjusted["returns"]), [0.019, 0.018])
        self.assertAlmostEqual(float(adjusted["cost_drag"].sum()), 0.0)

    def test_output_files_are_created_and_metadata_records_assumptions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final_dir, cap_dir, bench_dir = self._write_minimal_report(root)
            out_dir = root / "out"

            result = build_transaction_cost_sensitivity_report(
                final_report_dir=str(final_dir),
                output_dir=str(out_dir),
                v3_clean_no_dxy_cap_sensitivity_dir=str(cap_dir),
            )

            self.assertTrue((out_dir / "transaction_cost_sensitivity_summary.csv").exists())
            self.assertTrue((out_dir / "transaction_cost_sensitivity_winners.csv").exists())
            self.assertTrue((out_dir / "transaction_cost_sensitivity_metadata.json").exists())
            self.assertTrue((out_dir / "transaction_cost_sensitivity_summary.md").exists())
            metadata = json.loads((out_dir / "transaction_cost_sensitivity_metadata.json").read_text())
            self.assertIn("cost_assumptions", metadata)
            self.assertIn("IBKR", " ".join(metadata["source_notes"]))
            self.assertTrue(metadata["caveats"])
            self.assertFalse(result["summary"].empty)

    def test_winner_ranking_logic_works_on_synthetic_data(self):
        summary = pd.DataFrame(
            [
                {
                    "scenario": "existing",
                    "strategy": KEY_V3_CLEAN,
                    "strategy_type": "td3",
                    "mandate_score_or_available_score": 0.7,
                    "rank_within_scenario": 1,
                },
                {
                    "scenario": "existing",
                    "strategy": "BuyHold_GLD",
                    "strategy_type": "benchmark",
                    "mandate_score_or_available_score": 0.5,
                    "rank_within_scenario": 2,
                },
            ]
        )

        winners = build_winners_table(summary)

        self.assertEqual(winners.iloc[0]["best_overall"], KEY_V3_CLEAN)
        self.assertEqual(winners.iloc[0]["best_td3"], KEY_V3_CLEAN)
        self.assertEqual(winners.iloc[0]["best_benchmark"], "BuyHold_GLD")

    def _write_minimal_report(self, root: Path) -> tuple[Path, Path, Path]:
        final_dir = root / "final"
        cap_dir = root / "cap"
        bench_dir = root / "bench"
        final_dir.mkdir()
        (bench_dir / "benchmarks" / "histories").mkdir(parents=True)

        self._write_td3_history(cap_dir, value=0.02)
        self._write_history(
            bench_dir / "benchmarks" / "histories" / "BuyHold_GLD_history.csv",
            value=0.005,
        )
        pd.DataFrame(
            [
                {
                    "strategy_name": KEY_V3_CLEAN,
                    "base_candidate": "V3_real_macro_vintage_clean_no_dxy",
                    "source": "v3_clean_no_dxy_cap_sensitivity",
                    "selected_cap": 0.50,
                }
            ]
        ).to_csv(final_dir / "final_constrained_td3_selected_candidates.csv", index=False)
        pd.DataFrame(
            [
                {
                    "strategy_name": KEY_V3_CLEAN,
                    "strategy_type": "td3",
                    "robust_score": 0.7,
                    "average_effective_number_of_assets": 3.0,
                    "average_max_weight": 0.5,
                },
                {
                    "strategy_name": "BuyHold_GLD",
                    "strategy_type": "benchmark",
                    "robust_score": 0.6,
                    "average_effective_number_of_assets": 1.0,
                    "average_max_weight": 1.0,
                },
            ]
        ).to_csv(final_dir / "final_constrained_td3_main_ranking.csv", index=False)
        pd.DataFrame(
            [{"strategy_name": "BuyHold_GLD", "strategy_type": "benchmark"}]
        ).to_csv(final_dir / "final_constrained_td3_mandate_ranking.csv", index=False)
        (final_dir / "final_constrained_td3_metadata.json").write_text(
            json.dumps({"benchmark_comparison_dir": str(bench_dir)}),
            encoding="utf-8",
        )
        return final_dir, cap_dir, bench_dir

    def _write_td3_history(self, cap_dir: Path, value: float) -> None:
        base = "V3_real_macro_vintage_clean_no_dxy"
        history_dir = cap_dir / "per_candidate" / base / f"F1_{base}_cap_0p50_seed_7"
        history_dir.mkdir(parents=True)
        self._write_history(history_dir / "test_policy_history.csv", value=value)

    @staticmethod
    def _write_history(path: Path, value: float) -> None:
        dates = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
        weights = [0.5, 0.5] * 20
        pd.DataFrame(
            {
                "date": dates,
                "portfolio_return": [value] * len(dates),
                "financial_net_return": [value - 0.001] * len(dates),
                "turnover": [0.5] * len(dates),
                "max_weight": weights,
                "weight_SPY": weights,
                "weight_BTC-USD": [0.0] * len(dates),
                "weight_CASH": [0.5] * len(dates),
            }
        ).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
