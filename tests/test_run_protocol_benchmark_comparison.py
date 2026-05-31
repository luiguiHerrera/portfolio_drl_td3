"""Tests for the protocol benchmark-only comparison runner."""

import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd

from src.experiments.run_protocol_benchmark_comparison import (
    build_protocol_benchmark_weights,
    load_protocol_returns,
    parse_asset_transaction_cost_bps,
    resolve_transaction_cost_settings,
    run_protocol_benchmark_comparison,
)


class ProtocolBenchmarkComparisonRunnerTests(unittest.TestCase):
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

    def test_runner_creates_expected_output_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            output_dir = Path(temp_dir) / "outputs"

            result = run_protocol_benchmark_comparison(
                returns_path=str(returns_path),
                output_dir=str(output_dir),
                transaction_cost=0.01,
            )

            self.assertTrue(Path(result["paths"]["metrics_table"]).exists())
            self.assertTrue(Path(result["paths"]["comparison_summary"]).exists())
            self.assertTrue(Path(result["paths"]["diagnostics"]).exists())
            self.assertTrue(Path(result["paths"]["histories_dir"]).exists())
            self.assertTrue(result["paths"]["histories"])

    def test_static_benchmarks_are_net_cost_comparable_weight_strategies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            result = run_protocol_benchmark_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
                transaction_cost=0.01,
            )

            buy_hold_spy = result["evaluations"]["BuyHold_SPY"]["history"]

            self.assertAlmostEqual(buy_hold_spy.loc[0, "turnover"], 1.6)
            self.assertAlmostEqual(buy_hold_spy.loc[0, "transaction_cost"], 0.016)
            self.assertAlmostEqual(
                buy_hold_spy.loc[0, "financial_net_return"],
                buy_hold_spy.loc[0, "portfolio_return"]
                - buy_hold_spy.loc[0, "transaction_cost"],
            )

    def test_benchmark_histories_include_required_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            result = run_protocol_benchmark_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
            )
            history = result["evaluations"]["momentum_winner_12p"]["history"]

            for column in [
                "portfolio_return",
                "financial_net_return",
                "transaction_cost",
                "turnover",
                "portfolio_value",
                "drawdown",
                "weight_SPY",
                "weight_CASH",
            ]:
                self.assertIn(column, history.columns)

    def test_sixty_forty_weights_are_correct(self):
        returns = load_protocol_returns(str(self._write_returns_in_new_tempdir()))

        weights = build_protocol_benchmark_weights(returns)["60_40_SPY_TLT"]

        self.assertTrue((weights["SPY"] == 0.60).all())
        self.assertTrue((weights["TLT"] == 0.40).all())
        self.assertTrue((weights[["GLD", "BTC-USD", "CASH"]] == 0.0).all().all())

    def test_equal_weight_risky_excludes_cash(self):
        returns = load_protocol_returns(str(self._write_returns_in_new_tempdir()))

        weights = build_protocol_benchmark_weights(returns)["Equal_Weight_Risky"]

        self.assertTrue((weights["CASH"] == 0.0).all())
        self.assertTrue((weights[["SPY", "TLT", "GLD", "BTC-USD"]] == 0.25).all().all())

    def test_rolling_benchmarks_appear_in_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            result = run_protocol_benchmark_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
            )

            benchmark_names = set(result["metrics_table"]["benchmark_name"])

        self.assertIn("rolling_risk_parity_inverse_vol_12p", benchmark_names)
        self.assertIn("rolling_markowitz_long_only_52p", benchmark_names)
        self.assertIn("rolling_markowitz_min_variance_52p", benchmark_names)

    def test_asset_specific_benchmark_histories_include_cost_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            cost_bps = {
                "SPY": 2.0,
                "TLT": 2.0,
                "GLD": 2.0,
                "BTC-USD": 10.0,
                "CASH": 0.0,
            }

            result = run_protocol_benchmark_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
                transaction_cost_mode="asset_specific",
                asset_transaction_cost_bps=cost_bps,
            )
            history = result["evaluations"]["BuyHold_BTC-USD"]["history"]

            for column in [
                "transaction_cost_mode",
                "asset_turnover_SPY",
                "asset_turnover_TLT",
                "asset_turnover_GLD",
                "asset_turnover_BTC-USD",
                "asset_turnover_CASH",
                "asset_transaction_cost_contribution_SPY",
                "asset_transaction_cost_contribution_TLT",
                "asset_transaction_cost_contribution_GLD",
                "asset_transaction_cost_contribution_BTC-USD",
                "asset_transaction_cost_contribution_CASH",
            ]:
                self.assertIn(column, history.columns)
            self.assertEqual(history["transaction_cost_mode"].iloc[0], "asset_specific")
            self.assertGreater(
                history["asset_transaction_cost_contribution_BTC-USD"].sum(),
                history["asset_transaction_cost_contribution_SPY"].sum(),
            )

            metadata = json.loads(Path(result["paths"]["metadata"]).read_text())
            self.assertEqual(metadata["transaction_cost_mode"], "asset_specific")
            self.assertEqual(metadata["asset_transaction_cost_bps"], cost_bps)
            self.assertIn("Broker/exchange-style", metadata["cost_caveat"])

    def test_base_config_passes_asset_specific_settings_to_benchmark_evaluation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            config_path = self._write_asset_specific_config(temp_dir)

            result = run_protocol_benchmark_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
                **resolve_transaction_cost_settings(
                    base_config_path=str(config_path),
                    transaction_cost=None,
                    transaction_cost_mode=None,
                    asset_transaction_cost_bps=None,
                ),
                base_config_path=str(config_path),
            )
            history = result["evaluations"]["BuyHold_GLD"]["history"]
            metadata = json.loads(Path(result["paths"]["metadata"]).read_text())

            self.assertEqual(history["transaction_cost_mode"].iloc[0], "asset_specific")
            self.assertEqual(metadata["transaction_cost_mode"], "asset_specific")
            self.assertEqual(metadata["asset_transaction_cost_bps"]["BTC-USD"], 10.0)

    def test_parse_asset_transaction_cost_bps(self):
        parsed = parse_asset_transaction_cost_bps(
            "SPY:2.0,TLT:2.0,GLD:2.0,BTC-USD:10.0,CASH:0.0",
        )

        self.assertEqual(
            parsed,
            {"SPY": 2.0, "TLT": 2.0, "GLD": 2.0, "BTC-USD": 10.0, "CASH": 0.0},
        )

    def test_metrics_table_contains_protocol_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            returns_path = self._write_returns(temp_dir)
            result = run_protocol_benchmark_comparison(
                returns_path=str(returns_path),
                output_dir=str(Path(temp_dir) / "outputs"),
            )
            metrics = result["metrics_table"]

        for column in [
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "average_turnover",
            "total_transaction_cost",
            "average_max_weight",
            "average_effective_number_of_assets",
            "mean_cash_weight",
            "cash_above_10pct",
        ]:
            self.assertIn(column, metrics.columns)

    def _write_returns(self, temp_dir: str) -> Path:
        path = Path(temp_dir) / "returns.csv"
        self.returns.to_csv(path, index=False)
        return path

    def _write_returns_in_new_tempdir(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return self._write_returns(temp_dir.name)

    def _write_asset_specific_config(self, temp_dir: str) -> Path:
        path = Path(temp_dir) / "asset_specific_config.yaml"
        path.write_text(
            """
project:
  name: benchmark_asset_specific_test
data:
  assets: [SPY, TLT, GLD, BTC-USD, CASH]
  frequency: weekly
environment:
  initial_cash: 100000
  transaction_cost: 0.001
  transaction_cost_mode: asset_specific
  asset_transaction_cost_bps:
    SPY: 2.0
    TLT: 2.0
    GLD: 2.0
    BTC-USD: 10.0
    CASH: 0.0
reward: {}
td3:
  actor_learning_rate: 0.001
  critic_learning_rate: 0.001
  gamma: 0.99
  tau: 0.005
  policy_noise: 0.2
  noise_clip: 0.5
  policy_delay: 2
  batch_size: 16
  replay_buffer_size: 100
training:
  seed: 7
  train_ratio: 0.6
  validation_ratio: 0.2
  test_ratio: 0.2
  episodes: 1
""".lstrip(),
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
