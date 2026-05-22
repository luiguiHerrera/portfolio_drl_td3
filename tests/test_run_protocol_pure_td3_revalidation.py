"""Tests for the protocol-pure TD3 revalidation runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.experiments.run_protocol_pure_td3_revalidation import (
    DEFAULT_V3_MACRO_PATH,
    PROTOCOL_CANDIDATES,
    _build_candidate_run_config,
    _build_v3_features,
    _candidate_raw_features,
    _select_candidates,
    _validate_protocol_reward_semantics,
    run_protocol_pure_td3_revalidation,
)


class ProtocolPureTD3RevalidationTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2021-01-01", periods=12, freq="W-FRI")
        self.returns = pd.DataFrame(
            {
                "SPY": [0.01, -0.01, 0.02, 0.00] * 3,
                "TLT": [0.00, 0.01, -0.01, 0.01] * 3,
                "GLD": [0.005, 0.002, -0.003, 0.004] * 3,
                "BTC-USD": [0.03, -0.02, 0.01, -0.01] * 3,
                "CASH": [0.0] * 12,
            },
            index=self.index,
        )
        self.features = pd.DataFrame(
            {"feature": range(len(self.index))},
            index=self.index,
        )
        self.fold = {
            "fold_id": "F1",
            "description": "smoke",
            "train_start": "2021-01-08",
            "train_end": "2021-02-12",
            "validation_start": "2021-02-19",
            "validation_end": "2021-03-05",
            "test_start": "2021-03-12",
            "test_end": "2021-03-19",
        }

    def test_smoke_runner_writes_protocol_outputs_and_limits_training_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patched_dependencies() as train_mock:
                result = run_protocol_pure_td3_revalidation(
                    output_dir=str(Path(temp_dir) / "out"),
                    candidates=["V2_reference_full"],
                    folds=[self.fold, {**self.fold, "fold_id": "F2"}],
                    seeds=[7, 21],
                    episodes=3,
                    smoke=True,
                )

            output = Path(result["output_dir"])

            self.assertEqual(train_mock.call_count, 1)
            self.assertTrue((output / "overall_aggregate_by_strategy_split.csv").exists())
            self.assertTrue((output / "seed_level_aggregate_by_strategy_split.csv").exists())
            self.assertTrue((output / "fold_level_aggregate_by_strategy_split.csv").exists())
            self.assertTrue((output / "seed_fold_strategy_results.csv").exists())
            self.assertTrue((output / "robust_score_ranking.csv").exists())
            self.assertTrue(
                (output / "protocol_pure_td3_revalidation_metadata.json").exists()
            )
            self.assertEqual(result["metadata"]["episodes"], 1)
            self.assertEqual(result["metadata"]["seeds"], [7])
            self.assertFalse(result["metadata"]["lambda_sharpe_present_or_active"])
            self.assertEqual(
                result["metadata"]["robust_score_training_usage"],
                "evaluation_only",
            )

    def test_v6_candidate_config_uses_cash_permission_auxiliary_column(self):
        base_config = self._base_config()
        candidate = _candidate("V6_financial_state")

        config = _build_candidate_run_config(
            base_config=base_config,
            candidate=candidate,
            seed=7,
            episodes=30,
            batch_size=32,
            actor_learning_rate=0.0005,
            critic_learning_rate=0.0005,
        )

        self.assertEqual(config["features"]["version"], "v6")
        self.assertTrue(config["reward"]["use_cash_risk_off_penalty"])
        self.assertEqual(config["reward"]["cash_risk_off_column"], "cash_permission_score")
        self.assertNotIn("lambda_sharpe", config["reward"])

    def test_v2_candidate_config_keeps_reference_turnover_mode(self):
        base_config = self._base_config()
        candidate = _candidate("V2_reference_full")

        config = _build_candidate_run_config(
            base_config=base_config,
            candidate=candidate,
            seed=21,
            episodes=30,
            batch_size=32,
            actor_learning_rate=0.0005,
            critic_learning_rate=0.0005,
        )

        self.assertEqual(config["features"]["version"], "v2")
        self.assertFalse(config["reward"]["use_cash_risk_off_penalty"])
        self.assertEqual(config["reward"]["turnover_penalty_mode"], "linear")

    def test_lambda_sharpe_is_rejected(self):
        config = self._base_config()
        config["reward"]["lambda_sharpe"] = 0.1

        with self.assertRaisesRegex(ValueError, "lambda_sharpe"):
            _validate_protocol_reward_semantics(config)

    def test_unknown_candidate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown candidates"):
            _select_candidates(["not_a_candidate"])

    def test_v3_candidate_is_available_but_not_default_enabled(self):
        default_names = {candidate["name"] for candidate in _select_candidates(None)}
        explicit_names = {
            candidate["name"]
            for candidate in _select_candidates(["V3_real_macro_current"])
        }

        self.assertNotIn("V3_real_macro_current", default_names)
        self.assertIn("V2_reference_full", default_names)
        self.assertIn("V5_no_volatility_block", default_names)
        self.assertIn("V6_financial_state", default_names)
        self.assertEqual(explicit_names, {"V3_real_macro_current"})

    def test_v3_candidate_config_uses_required_macro_path(self):
        base_config = self._base_config()
        candidate = _candidate("V3_real_macro_current")

        config = _build_candidate_run_config(
            base_config=base_config,
            candidate=candidate,
            seed=7,
            episodes=5,
            batch_size=32,
            actor_learning_rate=0.0005,
            critic_learning_rate=0.0005,
        )

        self.assertEqual(config["features"]["version"], "v3")
        self.assertEqual(config["features"]["macro_path"], DEFAULT_V3_MACRO_PATH)
        self.assertFalse(config["reward"]["use_cash_risk_off_penalty"])

    def test_v3_raw_feature_generation_includes_macro_columns(self):
        returns = self._long_returns()
        macro_path = self._write_macro_csv_for_returns(returns)
        candidate = {
            **_candidate("V3_real_macro_current"),
            "macro_path": macro_path,
        }

        features = _build_v3_features(
            returns=returns,
            base_config=self._base_config(),
            candidate=candidate,
        )

        self.assertTrue(any(column.startswith("macro_") for column in features.columns))
        self.assertIs(_candidate_raw_features(candidate, {"v3_features": features}), features)

    def test_v3_missing_macro_path_raises_error(self):
        candidate = {
            **_candidate("V3_real_macro_current"),
            "macro_path": "does/not/exist.csv",
        }

        with self.assertRaises(FileNotFoundError):
            _build_v3_features(
                returns=self._long_returns(),
                base_config=self._base_config(),
                candidate=candidate,
            )

    def test_v3_stale_macro_coverage_raises_error(self):
        returns = self._long_returns()
        macro_path = self._write_macro_csv_for_returns(returns.iloc[:-10])
        candidate = {
            **_candidate("V3_real_macro_current"),
            "macro_path": macro_path,
        }

        with self.assertRaisesRegex(ValueError, "stale"):
            _build_v3_features(
                returns=returns,
                base_config=self._base_config(),
                candidate=candidate,
            )

    def _patched_dependencies(self):
        patches = [
            patch(
                "src.experiments.run_protocol_pure_td3_revalidation._build_base_config",
                return_value=self._base_config(),
            ),
            patch(
                "src.experiments.run_protocol_pure_td3_revalidation."
                "build_returns_dataset_from_config",
                return_value=self.returns,
            ),
            patch(
                "src.experiments.run_protocol_pure_td3_revalidation._build_feature_context",
                return_value={"v2_features": self.features},
            ),
            patch(
                "src.experiments.run_protocol_pure_td3_revalidation."
                "train_td3_ablation_on_datasets",
                side_effect=self._mock_train_result,
            ),
        ]

        class _Context:
            def __enter__(_self):
                entered = [p.start() for p in patches]
                _self.train_mock = entered[-1]
                return _self.train_mock

            def __exit__(_self, exc_type, exc_value, traceback):
                for p in reversed(patches):
                    p.stop()

        return _Context()

    def _mock_train_result(self, datasets, config):
        return {
            "episode_logs": [
                {
                    "episode": 1,
                    "final_portfolio_value": 100100.0,
                    "total_reward": 0.01,
                    "average_turnover": 0.2,
                    "average_transaction_cost": 0.0002,
                    "max_weight": 0.5,
                    "cash_weight": 0.1,
                }
            ],
            "validation_comparison": {"metrics_table": self._metrics_table()},
            "test_comparison": {"metrics_table": self._metrics_table()},
            "validation_evaluation": {
                "diagnostics": self._diagnostics(),
                "policy_history": self._policy_history(datasets["validation_returns"].index),
            },
            "test_evaluation": {
                "diagnostics": self._diagnostics(),
                "policy_history": self._policy_history(datasets["test_returns"].index),
            },
        }

    def _metrics_table(self):
        rows = [
            "agent",
            "equal_weight_gross",
            "equal_weight_rebalanced_net",
            "buy_and_hold",
            "buy_hold_SPY",
        ]
        return pd.DataFrame(
            {
                "cumulative_return": [0.02, 0.01, 0.01, 0.015, 0.012],
                "annualized_return": [0.10, 0.05, 0.05, 0.07, 0.06],
                "annualized_volatility": [0.15, 0.12, 0.12, 0.14, 0.13],
                "sharpe_ratio": [0.8, 0.4, 0.4, 0.5, 0.45],
                "sortino_ratio": [1.0, 0.5, 0.5, 0.6, 0.55],
                "calmar_ratio": [0.7, 0.3, 0.3, 0.4, 0.35],
                "max_drawdown": [-0.1, -0.2, -0.2, -0.15, -0.18],
            },
            index=rows,
        )

    def _diagnostics(self):
        return {
            "average_turnover": 0.2,
            "average_effective_number_of_assets": 1.5,
            "average_max_weight": 0.7,
            "final_max_weight": 0.7,
            "final_cash_weight": 0.1,
            "average_transaction_cost": 0.0002,
            "final_weights": {
                "SPY": 0.4,
                "TLT": 0.2,
                "GLD": 0.2,
                "BTC-USD": 0.1,
                "CASH": 0.1,
            },
        }

    def _policy_history(self, index):
        return pd.DataFrame(
            {
                "date": index,
                "portfolio_return": [0.01] * len(index),
                "financial_net_return": [0.009] * len(index),
                "portfolio_value": [100000 * (1.009 ** (i + 1)) for i in range(len(index))],
                "drawdown": [0.0] * len(index),
                "turnover": [0.2] * len(index),
                "transaction_cost": [0.0002] * len(index),
                "max_weight": [0.7] * len(index),
                "cash_weight": [0.1] * len(index),
                "weight_SPY": [0.4] * len(index),
                "weight_TLT": [0.2] * len(index),
                "weight_GLD": [0.2] * len(index),
                "weight_BTC-USD": [0.1] * len(index),
                "weight_CASH": [0.1] * len(index),
            }
        )

    def _base_config(self):
        return {
            "project": {"name": "test"},
            "data": {
                "assets": ["SPY", "TLT", "GLD", "BTC-USD", "CASH"],
                "frequency": "weekly",
                "returns_path": "returns.csv",
                "returns_date_column": "date",
            },
            "environment": {"initial_cash": 100000, "transaction_cost": 0.001},
            "reward": {
                "lambda_return": 1.0,
                "lambda_drawdown": 0.02,
                "lambda_transaction_cost": 1.0,
                "lambda_turnover": 0.0005,
                "lambda_concentration": 0.0,
            },
            "td3": {
                "actor_learning_rate": 0.0005,
                "critic_learning_rate": 0.0005,
                "batch_size": 32,
            },
            "training": {"seed": 42, "episodes": 30},
            "features": {"version": "v2"},
        }

    @staticmethod
    def _long_returns() -> pd.DataFrame:
        index = pd.date_range("2021-01-01", periods=80, freq="W-FRI")
        pattern = [0.01, -0.01, 0.02, 0.00, 0.015, -0.005, 0.008, -0.004]
        return pd.DataFrame(
            {
                "SPY": (pattern * 10)[: len(index)],
                "TLT": ([0.00, 0.01, -0.01, 0.01] * 20)[: len(index)],
                "GLD": ([0.005, 0.002, -0.003, 0.004] * 20)[: len(index)],
                "BTC-USD": ([0.03, -0.02, 0.01, -0.01] * 20)[: len(index)],
                "CASH": [0.0] * len(index),
            },
            index=index,
        )

    def _write_macro_csv_for_returns(self, returns: pd.DataFrame) -> str:
        temp_dir = tempfile.mkdtemp()
        path = Path(temp_dir) / "macro.csv"
        index = pd.date_range(
            returns.index.min() - pd.offsets.Week(weekday=4),
            returns.index.max(),
            freq="W-FRI",
        )
        macro = pd.DataFrame(
            {
                "date": index,
                "DGS10": [4.0 + i * 0.001 for i in range(len(index))],
                "DGS2": [3.0 + i * 0.001 for i in range(len(index))],
                "VIX": [20.0 + (i % 5) for i in range(len(index))],
                "DXY": [100.0 + i * 0.01 for i in range(len(index))],
                "CPI": [300.0 + i * 0.02 for i in range(len(index))],
            }
        )
        macro.to_csv(path, index=False)
        return str(path)


def _candidate(name):
    return next(candidate for candidate in PROTOCOL_CANDIDATES if candidate["name"] == name)


if __name__ == "__main__":
    unittest.main()
