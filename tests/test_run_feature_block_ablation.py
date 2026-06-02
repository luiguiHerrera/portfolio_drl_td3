"""Tests for feature-block ablation diagnostics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features_v2 import build_features_v2
from src.data.features_v5 import build_features_v5, build_v5_regime_auxiliary_features
from src.env.portfolio_env import PortfolioEnv
from src.memory.replay_buffer import ReplayBuffer
from src.experiments.run_feature_block_ablation import (
    FEATURE_VARIANTS,
    build_ablation_fold_datasets,
    build_cash_attribution_summary,
    build_feature_block_map,
    select_feature_columns,
    train_td3_ablation_on_datasets,
)
from src.utils.action_projection import project_portfolio_action


class FeatureBlockAblationTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2021-01-01", periods=40, freq="W-FRI")
        values = np.linspace(-0.02, 0.03, len(index))
        self.returns = pd.DataFrame(
            {
                "SPY": values,
                "TLT": values[::-1] / 2.0,
                "GLD": np.sin(np.arange(len(index))) / 100.0,
                "BTC-USD": np.cos(np.arange(len(index))) / 50.0,
                "CASH": 0.0,
            },
            index=index,
        )
        self.block_map = build_feature_block_map(list(self.returns.columns))

    def test_block_map_uses_explicit_known_columns(self):
        self.assertIn("SPY_mom_4p", self.block_map["momentum"])
        self.assertIn("SPY_vol_12p", self.block_map["volatility"])
        self.assertIn("SPY_rolling_drawdown_12p", self.block_map["drawdown"])
        self.assertIn("TLT_corr_vs_SPY_12p", self.block_map["correlation"])
        self.assertIn("risk_off_state", self.block_map["regime"])

    def test_no_momentum_variant_removes_momentum_columns(self):
        features = build_features_v5(self.returns)
        variant = _variant("V5_no_momentum_block")
        selected = select_feature_columns(features.columns, variant, self.block_map)
        self.assertNotIn("SPY_mom_4p", selected)
        self.assertNotIn("regime_market_momentum_12p", selected)
        self.assertIn("SPY_vol_12p", selected)

    def test_momentum_minimal_variant_keeps_only_base_and_momentum(self):
        features = build_features_v5(self.returns)
        variant = _variant("V5_momentum_only_or_minimal_momentum_regime")
        selected = select_feature_columns(features.columns, variant, self.block_map)
        self.assertIn("SPY_ret_1p", selected)
        self.assertIn("SPY_mom_12p", selected)
        self.assertNotIn("SPY_vol_12p", selected)
        self.assertNotIn("avg_pairwise_corr_12p", selected)

    def test_v2_reference_columns_are_selected_from_v2_only(self):
        features = build_features_v2(self.returns)
        variant = _variant("V2_reference_full")
        selected = select_feature_columns(features.columns, variant, self.block_map)
        self.assertEqual(selected, list(features.columns))
        self.assertNotIn("risk_off_state", selected)

    def test_build_ablation_fold_datasets_adds_auxiliary_only_when_requested(self):
        features = build_features_v5(self.returns)
        auxiliary = build_v5_regime_auxiliary_features(self.returns).shift(1).dropna()
        fold = {
            "fold_id": "F1",
            "description": "test",
            "train_start": "2021-04-02",
            "train_end": "2021-06-25",
            "validation_start": "2021-07-02",
            "validation_end": "2021-08-27",
            "test_start": "2021-09-03",
            "test_end": "2021-10-01",
        }
        datasets = build_ablation_fold_datasets(
            returns=self.returns,
            raw_features=features,
            raw_auxiliary_features=auxiliary,
            fold=fold,
            include_auxiliary=True,
        )
        self.assertIn("train_auxiliary_features", datasets)
        self.assertEqual(
            list(datasets["train_auxiliary_features"].columns),
            [
                "regime_market_drawdown_stress",
                "regime_market_high_vol",
                "correlation_stress",
                "risk_off_score",
                "risk_off_state",
            ],
        )
        self.assertTrue(
            datasets["train_returns"].index.equals(
                datasets["train_auxiliary_features"].index,
            )
        )

    def test_cash_attribution_summary_uses_risk_off_state(self):
        auxiliary = build_v5_regime_auxiliary_features(self.returns).shift(1).dropna()
        dates = auxiliary.index[:3]
        history = pd.DataFrame(
            {
                "date": dates,
                "cash_weight": [0.20, 0.05, 0.30],
                "weight_SPY": [0.8, 0.95, 0.7],
                "weight_CASH": [0.2, 0.05, 0.3],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test_policy_history.csv"
            history.to_csv(path, index=False)
            summary = build_cash_attribution_summary(
                [str(path)],
                ["strategy"],
                auxiliary,
            )
        self.assertIn("mean_unjustified_cash_excess", summary["aggregate"].columns)
        self.assertEqual(summary["aggregate"].loc[0, "strategy"], "strategy")

    def test_ablation_replay_buffer_receives_training_seed(self):
        datasets = {
            "train_returns": self.returns.iloc[:3],
            "validation_returns": self.returns.iloc[3:5],
            "test_returns": self.returns.iloc[5:7],
            "train_features": pd.DataFrame({"feature": [0.0, 1.0, 2.0]}, index=self.returns.index[:3]),
            "validation_features": pd.DataFrame({"feature": [3.0, 4.0]}, index=self.returns.index[3:5]),
            "test_features": pd.DataFrame({"feature": [5.0, 6.0]}, index=self.returns.index[5:7]),
        }
        config = {
            "environment": {"initial_cash": 100000, "transaction_cost": 0.001},
            "reward": {"lambda_return": 1.0, "lambda_transaction_cost": 1.0},
            "training": {"seed": 123, "episodes": 1},
            "td3": {
                "actor_learning_rate": 0.0003,
                "critic_learning_rate": 0.0003,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "policy_delay": 2,
                "batch_size": 10,
                "replay_buffer_size": 50,
            },
        }

        with unittest.mock.patch(
            "src.experiments.run_feature_block_ablation.ReplayBuffer",
            side_effect=ReplayBuffer,
        ) as replay_buffer:
            train_td3_ablation_on_datasets(datasets, config)

        self.assertEqual(replay_buffer.call_args.kwargs["seed"], 123)

    def test_ablation_replay_buffer_stores_env_executed_action(self):
        deterministic_action = np.array([0.80, 0.05, 0.05, 0.05, 0.05])
        datasets = {
            "train_returns": self.returns.iloc[:3],
            "validation_returns": self.returns.iloc[3:5],
            "test_returns": self.returns.iloc[5:7],
            "train_features": pd.DataFrame({"feature": [0.0, 1.0, 2.0]}, index=self.returns.index[:3]),
            "validation_features": pd.DataFrame({"feature": [3.0, 4.0]}, index=self.returns.index[3:5]),
            "test_features": pd.DataFrame({"feature": [5.0, 6.0]}, index=self.returns.index[5:7]),
        }
        config = {
            "environment": {"initial_cash": 100000, "transaction_cost": 0.001},
            "reward": {"lambda_return": 1.0, "lambda_transaction_cost": 1.0},
            "training": {"seed": 123, "episodes": 1},
            "td3": {
                "actor_learning_rate": 0.0003,
                "critic_learning_rate": 0.0003,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "policy_delay": 2,
                "batch_size": 10,
                "replay_buffer_size": 50,
            },
        }

        class FakeAgent:
            def __init__(self, *args, **kwargs):
                self.action = deterministic_action

            def select_action(self, state):
                return self.action.copy()

            def train_step(self, batch):
                return {
                    "critic_1_loss": 0.0,
                    "critic_2_loss": 0.0,
                    "actor_loss": 0.0,
                }

        class InternallyCappedEnv(PortfolioEnv):
            executed_actions = []

            def _normalize_action(self, action):
                return project_portfolio_action(action, max_weight=0.40)

            def step(self, action):
                result = super().step(action)
                self.__class__.executed_actions.append(result[3]["executed_action"].copy())
                return result

        class RecordingReplayBuffer(ReplayBuffer):
            added_actions = []

            def add(self, state, action, reward, next_state, done):
                self.__class__.added_actions.append(np.asarray(action, dtype=float).copy())
                super().add(state, action, reward, next_state, done)

        with (
            unittest.mock.patch(
                "src.experiments.run_feature_block_ablation.TD3Agent",
                side_effect=FakeAgent,
            ),
            unittest.mock.patch(
                "src.experiments.run_feature_block_ablation.PortfolioEnv",
                InternallyCappedEnv,
            ),
            unittest.mock.patch(
                "src.experiments.run_feature_block_ablation.ReplayBuffer",
                side_effect=RecordingReplayBuffer,
            ),
        ):
            train_td3_ablation_on_datasets(datasets, config)

        first_replay_action = RecordingReplayBuffer.added_actions[0]
        first_executed_action = InternallyCappedEnv.executed_actions[0]
        self.assertFalse(np.allclose(first_replay_action, deterministic_action))
        np.testing.assert_allclose(first_replay_action, first_executed_action)
        self.assertLessEqual(float(first_replay_action.max()), 0.40 + 1e-12)


def _variant(name: str) -> dict:
    return next(variant for variant in FEATURE_VARIANTS if variant["variant"] == name)


if __name__ == "__main__":
    unittest.main()
