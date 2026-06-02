"""Tests for configuration loading and validation."""

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from src.utils.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_returns_dict_for_valid_yaml(self):
        with self._temporary_config(_valid_config()) as config_path:
            config = load_config(str(config_path))

        self.assertIsInstance(config, dict)

    def test_load_config_rejects_missing_required_field(self):
        config = _valid_config()
        del config["td3"]["gamma"]

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(KeyError, "Missing required config field: td3.gamma"):
                load_config(str(config_path))

    def test_load_config_rejects_missing_training_seed(self):
        config = _valid_config()
        del config["training"]["seed"]

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                KeyError,
                "Missing required config field: training.seed",
            ):
                load_config(str(config_path))

    def test_load_config_rejects_missing_td3_batch_size(self):
        config = _valid_config()
        del config["td3"]["batch_size"]

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                KeyError,
                "Missing required config field: td3.batch_size",
            ):
                load_config(str(config_path))

    def test_load_config_rejects_missing_td3_replay_buffer_size(self):
        config = _valid_config()
        del config["td3"]["replay_buffer_size"]

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                KeyError,
                "Missing required config field: td3.replay_buffer_size",
            ):
                load_config(str(config_path))

    def test_load_config_rejects_empty_assets(self):
        config = _valid_config()
        config["data"]["assets"] = []

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "data.assets must be a non-empty list"):
                load_config(str(config_path))

    def test_load_config_rejects_assets_that_are_not_list(self):
        config = _valid_config()
        config["data"]["assets"] = "SPY"

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "data.assets must be a non-empty list"):
                load_config(str(config_path))

    def test_load_config_rejects_duplicate_assets(self):
        config = _valid_config()
        config["data"]["assets"] = ["SPY", "TLT", "SPY"]

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "must not contain duplicate assets"):
                load_config(str(config_path))

    def test_load_config_rejects_non_string_or_empty_asset_entries(self):
        config = _valid_config()
        config["data"]["assets"] = ["SPY", ""]

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "All entries in data.assets"):
                load_config(str(config_path))

    def test_load_config_rejects_non_positive_initial_cash(self):
        config = _valid_config()
        config["environment"]["initial_cash"] = 0

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                ValueError,
                "environment.initial_cash must be greater than 0",
            ):
                load_config(str(config_path))

    def test_load_config_rejects_negative_transaction_cost(self):
        config = _valid_config()
        config["environment"]["transaction_cost"] = -0.01

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                ValueError,
                "environment.transaction_cost must be in the range",
            ):
                load_config(str(config_path))

    def test_load_config_rejects_transaction_cost_greater_than_or_equal_to_one(self):
        config = _valid_config()
        config["environment"]["transaction_cost"] = 1.0

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                ValueError,
                "environment.transaction_cost must be in the range",
            ):
                load_config(str(config_path))

    def test_load_config_accepts_asset_specific_transaction_costs(self):
        config = _valid_config()
        config["environment"]["transaction_cost_mode"] = "asset_specific"
        config["environment"]["asset_transaction_cost_bps"] = {
            "SPY": 2.0,
            "TLT": 2.0,
            "GLD": 2.0,
            "BTC-USD": 10.0,
            "CASH": 0.0,
        }

        with self._temporary_config(config) as config_path:
            loaded = load_config(str(config_path))

        self.assertEqual(loaded["environment"]["transaction_cost_mode"], "asset_specific")
        self.assertEqual(loaded["environment"]["asset_transaction_cost_bps"]["CASH"], 0.0)

    def test_load_config_rejects_unknown_transaction_cost_mode(self):
        config = _valid_config()
        config["environment"]["transaction_cost_mode"] = "unknown"

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "transaction_cost_mode"):
                load_config(str(config_path))

    def test_load_config_rejects_missing_asset_costs_in_asset_specific_mode(self):
        config = _valid_config()
        config["environment"]["transaction_cost_mode"] = "asset_specific"
        config["environment"]["asset_transaction_cost_bps"] = {
            "SPY": 2.0,
            "TLT": 2.0,
            "GLD": 2.0,
            "CASH": 0.0,
        }

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "missing assets"):
                load_config(str(config_path))

    def test_load_config_rejects_negative_asset_transaction_costs(self):
        config = _valid_config()
        config["environment"]["asset_transaction_cost_bps"] = {
            "SPY": 2.0,
            "TLT": 2.0,
            "GLD": 2.0,
            "BTC-USD": -1.0,
            "CASH": 0.0,
        }

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "BTC-USD"):
                load_config(str(config_path))

    def test_load_config_rejects_training_ratios_that_do_not_sum_to_one(self):
        config = _valid_config()
        config["training"]["test_ratio"] = 0.2

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "training ratios must sum to 1.0"):
                load_config(str(config_path))

    def test_load_config_rejects_unsupported_frequency(self):
        config = _valid_config()
        config["data"]["frequency"] = "monthly"

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "data.frequency must be one of"):
                load_config(str(config_path))

    def test_load_config_rejects_policy_delay_less_than_one(self):
        config = _valid_config()
        config["td3"]["policy_delay"] = 0

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "td3.policy_delay must be an integer"):
                load_config(str(config_path))

    def test_load_config_rejects_non_integer_training_seed(self):
        config = _valid_config()
        config["training"]["seed"] = 42.5

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "training.seed must be an integer"):
                load_config(str(config_path))

    def test_load_config_rejects_bool_training_seed(self):
        config = _valid_config()
        config["training"]["seed"] = True

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "training.seed must be an integer"):
                load_config(str(config_path))

    def test_load_config_accepts_non_negative_exploration_noise(self):
        config = _valid_config()
        config["training"]["exploration_noise"] = 0.05
        config["training"]["exploration_noise_clip"] = 0.10
        config["training"]["exploration_warmup_steps"] = 0

        with self._temporary_config(config) as config_path:
            loaded = load_config(str(config_path))

        self.assertEqual(loaded["training"]["exploration_noise"], 0.05)
        self.assertEqual(loaded["training"]["exploration_noise_clip"], 0.10)

    def test_load_config_rejects_negative_exploration_noise(self):
        config = _valid_config()
        config["training"]["exploration_noise"] = -0.01

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "training.exploration_noise"):
                load_config(str(config_path))

    def test_load_config_rejects_td3_batch_size_less_than_one(self):
        config = _valid_config()
        config["td3"]["batch_size"] = 0

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                ValueError,
                "td3.batch_size must be an integer greater than or equal to 1",
            ):
                load_config(str(config_path))

    def test_load_config_rejects_td3_replay_buffer_size_less_than_one(self):
        config = _valid_config()
        config["td3"]["replay_buffer_size"] = 0

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                ValueError,
                "td3.replay_buffer_size must be an integer greater than or equal to 1",
            ):
                load_config(str(config_path))

    def test_load_config_rejects_replay_buffer_smaller_than_batch_size(self):
        config = _valid_config()
        config["td3"]["batch_size"] = 256
        config["td3"]["replay_buffer_size"] = 255

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                ValueError,
                "td3.replay_buffer_size must be greater than or equal to td3.batch_size",
            ):
                load_config(str(config_path))

    def test_load_config_rejects_negative_reward_lambda(self):
        config = _valid_config()
        config["reward"]["lambda_turnover"] = -0.1

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                ValueError,
                "reward.lambda_turnover must be greater than or equal to 0",
            ):
                load_config(str(config_path))

    def test_load_config_rejects_non_numeric_reward_lambda(self):
        config = _valid_config()
        config["reward"]["lambda_transaction_cost"] = "0.2"

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                ValueError,
                "reward.lambda_transaction_cost must be numeric",
            ):
                load_config(str(config_path))

    def test_load_config_accepts_net_return_first_reward_mode(self):
        config = _valid_config()
        config["reward"]["reward_mode"] = "net_return_first"

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["reward"]["reward_mode"], "net_return_first")

    def test_load_config_accepts_component_legacy_reward_mode(self):
        config = _valid_config()
        config["reward"]["reward_mode"] = "component_legacy"

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["reward"]["reward_mode"], "component_legacy")

    def test_load_config_rejects_unknown_reward_mode(self):
        config = _valid_config()
        config["reward"]["reward_mode"] = "unknown"

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "reward.reward_mode"):
                load_config(str(config_path))

    def test_load_config_accepts_missing_lambda_concentration(self):
        config = _valid_config()
        config["reward"].pop("lambda_concentration", None)

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertNotIn("lambda_concentration", loaded_config["reward"])

    def test_load_config_accepts_non_negative_lambda_concentration(self):
        config = _valid_config()
        config["reward"]["lambda_concentration"] = 0.3

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["reward"]["lambda_concentration"], 0.3)

    def test_load_config_rejects_unknown_reward_field(self):
        config = _valid_config()
        config["reward"]["unknown_reward_term"] = 0.1

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                ValueError,
                "reward contains unsupported keys",
            ):
                load_config(str(config_path))

    def test_load_config_rejects_lambda_sharpe_as_active_reward_field(self):
        config = _valid_config()
        config["reward"]["lambda_sharpe"] = 0.5

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                ValueError,
                "reward contains unsupported keys",
            ):
                load_config(str(config_path))

    def test_load_config_accepts_all_active_reward_fields(self):
        config = _valid_config()
        config["reward"].update(
            {
                "lambda_concentration": 0.1,
                "use_mandate_penalty": True,
                "lambda_mandate": 0.2,
                "mandate_profile": "moderate",
                "mandate_volatility_window": 12,
                "mandate_penalty_weights": {
                    "drawdown_breach": 1.0,
                    "volatility_breach": 1.0,
                    "max_weight_breach": 1.0,
                    "effective_assets_breach": 1.0,
                    "turnover_breach": 1.0,
                },
                "use_cash_risk_off_penalty": True,
                "normal_cash_max": 0.10,
                "cash_penalty_weight": 0.025,
                "cash_risk_off_state": False,
                "cash_risk_off_column": "risk_off_state",
                "turnover_penalty_mode": "excess_linear",
                "turnover_free_band": 0.20,
                "turnover_quadratic_weight": 0.0,
            }
        )

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["reward"]["mandate_volatility_window"], 12)
        self.assertEqual(
            loaded_config["reward"]["turnover_penalty_mode"],
            "excess_linear",
        )

    def test_load_config_accepts_missing_mandate_reward_fields(self):
        config = _valid_config()

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertNotIn("use_mandate_penalty", loaded_config["reward"])

    def test_load_config_accepts_valid_mandate_reward_fields(self):
        config = _valid_config()
        config["reward"].update(
            {
                "use_mandate_penalty": True,
                "lambda_mandate": 0.5,
                "mandate_profile": "moderate",
                "mandate_penalty_weights": {
                    "drawdown_breach": 0.0,
                    "volatility_breach": 1.0,
                    "max_weight_breach": 2.0,
                    "effective_assets_breach": 1.5,
                    "turnover_breach": 0.5,
                },
            }
        )

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertTrue(loaded_config["reward"]["use_mandate_penalty"])
        self.assertEqual(loaded_config["reward"]["lambda_mandate"], 0.5)

    def test_load_config_rejects_invalid_use_mandate_penalty(self):
        config = _valid_config()
        config["reward"]["use_mandate_penalty"] = "true"

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "reward.use_mandate_penalty"):
                load_config(str(config_path))

    def test_load_config_rejects_invalid_lambda_mandate(self):
        config = _valid_config()
        config["reward"]["lambda_mandate"] = -0.1

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "reward.lambda_mandate"):
                load_config(str(config_path))

    def test_load_config_rejects_invalid_mandate_profile(self):
        config = _valid_config()
        config["reward"]["mandate_profile"] = "unknown"

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "reward.mandate_profile"):
                load_config(str(config_path))

    def test_load_config_rejects_unknown_mandate_penalty_weight_key(self):
        config = _valid_config()
        config["reward"]["mandate_penalty_weights"] = {"unknown_breach": 1.0}

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "mandate_penalty_weights"):
                load_config(str(config_path))

    def test_load_config_rejects_invalid_mandate_penalty_weight_value(self):
        config = _valid_config()
        config["reward"]["mandate_penalty_weights"] = {"max_weight_breach": True}

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "reward.mandate_penalty_weights"):
                load_config(str(config_path))

    def test_load_config_accepts_missing_cash_penalty_fields(self):
        config = _valid_config()

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertNotIn("use_cash_risk_off_penalty", loaded_config["reward"])

    def test_load_config_accepts_valid_cash_penalty_fields(self):
        config = _valid_config()
        config["reward"].update(
            {
                "use_cash_risk_off_penalty": True,
                "normal_cash_max": 0.10,
                "cash_penalty_weight": 1.0,
                "cash_risk_off_state": False,
            }
        )

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertTrue(loaded_config["reward"]["use_cash_risk_off_penalty"])
        self.assertEqual(loaded_config["reward"]["normal_cash_max"], 0.10)

    def test_load_config_rejects_invalid_use_cash_risk_off_penalty(self):
        config = _valid_config()
        config["reward"]["use_cash_risk_off_penalty"] = "true"

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "reward.use_cash_risk_off_penalty"):
                load_config(str(config_path))

    def test_load_config_rejects_invalid_normal_cash_max(self):
        invalid_values = (-0.01, 1.01, True)
        for invalid_value in invalid_values:
            config = _valid_config()
            config["reward"]["normal_cash_max"] = invalid_value
            with self.subTest(invalid_value=invalid_value):
                with self._temporary_config(config) as config_path:
                    with self.assertRaisesRegex(ValueError, "reward.normal_cash_max"):
                        load_config(str(config_path))

    def test_load_config_rejects_invalid_cash_penalty_weight(self):
        invalid_values = (-0.01, True)
        for invalid_value in invalid_values:
            config = _valid_config()
            config["reward"]["cash_penalty_weight"] = invalid_value
            with self.subTest(invalid_value=invalid_value):
                with self._temporary_config(config) as config_path:
                    with self.assertRaisesRegex(ValueError, "reward.cash_penalty_weight"):
                        load_config(str(config_path))

    def test_load_config_rejects_invalid_cash_risk_off_state(self):
        config = _valid_config()
        config["reward"]["cash_risk_off_state"] = "false"

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "reward.cash_risk_off_state"):
                load_config(str(config_path))

    def test_load_config_accepts_valid_cash_risk_off_column(self):
        config = _valid_config()
        config["reward"]["cash_risk_off_column"] = "risk_off_state"

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(
            loaded_config["reward"]["cash_risk_off_column"],
            "risk_off_state",
        )

    def test_load_config_rejects_invalid_cash_risk_off_column(self):
        invalid_values = ("", True)
        for invalid_value in invalid_values:
            config = _valid_config()
            config["reward"]["cash_risk_off_column"] = invalid_value
            with self.subTest(invalid_value=invalid_value):
                with self._temporary_config(config) as config_path:
                    with self.assertRaisesRegex(
                        ValueError,
                        "reward.cash_risk_off_column",
                    ):
                        load_config(str(config_path))

    def test_load_config_accepts_valid_turnover_penalty_fields(self):
        config = _valid_config()
        config["reward"].update(
            {
                "turnover_penalty_mode": "excess_quadratic",
                "turnover_free_band": 0.10,
                "turnover_quadratic_weight": 0.05,
            }
        )

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(
            loaded_config["reward"]["turnover_penalty_mode"],
            "excess_quadratic",
        )
        self.assertEqual(loaded_config["reward"]["turnover_free_band"], 0.10)

    def test_load_config_rejects_invalid_turnover_penalty_mode(self):
        invalid_values = ("soft", True)
        for invalid_value in invalid_values:
            config = _valid_config()
            config["reward"]["turnover_penalty_mode"] = invalid_value
            with self.subTest(invalid_value=invalid_value):
                with self._temporary_config(config) as config_path:
                    with self.assertRaisesRegex(
                        ValueError,
                        "reward.turnover_penalty_mode",
                    ):
                        load_config(str(config_path))

    def test_load_config_rejects_invalid_turnover_free_band(self):
        invalid_values = (-0.01, True)
        for invalid_value in invalid_values:
            config = _valid_config()
            config["reward"]["turnover_free_band"] = invalid_value
            with self.subTest(invalid_value=invalid_value):
                with self._temporary_config(config) as config_path:
                    with self.assertRaisesRegex(ValueError, "reward.turnover_free_band"):
                        load_config(str(config_path))

    def test_load_config_rejects_invalid_turnover_quadratic_weight(self):
        invalid_values = (-0.01, True)
        for invalid_value in invalid_values:
            config = _valid_config()
            config["reward"]["turnover_quadratic_weight"] = invalid_value
            with self.subTest(invalid_value=invalid_value):
                with self._temporary_config(config) as config_path:
                    with self.assertRaisesRegex(
                        ValueError,
                        "reward.turnover_quadratic_weight",
                    ):
                        load_config(str(config_path))

    def test_load_config_accepts_missing_features_section(self):
        config = _valid_config()

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertNotIn("features", loaded_config)

    def test_load_config_accepts_valid_v2_features_section(self):
        config = _valid_config()
        config["features"] = {
            "version": "v2",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
        }

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["features"]["version"], "v2")

    def test_load_config_accepts_valid_v3_features_section(self):
        config = _valid_config()
        config["features"] = {
            "version": "v3",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
        }

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["features"]["version"], "v3")

    def test_load_config_accepts_valid_v4_features_section(self):
        config = _valid_config()
        config["features"] = {
            "version": "v4",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
            "include_garch_features": True,
            "garch_include_relative": True,
            "garch_omega": 1e-6,
            "garch_alpha": 0.05,
            "garch_beta": 0.90,
            "garch_periods_per_year": 52,
            "garch_mode": "rolling_fitted",
            "garch_min_history": 104,
            "garch_window": 156,
            "garch_annualize": False,
            "garch_exclude_cash": True,
            "garch_fallback": "rolling_realized_vol",
        }

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["features"]["version"], "v4")

    def test_load_config_accepts_valid_v5_features_section(self):
        config = _valid_config()
        config["features"] = {
            "version": "v5",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
            "correlation_window": 12,
            "drawdown_window": 12,
            "risk_off_threshold": 2.0,
        }

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["features"]["version"], "v5")

    def test_load_config_accepts_valid_v6_features_section(self):
        config = _valid_config()
        config["features"] = {
            "version": "v6",
            "market_asset": "SPY",
            "short_window": 4,
            "medium_window": 12,
            "long_window": 26,
            "ewma_short_span": 4,
            "ewma_long_span": 12,
            "correlation_window": 12,
            "zscore_window": 52,
        }

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["features"]["version"], "v6")

    def test_load_config_accepts_valid_v7_features_section(self):
        config = _valid_config()
        config["features"] = {
            "version": "v7",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
            "macro_path": "data/processed/macro_weekly_latest.csv",
            "macro_date_column": "date",
            "garch_include_relative": True,
            "garch_mode": "rolling_fitted",
            "garch_min_history": 104,
            "garch_window": 156,
            "garch_annualize": False,
            "garch_exclude_cash": True,
            "garch_fallback": "rolling_realized_vol",
        }

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["features"]["version"], "v7")

    def test_load_config_accepts_valid_v8_features_section(self):
        config = _valid_config()
        config["features"] = {
            "version": "v8",
            "market_asset": "SPY",
            "short_window": 4,
            "long_window": 12,
            "ewma_span": 12,
            "ewma_lambda": 0.94,
            "garch_include_relative": True,
            "garch_mode": "rolling_fitted",
            "garch_min_history": 104,
            "garch_window": 156,
            "garch_annualize": False,
            "garch_exclude_cash": True,
            "garch_fallback": "rolling_realized_vol",
        }

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["features"]["version"], "v8")
        self.assertEqual(loaded_config["features"]["ewma_lambda"], 0.94)

    def test_load_config_accepts_v3_macro_path_and_macro_date_column(self):
        config = _valid_config()
        config["features"] = {
            "version": "v3",
            "market_asset": "SPY",
            "macro_path": "data/macro/local_macro.csv",
            "macro_date_column": "observation_date",
        }

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["features"]["macro_path"], "data/macro/local_macro.csv")
        self.assertEqual(loaded_config["features"]["macro_date_column"], "observation_date")

    def test_load_config_rejects_v3_empty_macro_path(self):
        config = _valid_config()
        config["features"] = {"version": "v3", "macro_path": ""}

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "features.macro_path"):
                load_config(str(config_path))

    def test_load_config_rejects_v3_empty_macro_date_column(self):
        config = _valid_config()
        config["features"] = {"version": "v3", "macro_date_column": ""}

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "features.macro_date_column"):
                load_config(str(config_path))

    def test_load_config_v2_does_not_require_macro_path(self):
        config = _valid_config()
        config["features"] = {
            "version": "v2",
            "market_asset": "SPY",
            "macro_path": "",
        }

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["features"]["version"], "v2")
        self.assertEqual(loaded_config["features"]["macro_path"], "")

    def test_load_config_rejects_unsupported_feature_version(self):
        config = _valid_config()
        config["features"] = {"version": "v9"}

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "features.version"):
                load_config(str(config_path))

    def test_load_config_rejects_invalid_v4_garch_parameters(self):
        invalid_feature_sections = (
            {"version": "v4", "include_garch_features": "yes"},
            {"version": "v4", "garch_include_relative": "yes"},
            {"version": "v4", "garch_omega": 0.0},
            {"version": "v4", "garch_alpha": -0.1},
            {"version": "v4", "garch_beta": -0.1},
            {"version": "v4", "garch_alpha": 0.5, "garch_beta": 0.5},
            {"version": "v4", "garch_periods_per_year": 0},
            {"version": "v4", "garch_mode": "full_sample_oracle"},
            {"version": "v4", "garch_min_history": 0},
            {"version": "v4", "garch_window": 0},
            {"version": "v4", "garch_annualize": "no"},
            {"version": "v4", "garch_exclude_cash": "yes"},
            {"version": "v4", "garch_fallback": "deterministic_filter"},
        )

        for features in invalid_feature_sections:
            config = _valid_config()
            config["features"] = features
            with self.subTest(features=features):
                with self._temporary_config(config) as config_path:
                    with self.assertRaises(ValueError):
                        load_config(str(config_path))

    def test_load_config_rejects_invalid_v5_feature_parameters(self):
        invalid_feature_sections = (
            {"version": "v5", "correlation_window": 1},
            {"version": "v5", "drawdown_window": 1},
            {"version": "v5", "risk_off_threshold": -0.1},
            {"version": "v5", "risk_off_threshold": True},
        )

        for features in invalid_feature_sections:
            config = _valid_config()
            config["features"] = features
            with self.subTest(features=features):
                with self._temporary_config(config) as config_path:
                    with self.assertRaises(ValueError):
                        load_config(str(config_path))

    def test_load_config_rejects_invalid_v6_feature_parameters(self):
        invalid_feature_sections = (
            {"version": "v6", "short_window": 1},
            {"version": "v6", "medium_window": 1},
            {"version": "v6", "long_window": 1},
            {"version": "v6", "ewma_short_span": 1},
            {"version": "v6", "ewma_long_span": 1},
            {"version": "v6", "correlation_window": 1},
            {"version": "v6", "zscore_window": 1},
            {"version": "v6", "short_window": 12, "medium_window": 4},
            {"version": "v6", "medium_window": 26, "long_window": 12},
        )

        for features in invalid_feature_sections:
            config = _valid_config()
            config["features"] = features
            with self.subTest(features=features):
                with self._temporary_config(config) as config_path:
                    with self.assertRaises(ValueError):
                        load_config(str(config_path))

    def test_load_config_rejects_invalid_v8_ewma_lambda(self):
        invalid_feature_sections = (
            {"version": "v8", "ewma_lambda": 0.0},
            {"version": "v8", "ewma_lambda": 1.0},
            {"version": "v8", "ewma_lambda": True},
            {"version": "v8", "ewma_lambda": "0.94"},
        )

        for features in invalid_feature_sections:
            config = _valid_config()
            config["features"] = features
            with self.subTest(features=features):
                with self._temporary_config(config) as config_path:
                    with self.assertRaises(ValueError):
                        load_config(str(config_path))

    def test_load_config_rejects_v2_market_asset_not_in_data_assets(self):
        config = _valid_config()
        config["features"] = {"version": "v2", "market_asset": "QQQ"}

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(
                ValueError,
                "features.market_asset must exist in data.assets",
            ):
                load_config(str(config_path))

    def test_load_config_rejects_invalid_v2_feature_parameters(self):
        invalid_feature_sections = (
            {"version": "v2", "market_asset": ""},
            {"version": "v2", "short_window": 1},
            {"version": "v2", "long_window": 1},
            {"version": "v2", "ewma_span": 1},
            {"version": "v2", "short_window": 12, "long_window": 4},
        )

        for features in invalid_feature_sections:
            config = _valid_config()
            config["features"] = features
            with self.subTest(features=features):
                with self._temporary_config(config) as config_path:
                    with self.assertRaises(ValueError):
                        load_config(str(config_path))

    def test_load_config_accepts_cash_return_model_bil_proxy(self):
        config = _valid_config()
        config["data"]["cash_return_model"] = "bil_proxy"
        config["data"]["cash_proxy_asset"] = "BIL"

        with self._temporary_config(config) as config_path:
            loaded_config = load_config(str(config_path))

        self.assertEqual(loaded_config["data"]["cash_return_model"], "bil_proxy")
        self.assertEqual(loaded_config["data"]["cash_proxy_asset"], "BIL")

    def test_load_config_rejects_invalid_cash_return_model(self):
        config = _valid_config()
        config["data"]["cash_return_model"] = "money_market_magic"

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "cash_return_model"):
                load_config(str(config_path))

    def test_load_config_rejects_blank_cash_proxy_asset(self):
        config = _valid_config()
        config["data"]["cash_return_model"] = "bil_proxy"
        config["data"]["cash_proxy_asset"] = ""

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "cash_proxy_asset"):
                load_config(str(config_path))

    def _temporary_config(self, config: dict):
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        class TemporaryConfig:
            def __enter__(self_inner):
                return config_path

            def __exit__(self_inner, exc_type, exc_value, traceback):
                temp_dir.cleanup()

        return TemporaryConfig()


def _valid_config() -> dict:
    return copy.deepcopy(
        {
            "project": {
                "name": "portfolio_drl_td3_test",
                "description": "Temporary test config",
            },
            "data": {
                "assets": ["SPY", "TLT", "GLD", "BTC-USD", "CASH"],
                "frequency": "weekly",
                "start_date": "2020-01-01",
                "end_date": "2024-01-01",
            },
            "environment": {
                "initial_cash": 100000,
                "transaction_cost": 0.001,
                "allow_short": False,
                "max_weight_per_asset": 1.0,
            },
            "reward": {
                "lambda_return": 1.0,
                "lambda_drawdown": 1.0,
                "lambda_transaction_cost": 0.2,
                "lambda_turnover": 0.1,
            },
            "td3": {
                "actor_learning_rate": 0.0003,
                "critic_learning_rate": 0.0003,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "policy_delay": 2,
                "batch_size": 256,
                "replay_buffer_size": 100000,
            },
            "training": {
                "seed": 42,
                "episodes": 500,
                "train_ratio": 0.7,
                "validation_ratio": 0.15,
                "test_ratio": 0.15,
            },
        }
    )


if __name__ == "__main__":
    unittest.main()
