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
        config["features"] = {"version": "v4"}

        with self._temporary_config(config) as config_path:
            with self.assertRaisesRegex(ValueError, "features.version"):
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
                "lambda_sharpe": 0.5,
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
