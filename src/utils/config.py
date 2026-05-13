"""Configuration loading and validation utilities.

This module centralizes YAML configuration loading for the project and performs
schema checks required by the TD3 portfolio allocation pipeline. It does not
infer model dimensions, build datasets, create environments, or train models.
"""

from pathlib import Path

import yaml


REQUIRED_FIELDS = (
    ("project", "name"),
    ("data", "assets"),
    ("data", "frequency"),
    ("environment", "initial_cash"),
    ("environment", "transaction_cost"),
    ("reward",),
    ("td3", "actor_learning_rate"),
    ("td3", "critic_learning_rate"),
    ("td3", "gamma"),
    ("td3", "tau"),
    ("td3", "policy_noise"),
    ("td3", "noise_clip"),
    ("td3", "policy_delay"),
    ("td3", "batch_size"),
    ("td3", "replay_buffer_size"),
    ("training", "seed"),
    ("training", "train_ratio"),
    ("training", "validation_ratio"),
    ("training", "test_ratio"),
    ("training", "episodes"),
)
SUPPORTED_FREQUENCIES = {"daily", "weekly"}
RATIO_SUM_TOLERANCE = 1e-8
OPTIONAL_REWARD_LAMBDAS = (
    "lambda_return",
    "lambda_sharpe",
    "lambda_drawdown",
    "lambda_transaction_cost",
    "lambda_turnover",
    "lambda_concentration",
)


def load_config(path: str) -> dict:
    """Load and validate a YAML configuration file."""
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        config = {}

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    _validate_required_fields(config)
    _validate_project(config)
    _validate_data(config)
    _validate_environment(config)
    _validate_reward(config)
    _validate_td3(config)
    _validate_training(config)
    _validate_features(config)

    return config


def _validate_required_fields(config: dict) -> None:
    for field_path in REQUIRED_FIELDS:
        current = config
        for key in field_path:
            if not isinstance(current, dict) or key not in current:
                dotted_path = ".".join(field_path)
                raise KeyError(f"Missing required config field: {dotted_path}")
            current = current[key]


def _validate_project(config: dict) -> None:
    project_name = config["project"]["name"]
    if not isinstance(project_name, str) or not project_name.strip():
        raise ValueError("Config field project.name must be a non-empty string.")


def _validate_data(config: dict) -> None:
    frequency = config["data"]["frequency"]
    _validate_assets(config)

    if not isinstance(frequency, str):
        raise ValueError("Config field data.frequency must be a string.")
    if frequency not in SUPPORTED_FREQUENCIES:
        valid_values = ", ".join(sorted(SUPPORTED_FREQUENCIES))
        raise ValueError(f"Config field data.frequency must be one of: {valid_values}.")


def _validate_environment(config: dict) -> None:
    initial_cash = config["environment"]["initial_cash"]
    transaction_cost = config["environment"]["transaction_cost"]

    _validate_positive_number(initial_cash, "environment.initial_cash")
    if not _is_number(transaction_cost):
        raise ValueError("Config field environment.transaction_cost must be numeric.")
    if transaction_cost < 0.0 or transaction_cost >= 1.0:
        raise ValueError("Config field environment.transaction_cost must be in the range [0, 1).")


def _validate_reward(config: dict) -> None:
    if not isinstance(config["reward"], dict):
        raise ValueError("Config field reward must be a mapping.")

    for field_name in OPTIONAL_REWARD_LAMBDAS:
        if field_name in config["reward"]:
            _validate_non_negative_number(
                config["reward"][field_name],
                f"reward.{field_name}",
            )


def _validate_td3(config: dict) -> None:
    td3 = config["td3"]

    _validate_positive_number(td3["actor_learning_rate"], "td3.actor_learning_rate")
    _validate_positive_number(td3["critic_learning_rate"], "td3.critic_learning_rate")
    _validate_number_in_range(td3["gamma"], "td3.gamma", lower=0.0, upper=1.0)
    _validate_number_in_range(td3["tau"], "td3.tau", lower=0.0, upper=1.0)
    _validate_non_negative_number(td3["policy_noise"], "td3.policy_noise")
    _validate_non_negative_number(td3["noise_clip"], "td3.noise_clip")

    _validate_integer_at_least_one(td3["policy_delay"], "td3.policy_delay")
    _validate_integer_at_least_one(td3["batch_size"], "td3.batch_size")
    _validate_integer_at_least_one(td3["replay_buffer_size"], "td3.replay_buffer_size")

    if td3["replay_buffer_size"] < td3["batch_size"]:
        raise ValueError(
            "Config field td3.replay_buffer_size must be greater than or equal to td3.batch_size."
        )


def _validate_training(config: dict) -> None:
    training = config["training"]

    _validate_positive_number(training["train_ratio"], "training.train_ratio")
    _validate_positive_number(training["validation_ratio"], "training.validation_ratio")
    _validate_positive_number(training["test_ratio"], "training.test_ratio")

    _validate_integer(training["seed"], "training.seed")
    _validate_integer_at_least_one(training["episodes"], "training.episodes")

    _validate_ratio_sum(config)


def _validate_features(config: dict) -> None:
    features = config.get("features")
    if features is None:
        return
    if not isinstance(features, dict):
        raise ValueError("Config field features must be a mapping.")

    version = features.get("version", "v1")
    if version not in {"v1", "v2"}:
        raise ValueError("Config field features.version must be one of: v1, v2.")
    if version == "v1":
        return

    market_asset = features.get("market_asset", "SPY")
    if not isinstance(market_asset, str) or not market_asset.strip():
        raise ValueError("Config field features.market_asset must be a non-empty string.")
    if market_asset not in config["data"]["assets"]:
        raise ValueError("Config field features.market_asset must exist in data.assets.")

    short_window = features.get("short_window", 4)
    long_window = features.get("long_window", 12)
    ewma_span = features.get("ewma_span", 12)
    _validate_integer_at_least_two(short_window, "features.short_window")
    _validate_integer_at_least_two(long_window, "features.long_window")
    _validate_integer_at_least_two(ewma_span, "features.ewma_span")
    if long_window < short_window:
        raise ValueError(
            "Config field features.long_window must be greater than or equal to "
            "features.short_window."
        )


def _validate_assets(config: dict) -> None:
    assets = config["data"]["assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError("Config field data.assets must be a non-empty list.")
    if any(not isinstance(asset, str) or not asset.strip() for asset in assets):
        raise ValueError("All entries in data.assets must be non-empty strings.")
    if len(assets) != len(set(assets)):
        raise ValueError("Config field data.assets must not contain duplicate assets.")


def _validate_ratio_sum(config: dict) -> None:
    training = config["training"]
    ratio_sum = (
        training["train_ratio"]
        + training["validation_ratio"]
        + training["test_ratio"]
    )

    if abs(ratio_sum - 1.0) > RATIO_SUM_TOLERANCE:
        raise ValueError("Config field training ratios must sum to 1.0.")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_positive_number(value, field_name: str) -> None:
    if not _is_number(value):
        raise ValueError(f"Config field {field_name} must be numeric.")
    if value <= 0.0:
        raise ValueError(f"Config field {field_name} must be greater than 0.")


def _validate_non_negative_number(value, field_name: str) -> None:
    if not _is_number(value):
        raise ValueError(f"Config field {field_name} must be numeric.")
    if value < 0.0:
        raise ValueError(f"Config field {field_name} must be greater than or equal to 0.")


def _validate_number_in_range(value, field_name: str, lower: float, upper: float) -> None:
    if not _is_number(value):
        raise ValueError(f"Config field {field_name} must be numeric.")
    if value <= lower or value > upper:
        raise ValueError(f"Config field {field_name} must be in the range ({lower}, {upper}].")


def _validate_integer(value, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Config field {field_name} must be an integer.")


def _validate_integer_at_least_one(value, field_name: str) -> None:
    _validate_integer(value, field_name)
    if value < 1:
        raise ValueError(
            f"Config field {field_name} must be an integer greater than or equal to 1."
        )


def _validate_integer_at_least_two(value, field_name: str) -> None:
    _validate_integer(value, field_name)
    if value < 2:
        raise ValueError(
            f"Config field {field_name} must be an integer greater than or equal to 2."
        )
