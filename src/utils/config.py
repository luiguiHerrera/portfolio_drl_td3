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
    "lambda_drawdown",
    "lambda_transaction_cost",
    "lambda_turnover",
    "lambda_concentration",
    "lambda_mandate",
)
MANDATE_PROFILES = {"conservative", "moderate", "aggressive"}
MANDATE_PENALTY_WEIGHT_KEYS = {
    "drawdown_breach",
    "volatility_breach",
    "max_weight_breach",
    "effective_assets_breach",
    "turnover_breach",
}
TURNOVER_PENALTY_MODES = {
    "linear",
    "none",
    "excess_linear",
    "excess_quadratic",
}
SUPPORTED_REWARD_FIELDS = set(OPTIONAL_REWARD_LAMBDAS) | {
    "use_mandate_penalty",
    "mandate_profile",
    "mandate_penalty_weights",
    "mandate_volatility_window",
    "use_cash_risk_off_penalty",
    "normal_cash_max",
    "cash_penalty_weight",
    "cash_risk_off_state",
    "cash_risk_off_column",
    "turnover_penalty_mode",
    "turnover_free_band",
    "turnover_quadratic_weight",
}


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

    returns_path = config["data"].get("returns_path")
    returns_date_column = config["data"].get("returns_date_column")
    if returns_path is not None and (
        not isinstance(returns_path, str) or not returns_path.strip()
    ):
        raise ValueError("Config field data.returns_path must be a non-empty string.")
    if returns_date_column is not None and (
        not isinstance(returns_date_column, str) or not returns_date_column.strip()
    ):
        raise ValueError(
            "Config field data.returns_date_column must be a non-empty string."
        )


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

    _validate_supported_reward_fields(config["reward"])

    for field_name in OPTIONAL_REWARD_LAMBDAS:
        if field_name in config["reward"]:
            _validate_non_negative_number(
                config["reward"][field_name],
                f"reward.{field_name}",
            )

    _validate_reward_mandate_fields(config["reward"])
    _validate_reward_cash_penalty_fields(config["reward"])
    _validate_reward_turnover_penalty_fields(config["reward"])


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
    if version not in {"v1", "v2", "v3", "v4", "v5", "v6"}:
        raise ValueError(
            "Config field features.version must be one of: v1, v2, v3, v4, v5, v6."
        )
    if version == "v1":
        return

    market_asset = features.get("market_asset", "SPY")
    if not isinstance(market_asset, str) or not market_asset.strip():
        raise ValueError("Config field features.market_asset must be a non-empty string.")
    if market_asset not in config["data"]["assets"]:
        raise ValueError("Config field features.market_asset must exist in data.assets.")
    if version == "v6":
        _validate_v6_financial_state_config(features)
        return

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
    if version == "v3":
        macro_path = features.get("macro_path")
        macro_date_column = features.get("macro_date_column")
        if macro_path is not None and (
            not isinstance(macro_path, str) or not macro_path.strip()
        ):
            raise ValueError("Config field features.macro_path must be a non-empty string.")
        if macro_date_column is not None and (
            not isinstance(macro_date_column, str) or not macro_date_column.strip()
        ):
            raise ValueError(
                "Config field features.macro_date_column must be a non-empty string."
            )
    if version == "v4":
        _validate_v4_garch_feature_config(features)
    if version == "v5":
        _validate_v5_regime_feature_config(features)


def _validate_assets(config: dict) -> None:
    assets = config["data"]["assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError("Config field data.assets must be a non-empty list.")
    if any(not isinstance(asset, str) or not asset.strip() for asset in assets):
        raise ValueError("All entries in data.assets must be non-empty strings.")
    if len(assets) != len(set(assets)):
        raise ValueError("Config field data.assets must not contain duplicate assets.")


def _validate_supported_reward_fields(reward: dict) -> None:
    unsupported_keys = set(reward) - SUPPORTED_REWARD_FIELDS
    if unsupported_keys:
        raise ValueError(
            "Config field reward contains unsupported keys: "
            f"{sorted(unsupported_keys)}."
        )


def _validate_reward_mandate_fields(reward: dict) -> None:
    use_mandate_penalty = reward.get("use_mandate_penalty")
    if use_mandate_penalty is not None and not isinstance(use_mandate_penalty, bool):
        raise ValueError("Config field reward.use_mandate_penalty must be a bool.")

    mandate_profile = reward.get("mandate_profile")
    if mandate_profile is not None and mandate_profile not in MANDATE_PROFILES:
        valid_profiles = ", ".join(sorted(MANDATE_PROFILES))
        raise ValueError(
            "Config field reward.mandate_profile must be one of: "
            f"{valid_profiles}."
        )

    mandate_volatility_window = reward.get("mandate_volatility_window")
    if mandate_volatility_window is not None:
        _validate_integer_at_least_one(
            mandate_volatility_window,
            "reward.mandate_volatility_window",
        )

    mandate_penalty_weights = reward.get("mandate_penalty_weights")
    if mandate_penalty_weights is None:
        return
    if not isinstance(mandate_penalty_weights, dict):
        raise ValueError("Config field reward.mandate_penalty_weights must be a mapping.")

    unknown_keys = set(mandate_penalty_weights) - MANDATE_PENALTY_WEIGHT_KEYS
    if unknown_keys:
        raise ValueError(
            "Config field reward.mandate_penalty_weights contains unsupported keys: "
            f"{sorted(unknown_keys)}."
        )
    for key, value in mandate_penalty_weights.items():
        _validate_non_negative_number(value, f"reward.mandate_penalty_weights.{key}")


def _validate_reward_cash_penalty_fields(reward: dict) -> None:
    use_cash_risk_off_penalty = reward.get("use_cash_risk_off_penalty")
    if use_cash_risk_off_penalty is not None and not isinstance(
        use_cash_risk_off_penalty,
        bool,
    ):
        raise ValueError("Config field reward.use_cash_risk_off_penalty must be a bool.")

    normal_cash_max = reward.get("normal_cash_max")
    if normal_cash_max is not None:
        if not _is_number(normal_cash_max) or normal_cash_max < 0.0 or normal_cash_max > 1.0:
            raise ValueError(
                "Config field reward.normal_cash_max must be numeric and in the range [0, 1]."
            )

    cash_penalty_weight = reward.get("cash_penalty_weight")
    if cash_penalty_weight is not None:
        _validate_non_negative_number(
            cash_penalty_weight,
            "reward.cash_penalty_weight",
        )

    cash_risk_off_state = reward.get("cash_risk_off_state")
    if cash_risk_off_state is not None and not isinstance(cash_risk_off_state, bool):
        raise ValueError("Config field reward.cash_risk_off_state must be a bool.")

    cash_risk_off_column = reward.get("cash_risk_off_column")
    if cash_risk_off_column is not None and (
        not isinstance(cash_risk_off_column, str)
        or not cash_risk_off_column.strip()
    ):
        raise ValueError(
            "Config field reward.cash_risk_off_column must be a non-empty string."
        )


def _validate_reward_turnover_penalty_fields(reward: dict) -> None:
    turnover_penalty_mode = reward.get("turnover_penalty_mode")
    if turnover_penalty_mode is not None and (
        not isinstance(turnover_penalty_mode, str)
        or turnover_penalty_mode not in TURNOVER_PENALTY_MODES
    ):
        valid_modes = ", ".join(sorted(TURNOVER_PENALTY_MODES))
        raise ValueError(
            "Config field reward.turnover_penalty_mode must be one of: "
            f"{valid_modes}."
        )

    turnover_free_band = reward.get("turnover_free_band")
    if turnover_free_band is not None:
        _validate_non_negative_number(
            turnover_free_band,
            "reward.turnover_free_band",
        )

    turnover_quadratic_weight = reward.get("turnover_quadratic_weight")
    if turnover_quadratic_weight is not None:
        _validate_non_negative_number(
            turnover_quadratic_weight,
            "reward.turnover_quadratic_weight",
        )


def _validate_v4_garch_feature_config(features: dict) -> None:
    for field_name in (
        "include_garch_features",
        "garch_include_relative",
        "garch_annualize",
        "garch_exclude_cash",
    ):
        value = features.get(field_name)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"Config field features.{field_name} must be a bool.")

    omega = features.get("garch_omega", 1e-6)
    alpha = features.get("garch_alpha", 0.05)
    beta = features.get("garch_beta", 0.90)
    periods_per_year = features.get("garch_periods_per_year", 52)
    if not _is_number(omega) or omega <= 0.0:
        raise ValueError("Config field features.garch_omega must be greater than 0.")
    if not _is_number(alpha) or alpha < 0.0:
        raise ValueError(
            "Config field features.garch_alpha must be greater than or equal to 0."
        )
    if not _is_number(beta) or beta < 0.0:
        raise ValueError(
            "Config field features.garch_beta must be greater than or equal to 0."
        )
    if alpha + beta >= 1.0:
        raise ValueError("Config field features.garch_alpha + garch_beta must be less than 1.")
    _validate_integer_at_least_one(periods_per_year, "features.garch_periods_per_year")
    garch_mode = features.get("garch_mode", "deterministic_filter")
    if garch_mode not in {"deterministic_filter", "rolling_fitted"}:
        raise ValueError(
            "Config field features.garch_mode must be deterministic_filter or rolling_fitted."
        )
    garch_min_history = features.get("garch_min_history", 104)
    _validate_integer_at_least_one(garch_min_history, "features.garch_min_history")
    garch_window = features.get("garch_window", 156)
    if garch_window is not None:
        _validate_integer_at_least_one(garch_window, "features.garch_window")
    garch_fallback = features.get("garch_fallback", "rolling_realized_vol")
    if garch_fallback != "rolling_realized_vol":
        raise ValueError(
            "Config field features.garch_fallback must be rolling_realized_vol."
        )


def _validate_v5_regime_feature_config(features: dict) -> None:
    correlation_window = features.get("correlation_window", 12)
    drawdown_window = features.get("drawdown_window", 12)
    risk_off_threshold = features.get("risk_off_threshold", 2.0)
    _validate_integer_at_least_two(correlation_window, "features.correlation_window")
    _validate_integer_at_least_two(drawdown_window, "features.drawdown_window")
    _validate_non_negative_number(
        risk_off_threshold,
        "features.risk_off_threshold",
    )


def _validate_v6_financial_state_config(features: dict) -> None:
    short_window = features.get("short_window", 4)
    medium_window = features.get("medium_window", 12)
    long_window = features.get("long_window", 26)
    ewma_short_span = features.get("ewma_short_span", 4)
    ewma_long_span = features.get("ewma_long_span", 12)
    correlation_window = features.get("correlation_window", 12)
    zscore_window = features.get("zscore_window", 52)

    _validate_integer_at_least_two(short_window, "features.short_window")
    _validate_integer_at_least_two(medium_window, "features.medium_window")
    _validate_integer_at_least_two(long_window, "features.long_window")
    _validate_integer_at_least_two(ewma_short_span, "features.ewma_short_span")
    _validate_integer_at_least_two(ewma_long_span, "features.ewma_long_span")
    _validate_integer_at_least_two(correlation_window, "features.correlation_window")
    _validate_integer_at_least_two(zscore_window, "features.zscore_window")
    if medium_window < short_window:
        raise ValueError(
            "Config field features.medium_window must be greater than or equal to "
            "features.short_window."
        )
    if long_window < medium_window:
        raise ValueError(
            "Config field features.long_window must be greater than or equal to "
            "features.medium_window."
        )


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
