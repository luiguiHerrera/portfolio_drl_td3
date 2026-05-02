"""Configuration loading and validation utilities.

This module centralizes YAML configuration loading for the project and performs
minimal schema checks required by the early TD3 portfolio allocation pipeline.
It does not infer model dimensions or implement domain logic.
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
    ("td3",),
    ("training",),
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
    _validate_assets(config)

    return config


def _validate_required_fields(config: dict) -> None:
    for field_path in REQUIRED_FIELDS:
        current = config
        for key in field_path:
            if not isinstance(current, dict) or key not in current:
                dotted_path = ".".join(field_path)
                raise KeyError(f"Missing required config field: {dotted_path}")
            current = current[key]


def _validate_assets(config: dict) -> None:
    assets = config["data"]["assets"]
    if not assets:
        raise ValueError("Config field data.assets must not be empty.")
