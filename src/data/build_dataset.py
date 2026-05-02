"""Dataset construction utilities.

This module provides a minimal orchestration layer for building the first
returns dataset from project configuration. It does not persist files or perform
feature engineering.
"""

import pandas as pd

from src.data.download import download_prices
from src.data.preprocess import compute_returns
from src.utils.config import load_config


def build_returns_dataset(config_path: str) -> pd.DataFrame:
    """Build a returns DataFrame from configuration, downloaded prices, and preprocessing."""
    config = load_config(config_path)
    data_config = config["data"]

    assets = data_config["assets"]
    frequency = data_config["frequency"]
    start_date = _optional_string(data_config["start_date"])
    end_date = _optional_string(data_config["end_date"])

    prices = download_prices(assets, start_date, end_date)

    return compute_returns(prices, assets, frequency)


def _optional_string(value) -> str | None:
    if value is None:
        return None

    return str(value)
