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
    returns = compute_returns(prices, assets, frequency)

    return _apply_date_boundaries(returns, start_date, end_date)


def _apply_date_boundaries(
    returns: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    bounded_returns = returns
    if start_date is not None:
        bounded_returns = bounded_returns.loc[bounded_returns.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        bounded_returns = bounded_returns.loc[bounded_returns.index <= pd.Timestamp(end_date)]

    return bounded_returns


def _optional_string(value) -> str | None:
    if value is None:
        return None

    return str(value)
