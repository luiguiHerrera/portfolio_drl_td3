"""Dataset construction utilities.

This module provides a minimal orchestration layer for building the first
returns dataset from project configuration. It does not persist files or perform
feature engineering.
"""

import pandas as pd
from pathlib import Path

from src.data.download import download_prices
from src.data.preprocess import (
    CASH_RETURN_MODEL_BIL_PROXY,
    CASH_RETURN_MODEL_ZERO,
    compute_returns,
)
from src.utils.config import load_config


def build_returns_dataset(config_path: str) -> pd.DataFrame:
    """Build a returns DataFrame from configuration, downloaded prices, and preprocessing."""
    config = load_config(config_path)
    data_config = config["data"]

    assets = data_config["assets"]
    frequency = data_config["frequency"]
    start_date = _optional_string(data_config["start_date"])
    end_date = _optional_string(data_config["end_date"])
    returns_path = data_config.get("returns_path")
    if returns_path is not None:
        returns = _load_returns_snapshot(
            path=returns_path,
            assets=assets,
            date_column=data_config.get("returns_date_column", "date"),
        )
        return _apply_date_boundaries(returns, start_date, end_date)

    cash_return_model = data_config.get("cash_return_model", CASH_RETURN_MODEL_ZERO)
    cash_proxy_asset = data_config.get("cash_proxy_asset")
    extra_assets = []
    if cash_return_model == CASH_RETURN_MODEL_BIL_PROXY:
        extra_assets.append(cash_proxy_asset or "BIL")
    prices = download_prices(assets, start_date, end_date, extra_assets=extra_assets)
    returns = compute_returns(
        prices,
        assets,
        frequency,
        cash_return_model=cash_return_model,
        cash_proxy_asset=cash_proxy_asset,
    )

    return _apply_date_boundaries(returns, start_date, end_date)


def _load_returns_snapshot(
    path: str,
    assets: list[str],
    date_column: str = "date",
) -> pd.DataFrame:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Returns snapshot not found: {path}")

    snapshot = pd.read_csv(snapshot_path)
    if date_column not in snapshot.columns:
        raise KeyError(f"Returns snapshot is missing date column: {date_column}")

    snapshot[date_column] = pd.to_datetime(snapshot[date_column], errors="coerce")
    snapshot = snapshot.dropna(subset=[date_column])
    snapshot = snapshot.sort_values(date_column)
    snapshot = snapshot.drop_duplicates(subset=[date_column], keep="last")
    snapshot = snapshot.set_index(date_column)
    snapshot.index.name = None

    missing_assets = [asset for asset in assets if asset not in snapshot.columns]
    if missing_assets:
        raise KeyError(f"Returns snapshot is missing asset columns: {missing_assets}")

    returns = snapshot.loc[:, assets].apply(pd.to_numeric, errors="coerce").dropna()
    if returns.empty:
        raise ValueError("Returns snapshot has no usable return rows.")

    return returns


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
