"""Market price download utilities.

This module provides the first minimal data-ingestion boundary for the project.
It downloads market asset prices from yfinance while leaving synthetic assets,
such as CASH, to be handled by preprocessing.
"""

import pandas as pd
import yfinance as yf


SYNTHETIC_ASSETS = {"CASH"}


def download_prices(
    assets: list[str],
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """Download adjusted close prices for non-synthetic assets."""
    market_assets = [asset for asset in assets if asset not in SYNTHETIC_ASSETS]

    if not market_assets:
        return pd.DataFrame()

    raw_prices = yf.download(
        tickers=market_assets,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if isinstance(raw_prices.columns, pd.MultiIndex):
        prices = raw_prices["Adj Close"]
    else:
        prices = raw_prices[["Adj Close"]]
        prices.columns = market_assets

    return prices.loc[:, market_assets]
