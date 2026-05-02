"""Price preprocessing utilities.

This module converts price series into return series for the portfolio
allocation pipeline. Synthetic CASH exposure is represented as a zero-return
column aligned with market asset returns.
"""

import pandas as pd


SYNTHETIC_ASSETS = {"CASH"}


def compute_returns(
    prices: pd.DataFrame,
    assets: list[str],
    frequency: str = "weekly",
) -> pd.DataFrame:
    """Compute asset returns and add synthetic CASH returns when requested."""
    market_assets = [asset for asset in assets if asset not in SYNTHETIC_ASSETS]
    missing_assets = [asset for asset in market_assets if asset not in prices.columns]

    if missing_assets:
        missing = ", ".join(missing_assets)
        raise KeyError(f"Missing price columns for assets: {missing}")

    market_prices = prices.loc[:, market_assets]

    if frequency == "weekly":
        market_prices = market_prices.resample("W-FRI").last()
    elif frequency != "daily":
        raise ValueError(f"Unsupported return frequency: {frequency}")

    returns = market_prices.pct_change().dropna()

    if "CASH" in assets:
        returns["CASH"] = 0.0

    return returns.loc[:, assets]
