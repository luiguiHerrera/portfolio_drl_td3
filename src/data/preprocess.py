"""Price preprocessing utilities.

This module converts price series into return series for the portfolio
allocation pipeline. The default CASH exposure remains a zero-return synthetic
asset. Robustness datasets may instead map an investable short-term Treasury
ETF proxy into the CASH return column while preserving the action-space asset
name.
"""

import pandas as pd


SYNTHETIC_ASSETS = {"CASH"}
CASH_RETURN_MODEL_ZERO = "zero"
CASH_RETURN_MODEL_BIL_PROXY = "bil_proxy"
CASH_PROXY_ASSETS = {
    CASH_RETURN_MODEL_BIL_PROXY: "BIL",
}


def compute_returns(
    prices: pd.DataFrame,
    assets: list[str],
    frequency: str = "weekly",
    cash_return_model: str = CASH_RETURN_MODEL_ZERO,
    cash_proxy_asset: str | None = None,
) -> pd.DataFrame:
    """Compute asset returns and add configured CASH returns when requested."""
    cash_return_model = _validate_cash_return_model(cash_return_model)
    resolved_cash_proxy_asset = _resolve_cash_proxy_asset(
        cash_return_model,
        cash_proxy_asset,
    )
    market_assets = [asset for asset in assets if asset not in SYNTHETIC_ASSETS]
    price_assets = list(market_assets)
    if resolved_cash_proxy_asset and resolved_cash_proxy_asset not in price_assets:
        price_assets.append(resolved_cash_proxy_asset)
    missing_assets = [asset for asset in price_assets if asset not in prices.columns]

    if missing_assets:
        missing = ", ".join(missing_assets)
        raise KeyError(f"Missing price columns for assets: {missing}")

    market_prices = prices.loc[:, price_assets]

    if frequency == "weekly":
        market_prices = market_prices.resample("W-FRI").last()
    elif frequency != "daily":
        raise ValueError(f"Unsupported return frequency: {frequency}")

    returns = market_prices.pct_change().dropna()

    if "CASH" in assets:
        if cash_return_model == CASH_RETURN_MODEL_ZERO:
            returns["CASH"] = 0.0
        elif cash_return_model == CASH_RETURN_MODEL_BIL_PROXY:
            returns["CASH"] = returns[resolved_cash_proxy_asset]
        else:
            raise ValueError(f"Unsupported cash_return_model: {cash_return_model}")

    return returns.loc[:, assets]


def _validate_cash_return_model(value: str | None) -> str:
    model = CASH_RETURN_MODEL_ZERO if value is None else str(value).strip()
    if model not in {CASH_RETURN_MODEL_ZERO, CASH_RETURN_MODEL_BIL_PROXY}:
        raise ValueError(
            "cash_return_model must be one of: zero, bil_proxy."
        )
    return model


def _resolve_cash_proxy_asset(
    cash_return_model: str,
    cash_proxy_asset: str | None,
) -> str | None:
    if cash_return_model == CASH_RETURN_MODEL_ZERO:
        return None
    proxy_asset = cash_proxy_asset or CASH_PROXY_ASSETS[cash_return_model]
    if not isinstance(proxy_asset, str) or not proxy_asset.strip():
        raise ValueError("cash_proxy_asset must be a non-empty string.")
    return proxy_asset.strip()
