"""Basic benchmark return utilities for portfolio backtesting.

This module contains simple benchmark return calculations that can be used
before introducing TD3. More advanced benchmarks, such as rolling Markowitz and
risk parity, are intentionally kept in separate modules.
"""

import numpy as np
import pandas as pd


def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    """Compute equal-weight portfolio returns as row-wise average returns."""
    return returns.mean(axis=1)


def buy_and_hold_returns(
    returns: pd.DataFrame,
    initial_weights: pd.Series | None = None,
) -> pd.Series:
    """Compute buy-and-hold portfolio returns with weights drifting over time."""
    weights = _prepare_initial_weights(returns, initial_weights)
    asset_values = weights.to_numpy(dtype=float).copy()
    portfolio_returns = []

    for _, period_returns in returns.iterrows():
        asset_returns = period_returns.to_numpy(dtype=float)
        portfolio_return = float(np.dot(asset_values / asset_values.sum(), asset_returns))
        portfolio_returns.append(portfolio_return)

        asset_values *= 1.0 + asset_returns
        total_value = float(asset_values.sum())
        if total_value <= 0.0:
            raise ValueError("Buy-and-hold portfolio value must remain positive.")

    return pd.Series(portfolio_returns, index=returns.index, name="buy_and_hold")


def _prepare_initial_weights(
    returns: pd.DataFrame,
    initial_weights: pd.Series | None,
) -> pd.Series:
    if initial_weights is None:
        return pd.Series(1.0 / len(returns.columns), index=returns.columns)

    weights = initial_weights.reindex(returns.columns)

    if weights.isna().any():
        missing_assets = weights[weights.isna()].index.tolist()
        missing = ", ".join(missing_assets)
        raise KeyError(f"Missing initial weights for assets: {missing}")
    if (weights < 0.0).any():
        raise ValueError("Initial weights must be non-negative.")
    if not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("Initial weights must sum to 1.")

    return weights.astype(float)
