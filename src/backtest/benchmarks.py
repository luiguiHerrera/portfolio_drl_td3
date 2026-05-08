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


def equal_weight_rebalanced_benchmark(
    returns: pd.DataFrame,
    transaction_cost: float = 0.0,
    initial_weights: pd.Series | None = None,
) -> dict:
    """Compute equal-weight rebalanced benchmark returns with transaction costs.

    Returned weights are the post-rebalance target weights for each period.
    """
    if returns.empty:
        raise ValueError("returns must not be empty.")
    if transaction_cost < 0.0:
        raise ValueError("transaction_cost must be non-negative.")

    target_weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    previous_weights = _prepare_initial_weights(returns, initial_weights)
    gross_returns = []
    net_returns = []
    turnover_values = []
    transaction_cost_values = []
    weights = []

    for _, period_returns in returns.iterrows():
        asset_returns = period_returns.to_numpy(dtype=float)
        target_weights_array = target_weights.to_numpy(dtype=float)
        previous_weights_array = previous_weights.to_numpy(dtype=float)
        portfolio_return = float(np.dot(previous_weights_array, asset_returns))
        post_return_values = previous_weights_array * (1.0 + asset_returns)
        post_return_total_value = float(post_return_values.sum())
        if post_return_total_value <= 0.0:
            raise ValueError("Equal-weight rebalanced portfolio value must remain positive.")

        drifted_weights = post_return_values / post_return_total_value
        turnover = float(np.sum(np.abs(target_weights_array - drifted_weights)))
        realized_transaction_cost = float(transaction_cost * turnover)
        net_return = portfolio_return - realized_transaction_cost

        gross_returns.append(portfolio_return)
        net_returns.append(net_return)
        turnover_values.append(turnover)
        transaction_cost_values.append(realized_transaction_cost)
        weights.append(target_weights_array.copy())
        previous_weights = target_weights

    return {
        "gross_returns": pd.Series(
            gross_returns,
            index=returns.index,
            name="equal_weight_rebalanced_gross",
        ),
        "net_returns": pd.Series(
            net_returns,
            index=returns.index,
            name="equal_weight_rebalanced_net",
        ),
        "turnover": pd.Series(
            turnover_values,
            index=returns.index,
            name="equal_weight_rebalanced_turnover",
        ),
        "transaction_costs": pd.Series(
            transaction_cost_values,
            index=returns.index,
            name="equal_weight_rebalanced_transaction_costs",
        ),
        "weights": pd.DataFrame(weights, index=returns.index, columns=returns.columns),
    }


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
