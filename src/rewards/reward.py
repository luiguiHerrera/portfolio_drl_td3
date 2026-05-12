"""Reward functions for portfolio allocation.

This module keeps the minimal net-return reward baseline for backward
compatibility and provides a configurable risk-aware reward used by the
portfolio environment.
"""

import numpy as np


def compute_net_return_reward(portfolio_return: float, transaction_cost: float) -> float:
    """Return portfolio return net of realized transaction cost."""
    return portfolio_return - transaction_cost


def concentration_penalty(weights: np.ndarray) -> float:
    """Return Herfindahl concentration for long-only, fully invested weights."""
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.ndim != 1:
        raise ValueError("weights must be a one-dimensional array.")
    if np.any(weights_array < 0.0):
        raise ValueError("weights must be non-negative.")
    if not np.isclose(weights_array.sum(), 1.0):
        raise ValueError("weights must sum to 1.")

    return float(np.sum(weights_array ** 2))


def drawdown_penalty(portfolio_value: float, peak_portfolio_value: float) -> float:
    """Return current drawdown from the prior peak portfolio value."""
    if not _is_positive_number(portfolio_value):
        raise ValueError("portfolio_value must be positive.")
    if not _is_positive_number(peak_portfolio_value):
        raise ValueError("peak_portfolio_value must be positive.")

    return max(0.0, 1.0 - portfolio_value / peak_portfolio_value)


def compute_risk_aware_reward(
    portfolio_return: float,
    transaction_cost: float,
    turnover: float,
    weights: np.ndarray,
    portfolio_value: float,
    peak_portfolio_value: float,
    reward_config: dict,
) -> float:
    """Compute portfolio reward with configurable risk penalties."""
    if not isinstance(reward_config, dict):
        raise ValueError("reward_config must be a dict.")
    if not _is_number(portfolio_return):
        raise ValueError("portfolio_return must be numeric.")
    if not _is_number(transaction_cost) or transaction_cost < 0.0:
        raise ValueError("transaction_cost must be non-negative.")
    if not _is_number(turnover) or turnover < 0.0:
        raise ValueError("turnover must be non-negative.")

    lambda_return = _get_lambda(reward_config, "lambda_return", 1.0)
    lambda_transaction_cost = _get_lambda(
        reward_config,
        "lambda_transaction_cost",
        1.0,
    )
    lambda_turnover = _get_lambda(reward_config, "lambda_turnover", 0.0)
    lambda_concentration = _get_lambda(reward_config, "lambda_concentration", 0.0)
    lambda_drawdown = _get_lambda(reward_config, "lambda_drawdown", 0.0)

    concentration = concentration_penalty(weights)
    drawdown = drawdown_penalty(portfolio_value, peak_portfolio_value)

    return float(
        lambda_return * portfolio_return
        - lambda_transaction_cost * transaction_cost
        - lambda_turnover * turnover
        - lambda_concentration * concentration
        - lambda_drawdown * drawdown
    )


def _get_lambda(reward_config: dict, field_name: str, default: float) -> float:
    value = reward_config.get(field_name, default)
    if not _is_number(value) or value < 0.0:
        raise ValueError(f"Reward config field {field_name} must be numeric and non-negative.")
    return float(value)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_positive_number(value) -> bool:
    return _is_number(value) and value > 0.0
