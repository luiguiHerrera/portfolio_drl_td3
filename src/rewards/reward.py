"""Reward functions for portfolio allocation.

This module keeps the minimal net-return reward baseline for backward
compatibility and provides a configurable risk-aware reward used by the
portfolio environment.
"""

import numpy as np

REWARD_MODES = {"net_return_first", "component_legacy"}


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
    lambda_turnover = _get_lambda(reward_config, "lambda_turnover", 0.0)
    lambda_concentration = _get_lambda(reward_config, "lambda_concentration", 0.0)
    lambda_drawdown = _get_lambda(reward_config, "lambda_drawdown", 0.0)
    reward_mode = reward_config.get("reward_mode", "net_return_first")
    if not isinstance(reward_mode, str) or reward_mode not in REWARD_MODES:
        valid_modes = ", ".join(sorted(REWARD_MODES))
        raise ValueError(f"reward_mode must be one of: {valid_modes}.")

    concentration = concentration_penalty(weights)
    drawdown = drawdown_penalty(portfolio_value, peak_portfolio_value)
    turnover_penalty = compute_turnover_penalty(
        turnover=turnover,
        lambda_turnover=lambda_turnover,
        mode=reward_config.get("turnover_penalty_mode", "linear"),
        free_band=reward_config.get("turnover_free_band", 0.0),
        quadratic_weight=reward_config.get("turnover_quadratic_weight", 0.0),
    )

    if reward_mode == "component_legacy":
        lambda_transaction_cost = _get_lambda(
            reward_config,
            "lambda_transaction_cost",
            1.0,
        )
        base_reward = lambda_return * portfolio_return - lambda_transaction_cost * transaction_cost
    else:
        financial_net_return = portfolio_return - transaction_cost
        base_reward = lambda_return * financial_net_return

    return float(
        base_reward
        - turnover_penalty
        - lambda_concentration * concentration
        - lambda_drawdown * drawdown
    )


def compute_turnover_penalty(
    turnover: float,
    lambda_turnover: float,
    mode: str = "linear",
    free_band: float = 0.0,
    quadratic_weight: float = 0.0,
) -> float:
    """Compute the configured turnover penalty component."""
    if not _is_number(turnover) or turnover < 0.0:
        raise ValueError("turnover must be numeric and non-negative.")
    if not _is_number(lambda_turnover) or lambda_turnover < 0.0:
        raise ValueError("lambda_turnover must be numeric and non-negative.")
    if not isinstance(mode, str) or mode not in {
        "linear",
        "none",
        "excess_linear",
        "excess_quadratic",
    }:
        raise ValueError(
            "turnover_penalty_mode must be one of: linear, none, "
            "excess_linear, excess_quadratic."
        )
    if not _is_number(free_band) or free_band < 0.0:
        raise ValueError("turnover_free_band must be numeric and non-negative.")
    if not _is_number(quadratic_weight) or quadratic_weight < 0.0:
        raise ValueError(
            "turnover_quadratic_weight must be numeric and non-negative."
        )

    if mode == "none":
        return 0.0
    if mode == "linear":
        return float(lambda_turnover * turnover)

    turnover_excess = max(float(turnover) - float(free_band), 0.0)
    linear_penalty = float(lambda_turnover * turnover_excess)
    if mode == "excess_linear":
        return linear_penalty

    return float(linear_penalty + quadratic_weight * turnover_excess ** 2)


def _get_lambda(reward_config: dict, field_name: str, default: float) -> float:
    value = reward_config.get(field_name, default)
    if not _is_number(value) or value < 0.0:
        raise ValueError(f"Reward config field {field_name} must be numeric and non-negative.")
    return float(value)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_positive_number(value) -> bool:
    return _is_number(value) and value > 0.0
