"""Portfolio evaluation metrics.

This module contains shared return-based metrics for benchmark policies and
future TD3 evaluations. It does not execute policies or interact with the
environment yet.
"""

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Compute compounded portfolio equity from periodic returns."""
    curve = initial_value * (1.0 + returns).cumprod()
    curve.name = "equity_curve"
    return curve


def cumulative_return(returns: pd.Series) -> float:
    """Compute total compounded return over the full series."""
    return float((1.0 + returns).prod() - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = 52) -> float:
    """Compute compounded annualized return."""
    _validate_non_empty_returns(returns)
    total_return = cumulative_return(returns)
    n_periods = len(returns)

    return float((1.0 + total_return) ** (periods_per_year / n_periods) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 52) -> float:
    """Compute annualized sample volatility."""
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """Compute annualized Sharpe ratio using an annual risk-free rate."""
    rf_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = returns - rf_period
    excess_volatility = excess_returns.std()

    if excess_volatility == 0.0 or np.isnan(excess_volatility):
        return 0.0

    return float(excess_returns.mean() / excess_volatility * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown as the minimum drawdown value."""
    curve = equity_curve(returns)
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0

    return float(drawdown.min())


def summary_metrics(
    returns: pd.Series,
    periods_per_year: int = 52,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute a compact dictionary of standard portfolio metrics."""
    return {
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
    }


def _validate_non_empty_returns(returns: pd.Series) -> None:
    if returns.empty:
        raise ValueError("returns must not be empty.")
