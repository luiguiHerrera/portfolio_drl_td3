"""Extended portfolio performance metrics for analysis and reporting."""

import numpy as np
import pandas as pd

from src.backtest.evaluate_policy import annualized_return, max_drawdown


def downside_deviation(
    returns: pd.Series,
    minimum_acceptable_return: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """Compute annualized downside deviation versus an annual MAR."""
    _validate_non_empty_series(returns, "returns")
    _validate_number(minimum_acceptable_return, "minimum_acceptable_return")
    _validate_periods_per_year(periods_per_year)
    mar_periodic = minimum_acceptable_return / periods_per_year
    downside_returns = np.minimum(returns - mar_periodic, 0.0)
    deviation = float(np.sqrt(np.mean(np.square(downside_returns))) * np.sqrt(periods_per_year))

    if deviation == 0.0:
        return 0.0

    return deviation


def sortino_ratio(
    returns: pd.Series,
    minimum_acceptable_return: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """Compute Sortino ratio with annualized return over annualized downside deviation."""
    _validate_number(minimum_acceptable_return, "minimum_acceptable_return")
    _validate_periods_per_year(periods_per_year)
    numerator = annualized_return(returns, periods_per_year) - minimum_acceptable_return
    denominator = downside_deviation(
        returns,
        minimum_acceptable_return=minimum_acceptable_return,
        periods_per_year=periods_per_year,
    )

    return _ratio_with_zero_denominator(numerator, denominator)


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 52,
) -> float:
    """Compute Calmar ratio as annualized return over absolute max drawdown."""
    _validate_periods_per_year(periods_per_year)
    numerator = annualized_return(returns, periods_per_year)
    denominator = abs(max_drawdown(returns))

    return _ratio_with_zero_denominator(numerator, denominator)


def align_return_series(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align two return series by shared index and drop rows with missing values."""
    _validate_non_empty_series(strategy_returns, "strategy_returns")
    _validate_non_empty_series(benchmark_returns, "benchmark_returns")

    shared_index = strategy_returns.index.intersection(benchmark_returns.index)
    if shared_index.empty:
        raise ValueError("return series must have at least one shared index value.")

    aligned = pd.concat(
        [
            strategy_returns.loc[shared_index].rename("strategy"),
            benchmark_returns.loc[shared_index].rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        raise ValueError("aligned return series must not be empty after dropping NaNs.")

    return aligned["strategy"], aligned["benchmark"]


def tracking_error(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 52,
) -> float:
    """Compute annualized sample standard deviation of active returns."""
    _validate_periods_per_year(periods_per_year)
    strategy_aligned, benchmark_aligned = align_return_series(strategy_returns, benchmark_returns)
    if len(strategy_aligned) < 2:
        raise ValueError("tracking error requires at least two aligned observations.")

    active_returns = strategy_aligned - benchmark_aligned
    return float(active_returns.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 52,
) -> float:
    """Compute information ratio versus a benchmark return series."""
    _validate_periods_per_year(periods_per_year)
    strategy_aligned, benchmark_aligned = align_return_series(strategy_returns, benchmark_returns)
    numerator = annualized_return(strategy_aligned, periods_per_year) - annualized_return(
        benchmark_aligned,
        periods_per_year,
    )
    denominator = tracking_error(
        strategy_aligned,
        benchmark_aligned,
        periods_per_year=periods_per_year,
    )

    return _ratio_with_zero_denominator(numerator, denominator)


def capm_beta_alpha(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52,
) -> dict:
    """Compute CAPM beta and annualized alpha from periodic returns."""
    _validate_number(risk_free_rate, "risk_free_rate")
    _validate_periods_per_year(periods_per_year)
    strategy_aligned, market_aligned = align_return_series(strategy_returns, market_returns)
    if len(strategy_aligned) < 2:
        raise ValueError("CAPM beta and alpha require at least two aligned observations.")

    periodic_rf = risk_free_rate / periods_per_year
    strategy_excess = strategy_aligned - periodic_rf
    market_excess = market_aligned - periodic_rf
    market_variance = market_excess.var(ddof=1)
    if market_variance == 0.0 or np.isnan(market_variance):
        raise ValueError("market excess return variance must be non-zero.")

    beta = strategy_excess.cov(market_excess) / market_variance
    alpha_periodic = strategy_excess.mean() - beta * market_excess.mean()

    return {
        "capm_beta": float(beta),
        "capm_alpha": float(alpha_periodic * periods_per_year),
    }


def extended_summary_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    market_returns: pd.Series | None = None,
    periods_per_year: int = 52,
    risk_free_rate: float = 0.0,
    minimum_acceptable_return: float = 0.0,
) -> dict:
    """Compute extended metrics without duplicating base summary metrics."""
    _validate_periods_per_year(periods_per_year)
    _validate_number(risk_free_rate, "risk_free_rate")
    _validate_number(minimum_acceptable_return, "minimum_acceptable_return")
    metrics = {
        "sortino_ratio": sortino_ratio(
            returns,
            minimum_acceptable_return=minimum_acceptable_return,
            periods_per_year=periods_per_year,
        ),
        "calmar_ratio": calmar_ratio(returns, periods_per_year=periods_per_year),
    }

    if benchmark_returns is not None:
        metrics["tracking_error"] = tracking_error(
            returns,
            benchmark_returns,
            periods_per_year=periods_per_year,
        )
        metrics["information_ratio"] = information_ratio(
            returns,
            benchmark_returns,
            periods_per_year=periods_per_year,
        )

    if market_returns is not None:
        metrics.update(
            capm_beta_alpha(
                returns,
                market_returns,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
        )

    return metrics


def _validate_non_empty_series(series: pd.Series, name: str) -> None:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if series.empty:
        raise ValueError(f"{name} must not be empty.")


def _validate_periods_per_year(periods_per_year: int) -> None:
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int):
        raise TypeError("periods_per_year must be a non-bool integer.")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be at least 1.")


def _validate_number(value: int | float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a non-bool number.")


def _ratio_with_zero_denominator(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        if numerator > 0.0:
            return float("inf")
        if numerator < 0.0:
            return float("-inf")
        return 0.0

    return float(numerator / denominator)
