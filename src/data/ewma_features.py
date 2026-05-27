"""Lagged EWMA volatility features.

The EWMA volatility forecast assigned to date t uses returns observed only
through t-1:

variance_t = lambda * variance_{t-1} + (1 - lambda) * r_{t-1}^2

The output volatility unit is weekly by default. CASH is excluded by default
because zero-return synthetic cash has degenerate volatility behavior.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def compute_lagged_ewma_volatility_series(
    returns: pd.Series,
    ewma_lambda: float = 0.94,
    min_vol: float = 1e-8,
    annualize: bool = False,
    periods_per_year: int = 52,
) -> pd.Series:
    """Compute one-step-ahead EWMA volatility from lagged returns only."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    _validate_ewma_options(ewma_lambda, min_vol, periods_per_year)

    numeric_returns = pd.to_numeric(returns, errors="coerce")
    if numeric_returns.isna().any():
        raise ValueError("returns must not contain missing or non-numeric values.")

    variance = np.empty(len(numeric_returns), dtype=float)
    variance[:] = np.nan
    initial_variance = _initial_variance(numeric_returns, min_vol)
    if len(numeric_returns) > 0:
        variance[0] = initial_variance
    for position in range(1, len(numeric_returns)):
        lagged_return = float(numeric_returns.iloc[position - 1])
        previous_variance = variance[position - 1]
        variance[position] = (
            ewma_lambda * previous_variance
            + (1.0 - ewma_lambda) * lagged_return**2
        )
        variance[position] = max(float(variance[position]), min_vol**2)

    volatility = np.sqrt(variance)
    if annualize:
        volatility = volatility * math.sqrt(periods_per_year)
    name = f"{returns.name}_ewma_vol" if returns.name is not None else "ewma_vol"
    return pd.Series(volatility, index=numeric_returns.index, name=name)


def build_ewma_volatility_features(
    returns: pd.DataFrame,
    assets: list[str] | None = None,
    ewma_lambda: float = 0.94,
    min_vol: float = 1e-8,
    annualize: bool = False,
    periods_per_year: int = 52,
    exclude_cash: bool = True,
    prefix: str = "ewma_vol",
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build lagged EWMA volatility forecasts for selected assets."""
    _validate_returns_dataframe(returns)
    _validate_ewma_options(ewma_lambda, min_vol, periods_per_year)
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string.")

    selected_assets = _select_ewma_assets(returns, assets=assets, exclude_cash=exclude_cash)
    features = pd.DataFrame(index=returns.index)
    diagnostics: list[dict[str, Any]] = []
    for asset in selected_assets:
        series = compute_lagged_ewma_volatility_series(
            returns[asset],
            ewma_lambda=ewma_lambda,
            min_vol=min_vol,
            annualize=annualize,
            periods_per_year=periods_per_year,
        )
        features[f"{prefix}_{asset}"] = series
        diagnostics.append(
            {
                "asset": asset,
                "ewma_lambda": ewma_lambda,
                "status": "computed",
                "exclude_cash": exclude_cash,
                "volatility_unit": "annualized" if annualize else "weekly",
                "n_rows": len(series),
                "missing_count": int(series.isna().sum()),
                "first_date": series.index.min(),
                "last_date": series.index.max(),
            }
        )

    diagnostics_frame = pd.DataFrame(diagnostics)
    if return_diagnostics:
        return features, diagnostics_frame
    return features


def _initial_variance(returns: pd.Series, min_vol: float) -> float:
    # At the first timestamp there is no prior return history. Use a small
    # deterministic floor rather than peeking into the sample.
    return min_vol**2


def _validate_returns_dataframe(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a pandas DatetimeIndex.")


def _validate_ewma_options(
    ewma_lambda: float,
    min_vol: float,
    periods_per_year: int,
) -> None:
    if not isinstance(ewma_lambda, (int, float)) or isinstance(ewma_lambda, bool):
        raise ValueError("ewma_lambda must be numeric.")
    if ewma_lambda <= 0.0 or ewma_lambda >= 1.0:
        raise ValueError("ewma_lambda must be in the range (0, 1).")
    if not isinstance(min_vol, (int, float)) or isinstance(min_vol, bool):
        raise ValueError("min_vol must be numeric.")
    if min_vol <= 0.0:
        raise ValueError("min_vol must be greater than 0.")
    if not isinstance(periods_per_year, int) or isinstance(periods_per_year, bool):
        raise ValueError("periods_per_year must be a positive integer.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")


def _select_ewma_assets(
    returns: pd.DataFrame,
    assets: list[str] | None,
    exclude_cash: bool,
) -> list[str]:
    selected_assets = list(returns.columns) if assets is None else list(assets)
    missing_assets = [asset for asset in selected_assets if asset not in returns.columns]
    if missing_assets:
        raise ValueError(f"Requested assets are missing from returns: {missing_assets}.")
    if exclude_cash:
        selected_assets = [asset for asset in selected_assets if asset != "CASH"]
    if not selected_assets:
        raise ValueError("At least one non-CASH asset is required for EWMA features.")
    return selected_assets
