"""Deterministic GARCH-style volatility features.

This module provides a simple GARCH(1,1)-style conditional volatility filter
that can be used as an opt-in feature source later. It is not a fitted
maximum-likelihood GARCH model.
"""

import math
from numbers import Real

import numpy as np
import pandas as pd


def validate_garch_parameters(
    omega: float,
    alpha: float,
    beta: float,
    periods_per_year: int,
) -> None:
    """Validate deterministic GARCH filter parameters."""
    _validate_numeric(omega, "omega")
    _validate_numeric(alpha, "alpha")
    _validate_numeric(beta, "beta")
    if not isinstance(periods_per_year, int) or isinstance(periods_per_year, bool):
        raise ValueError("periods_per_year must be a positive integer.")

    if omega <= 0.0:
        raise ValueError("omega must be greater than 0.")
    if alpha < 0.0:
        raise ValueError("alpha must be greater than or equal to 0.")
    if beta < 0.0:
        raise ValueError("beta must be greater than or equal to 0.")
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be less than 1.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")


def compute_garch_volatility_series(
    returns: pd.Series,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.90,
    periods_per_year: int = 52,
    annualize: bool = True,
) -> pd.Series:
    """Compute a deterministic GARCH(1,1)-style volatility series.

    The recursion uses lagged returns only:
    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}.
    """
    validate_garch_parameters(omega, alpha, beta, periods_per_year)
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    numeric_returns = pd.to_numeric(returns, errors="coerce")
    if numeric_returns.isna().any():
        raise ValueError("returns must not contain missing or non-numeric values.")

    sigma2 = np.empty(len(numeric_returns), dtype=float)
    initial_sigma2 = omega / (1.0 - alpha - beta)
    if len(numeric_returns) > 0:
        sigma2[0] = initial_sigma2
    for index in range(1, len(numeric_returns)):
        lagged_return = float(numeric_returns.iloc[index - 1])
        sigma2[index] = omega + alpha * lagged_return**2 + beta * sigma2[index - 1]

    volatility = np.sqrt(sigma2)
    if annualize:
        volatility = volatility * math.sqrt(periods_per_year)

    name = f"{returns.name}_garch_vol" if returns.name is not None else "garch_vol"

    return pd.Series(volatility, index=returns.index, name=name)


def build_garch_volatility_features(
    returns: pd.DataFrame,
    assets: list[str] | None = None,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.90,
    periods_per_year: int = 52,
    annualize: bool = True,
    prefix: str = "garch_vol",
) -> pd.DataFrame:
    """Build absolute deterministic GARCH volatility features for assets."""
    _validate_returns_dataframe(returns)
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string.")

    selected_assets = list(returns.columns) if assets is None else list(assets)
    missing_assets = [asset for asset in selected_assets if asset not in returns.columns]
    if missing_assets:
        raise ValueError(f"Requested assets are missing from returns: {missing_assets}.")

    features = pd.DataFrame(index=returns.index)
    for asset in selected_assets:
        volatility = compute_garch_volatility_series(
            returns[asset],
            omega=omega,
            alpha=alpha,
            beta=beta,
            periods_per_year=periods_per_year,
            annualize=annualize,
        )
        features[f"{prefix}_{asset}"] = volatility

    return features


def build_garch_relative_features(
    garch_vol_features: pd.DataFrame,
    market_asset: str = "SPY",
    prefix: str = "garch",
) -> pd.DataFrame:
    """Build cross-sectional GARCH volatility ratios and ranks."""
    if not isinstance(garch_vol_features, pd.DataFrame):
        raise TypeError("garch_vol_features must be a pandas DataFrame.")
    if garch_vol_features.empty:
        raise ValueError("garch_vol_features must be a non-empty DataFrame.")
    if not isinstance(market_asset, str) or not market_asset.strip():
        raise ValueError("market_asset must be a non-empty string.")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string.")

    absolute_prefix = f"{prefix}_vol_"
    volatility_columns = [
        column
        for column in garch_vol_features.columns
        if isinstance(column, str) and column.startswith(absolute_prefix)
    ]
    if not volatility_columns:
        raise ValueError(f"No columns found with prefix '{absolute_prefix}'.")

    market_column = f"{absolute_prefix}{market_asset}"
    if market_column not in garch_vol_features.columns:
        raise ValueError(f"Market volatility column '{market_column}' is missing.")

    volatility = garch_vol_features[volatility_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if volatility.isna().any().any():
        raise ValueError("garch_vol_features must contain only numeric values.")
    if (volatility[market_column] <= 0.0).any():
        raise ValueError("market volatility must be strictly positive.")

    relative = pd.DataFrame(index=garch_vol_features.index)
    market_volatility = volatility[market_column]
    ranks = volatility.rank(axis=1, method="average", ascending=True)
    for column in volatility_columns:
        asset = column.removeprefix(absolute_prefix)
        relative[f"{prefix}_vol_ratio_{asset}_vs_{market_asset}"] = (
            volatility[column] / market_volatility
        )
        relative[f"{prefix}_vol_rank_{asset}"] = ranks[column]

    return relative


def build_garch_feature_set(
    returns: pd.DataFrame,
    assets: list[str] | None = None,
    market_asset: str = "SPY",
    include_relative: bool = True,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.90,
    periods_per_year: int = 52,
) -> pd.DataFrame:
    """Build absolute and optional relative GARCH-style volatility features."""
    absolute_features = build_garch_volatility_features(
        returns=returns,
        assets=assets,
        omega=omega,
        alpha=alpha,
        beta=beta,
        periods_per_year=periods_per_year,
        annualize=True,
        prefix="garch_vol",
    )
    if not include_relative:
        return absolute_features

    relative_features = build_garch_relative_features(
        absolute_features,
        market_asset=market_asset,
        prefix="garch",
    )

    return pd.concat([absolute_features, relative_features], axis=1)


def _validate_returns_dataframe(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a pandas DatetimeIndex.")


def _validate_numeric(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a numeric, non-boolean value.")
