"""Feature Set V8: hybrid EWMA plus rolling fitted GARCH volatility state.

V8 is a volatility-focused candidate. It starts from V2 base features, adds the
V4 rolling fitted real-GARCH volatility block, adds lagged EWMA volatility
forecasts, and includes simple comparison features between the two conditional
volatility estimates. It does not include macro, dynamic correlation,
cointegration, or sequence-model features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.ewma_features import build_ewma_volatility_features
from src.data.features_v2 import build_features_v2
from src.data.garch_features import (
    GARCH_FALLBACK_ROLLING_REALIZED,
    GARCH_MODE_ROLLING_FITTED,
    build_garch_feature_set_by_mode,
)


def build_features_v8(
    returns: pd.DataFrame,
    market_asset: str = "SPY",
    short_window: int = 4,
    long_window: int = 12,
    ewma_span: int = 12,
    ewma_lambda: float = 0.94,
    volatility_min_vol: float = 1e-8,
    comparison_epsilon: float = 1e-8,
    garch_include_relative: bool = True,
    garch_periods_per_year: int = 52,
    garch_mode: str = GARCH_MODE_ROLLING_FITTED,
    garch_min_history: int = 104,
    garch_window: int | None = 156,
    garch_annualize: bool = False,
    garch_exclude_cash: bool = True,
    garch_fallback: str = GARCH_FALLBACK_ROLLING_REALIZED,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build V8 volatility-focused features."""
    v2_features = build_features_v2(
        returns,
        market_asset=market_asset,
        short_window=short_window,
        long_window=long_window,
        ewma_span=ewma_span,
    )
    garch_features, garch_diagnostics = build_garch_feature_set_by_mode(
        returns=returns,
        assets=list(returns.columns),
        market_asset=market_asset,
        include_relative=garch_include_relative,
        mode=garch_mode,
        periods_per_year=garch_periods_per_year,
        min_history=garch_min_history,
        window=garch_window,
        annualize=garch_annualize,
        exclude_cash=garch_exclude_cash,
        fallback=garch_fallback,
        return_diagnostics=True,
    )
    ewma_features, ewma_diagnostics = build_ewma_volatility_features(
        returns=returns,
        assets=list(returns.columns),
        ewma_lambda=ewma_lambda,
        min_vol=volatility_min_vol,
        annualize=garch_annualize,
        periods_per_year=garch_periods_per_year,
        exclude_cash=garch_exclude_cash,
        prefix="ewma_vol",
        return_diagnostics=True,
    )
    comparison_features = build_garch_ewma_comparison_features(
        garch_features=garch_features,
        ewma_features=ewma_features,
        epsilon=comparison_epsilon,
    )
    features = pd.concat(
        [v2_features, garch_features, ewma_features, comparison_features],
        axis=1,
        sort=False,
    ).dropna()
    if features.empty:
        raise ValueError(
            "Feature Set V8 output is empty after aligning V2, GARCH, and EWMA features."
        )
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature Set V8 output contains non-finite values.")

    if return_diagnostics:
        return features, {
            "garch": garch_diagnostics,
            "ewma": ewma_diagnostics,
        }
    return features


def build_garch_ewma_comparison_features(
    garch_features: pd.DataFrame,
    ewma_features: pd.DataFrame,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Build per-asset GARCH minus EWMA and GARCH/EWMA ratio features."""
    if not isinstance(garch_features, pd.DataFrame):
        raise TypeError("garch_features must be a pandas DataFrame.")
    if not isinstance(ewma_features, pd.DataFrame):
        raise TypeError("ewma_features must be a pandas DataFrame.")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon <= 0.0:
        raise ValueError("epsilon must be a positive number.")

    garch_vol_columns = [
        column
        for column in garch_features.columns
        if isinstance(column, str) and column.startswith("garch_vol_")
        and "_vs_" not in column
        and not column.startswith("garch_vol_ratio_")
        and not column.startswith("garch_vol_rank_")
    ]
    comparison = pd.DataFrame(index=garch_features.index)
    for garch_column in garch_vol_columns:
        asset = garch_column.removeprefix("garch_vol_")
        ewma_column = f"ewma_vol_{asset}"
        if ewma_column not in ewma_features.columns:
            continue
        garch_vol = pd.to_numeric(garch_features[garch_column], errors="coerce")
        ewma_vol = pd.to_numeric(ewma_features[ewma_column], errors="coerce")
        denominator = ewma_vol.clip(lower=epsilon)
        comparison[f"garch_minus_ewma_vol_{asset}"] = garch_vol - ewma_vol
        comparison[f"garch_to_ewma_vol_ratio_{asset}"] = garch_vol / denominator

    if comparison.empty:
        raise ValueError("No matching GARCH/EWMA volatility columns were found.")
    finite_values = comparison.dropna()
    if not finite_values.empty and not np.isfinite(finite_values.to_numpy(dtype=float)).all():
        raise ValueError("GARCH/EWMA comparison features contain non-finite values.")
    return comparison
