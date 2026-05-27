"""Feature Set V7: real macro plus rolling fitted GARCH features.

V7 combines the current-window V3 macro feature block with the V4 rolling
fitted real-GARCH volatility block. It does not shift features internally; the
dataset pipeline remains responsible for the project-wide one-period
anti-leakage shift.
"""

from __future__ import annotations

import pandas as pd

from src.data.features_v3 import build_features_v3
from src.data.garch_features import (
    GARCH_FALLBACK_ROLLING_REALIZED,
    GARCH_MODE_ROLLING_FITTED,
    build_garch_feature_set_by_mode,
)


def build_features_v7(
    returns: pd.DataFrame,
    macro_data: pd.DataFrame,
    market_asset: str = "SPY",
    short_window: int = 4,
    long_window: int = 12,
    ewma_span: int = 12,
    garch_include_relative: bool = True,
    garch_periods_per_year: int = 52,
    garch_mode: str = GARCH_MODE_ROLLING_FITTED,
    garch_min_history: int = 104,
    garch_window: int | None = 156,
    garch_annualize: bool = False,
    garch_exclude_cash: bool = True,
    garch_fallback: str = GARCH_FALLBACK_ROLLING_REALIZED,
    return_garch_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build V3 macro features plus rolling fitted GARCH features."""
    if macro_data is None:
        raise ValueError("V7 real macro/GARCH features require macro_data.")

    macro_features = build_features_v3(
        returns=returns,
        macro_data=macro_data,
        market_asset=market_asset,
        short_window=short_window,
        long_window=long_window,
        ewma_span=ewma_span,
    )
    garch_features, diagnostics = build_garch_feature_set_by_mode(
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
    features = pd.concat([macro_features, garch_features], axis=1, sort=False).dropna()
    if features.empty:
        raise ValueError(
            "Feature Set V7 output is empty after aligning macro and GARCH features."
        )
    if return_garch_diagnostics:
        return features, diagnostics
    return features
