"""Feature Set V4 with GARCH-style volatility features.

V4 is additive and opt-in. It starts from Feature Set V2 and optionally appends
GARCH-style volatility features. Existing deterministic fixed-parameter filter
behavior is preserved by default. The `rolling_fitted` mode estimates GARCH
parameters with information available only through t-1. This module does not
shift features internally; dataset preparation remains responsible for the
one-period anti-leakage shift.
"""

import pandas as pd

from src.data.features_v2 import build_features_v2
from src.data.garch_features import (
    GARCH_FALLBACK_ROLLING_REALIZED,
    GARCH_MODE_DETERMINISTIC,
    build_garch_feature_set_by_mode,
)


def build_features_v4(
    returns: pd.DataFrame,
    market_asset: str = "SPY",
    short_window: int = 4,
    long_window: int = 12,
    ewma_span: int = 12,
    include_garch_features: bool = True,
    garch_include_relative: bool = True,
    garch_omega: float = 1e-6,
    garch_alpha: float = 0.05,
    garch_beta: float = 0.90,
    garch_periods_per_year: int = 52,
    garch_mode: str = GARCH_MODE_DETERMINISTIC,
    garch_min_history: int = 104,
    garch_window: int | None = 156,
    garch_annualize: bool = False,
    garch_exclude_cash: bool = False,
    garch_fallback: str = GARCH_FALLBACK_ROLLING_REALIZED,
) -> pd.DataFrame:
    """Build V2 features plus optional GARCH-style volatility features."""
    v2_features = build_features_v2(
        returns,
        market_asset=market_asset,
        short_window=short_window,
        long_window=long_window,
        ewma_span=ewma_span,
    )
    if not include_garch_features:
        return v2_features.copy()

    garch_features = build_garch_feature_set_by_mode(
        returns=returns,
        assets=list(returns.columns),
        market_asset=market_asset,
        include_relative=garch_include_relative,
        mode=garch_mode,
        omega=garch_omega,
        alpha=garch_alpha,
        beta=garch_beta,
        periods_per_year=garch_periods_per_year,
        min_history=garch_min_history,
        window=garch_window,
        annualize=garch_annualize,
        exclude_cash=garch_exclude_cash,
        fallback=garch_fallback,
    )
    features = pd.concat([v2_features, garch_features], axis=1, sort=False).dropna()
    if features.empty:
        raise ValueError(
            "Feature Set V4 output is empty after aligning V2 and GARCH features."
        )

    return features
