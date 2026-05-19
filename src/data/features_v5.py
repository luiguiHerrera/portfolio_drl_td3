"""Feature Set V5 with return-derived regime and correlation features.

V5 is additive and opt-in. It starts from Feature Set V2 and appends
interpretable market-regime and dynamic-correlation features built only from
historical returns. It does not shift features internally; dataset preparation
remains responsible for the one-period anti-leakage shift.
"""

from itertools import combinations
from numbers import Real

import numpy as np
import pandas as pd

from src.data.features_v2 import build_features_v2, rolling_correlation, rolling_drawdown


def build_features_v5(
    returns: pd.DataFrame,
    market_asset: str = "SPY",
    short_window: int = 4,
    long_window: int = 12,
    ewma_span: int = 12,
    correlation_window: int = 12,
    drawdown_window: int = 12,
    risk_off_threshold: float = 2.0,
) -> pd.DataFrame:
    """Build V2 features plus regime and correlation state features."""
    _validate_inputs(
        returns=returns,
        market_asset=market_asset,
        short_window=short_window,
        long_window=long_window,
        ewma_span=ewma_span,
        correlation_window=correlation_window,
        drawdown_window=drawdown_window,
        risk_off_threshold=risk_off_threshold,
    )
    v2_features = build_features_v2(
        returns,
        market_asset=market_asset,
        short_window=short_window,
        long_window=long_window,
        ewma_span=ewma_span,
    )
    regime_features = _build_regime_correlation_features(
        returns=returns,
        market_asset=market_asset,
        short_window=short_window,
        long_window=long_window,
        correlation_window=correlation_window,
        drawdown_window=drawdown_window,
        risk_off_threshold=risk_off_threshold,
    )
    features = pd.concat([v2_features, regime_features], axis=1).dropna()
    if features.empty:
        raise ValueError(
            "Feature Set V5 output is empty after aligning V2 and regime features."
        )

    return features


def _build_regime_correlation_features(
    returns: pd.DataFrame,
    market_asset: str,
    short_window: int,
    long_window: int,
    correlation_window: int,
    drawdown_window: int,
    risk_off_threshold: float,
) -> pd.DataFrame:
    features = pd.DataFrame(index=returns.index)
    market_returns = returns[market_asset]
    market_momentum = _rolling_compounded_return(market_returns, long_window)
    market_drawdown = rolling_drawdown(market_returns, drawdown_window)
    short_volatility = market_returns.rolling(short_window).std()
    long_volatility = market_returns.rolling(long_window).std()

    features[f"regime_market_momentum_{long_window}p"] = market_momentum
    features["regime_market_trend_positive"] = _float_indicator(
        market_momentum > 0.0,
        market_momentum,
    )
    features["regime_market_trend_negative"] = _float_indicator(
        market_momentum < 0.0,
        market_momentum,
    )
    features[f"regime_market_rolling_drawdown_{drawdown_window}p"] = market_drawdown
    features["regime_market_drawdown_stress"] = _float_indicator(
        market_drawdown <= -0.10,
        market_drawdown,
    )
    features[f"regime_market_vol_{short_window}p"] = short_volatility
    features[f"regime_market_vol_{long_window}p"] = long_volatility
    volatility_availability = short_volatility.mask(long_volatility.isna())
    features["regime_market_high_vol"] = _float_indicator(
        short_volatility > long_volatility,
        volatility_availability,
    )

    risky_assets = _risky_assets(returns)
    asset_market_correlations = {}
    for asset in risky_assets:
        if asset == market_asset:
            continue
        column = f"corr_{asset}_vs_{market_asset}_{correlation_window}p"
        asset_market_correlations[asset] = rolling_correlation(
            returns[asset],
            market_returns,
            correlation_window,
        )
        features[column] = asset_market_correlations[asset]

    pairwise_correlations = []
    for first_asset, second_asset in combinations(risky_assets, 2):
        pairwise_correlations.append(
            rolling_correlation(
                returns[first_asset],
                returns[second_asset],
                correlation_window,
            )
        )
    avg_pairwise_corr = pd.concat(pairwise_correlations, axis=1).mean(axis=1)
    features[f"avg_pairwise_corr_{correlation_window}p"] = avg_pairwise_corr
    features["correlation_stress"] = _float_indicator(
        avg_pairwise_corr > 0.50,
        avg_pairwise_corr,
    )
    features["diversification_benefit_score"] = 1.0 - avg_pairwise_corr

    if "TLT" in asset_market_correlations:
        tlt_corr = asset_market_correlations["TLT"]
        features["tlt_equity_hedge_signal"] = _float_indicator(tlt_corr < 0.0, tlt_corr)
    if "GLD" in asset_market_correlations:
        gld_corr = asset_market_correlations["GLD"]
        features["gld_equity_hedge_signal"] = _float_indicator(gld_corr < 0.0, gld_corr)

    risk_off_components = features[
        [
            "regime_market_drawdown_stress",
            "regime_market_high_vol",
            "correlation_stress",
            "regime_market_trend_negative",
        ]
    ]
    features["risk_off_score"] = risk_off_components.sum(axis=1)
    features["risk_off_score"] = features["risk_off_score"].mask(
        risk_off_components.isna().any(axis=1)
    )
    features["risk_off_state"] = _float_indicator(
        features["risk_off_score"] >= risk_off_threshold,
        features["risk_off_score"],
    )

    return features


def _rolling_compounded_return(returns: pd.Series, window: int) -> pd.Series:
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def _float_indicator(condition: pd.Series, availability: pd.Series) -> pd.Series:
    indicator = condition.astype(float)

    return indicator.mask(availability.isna())


def _risky_assets(returns: pd.DataFrame) -> list[str]:
    return [asset for asset in returns.columns if asset != "CASH"]


def _validate_inputs(
    returns: pd.DataFrame,
    market_asset: str,
    short_window: int,
    long_window: int,
    ewma_span: int,
    correlation_window: int,
    drawdown_window: int,
    risk_off_threshold: float,
) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a pandas DatetimeIndex.")
    if not isinstance(market_asset, str) or not market_asset.strip():
        raise ValueError("market_asset must be a non-empty string.")
    if market_asset not in returns.columns:
        raise ValueError(f"market_asset '{market_asset}' must exist in returns columns.")

    _validate_window(short_window, "short_window")
    _validate_window(long_window, "long_window")
    _validate_window(ewma_span, "ewma_span")
    _validate_window(correlation_window, "correlation_window")
    _validate_window(drawdown_window, "drawdown_window")
    if long_window < short_window:
        raise ValueError("long_window must be greater than or equal to short_window.")
    if isinstance(risk_off_threshold, bool) or not isinstance(risk_off_threshold, Real):
        raise ValueError("risk_off_threshold must be a numeric, non-boolean value.")
    if risk_off_threshold < 0.0:
        raise ValueError("risk_off_threshold must be greater than or equal to 0.")
    if len(_risky_assets(returns)) < 2:
        raise ValueError(
            "At least two risky assets excluding CASH are required for pairwise "
            "correlation features."
        )


def _validate_window(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer greater than or equal to 2.")
    if value < 2:
        raise ValueError(f"{name} must be an integer greater than or equal to 2.")
