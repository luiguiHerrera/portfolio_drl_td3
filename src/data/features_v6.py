"""Feature Set V6 for parsimonious financial state construction.

V6 is an opt-in return-derived state representation. It emphasizes
momentum/trend, interpretable risk-regime probabilities, volatility proxies,
and defensive-asset attractiveness. It uses only information contained in
returns observed through the feature date; dataset preparation remains
responsible for the external one-period anti-leakage shift.
"""

from itertools import combinations

import numpy as np
import pandas as pd


PROBABILITY_COLUMNS = (
    "p_market_trend_positive",
    "p_market_drawdown_stress",
    "p_market_high_vol",
    "p_correlation_stress",
    "p_risk_off",
    "cash_permission_score",
)


def build_features_v6(
    returns: pd.DataFrame,
    market_asset: str = "SPY",
    short_window: int = 4,
    medium_window: int = 12,
    long_window: int = 26,
    ewma_short_span: int = 4,
    ewma_long_span: int = 12,
    correlation_window: int = 12,
    zscore_window: int = 52,
) -> pd.DataFrame:
    """Build V6 financial-state features without internal time shifting."""
    _validate_inputs(
        returns=returns,
        market_asset=market_asset,
        short_window=short_window,
        medium_window=medium_window,
        long_window=long_window,
        ewma_short_span=ewma_short_span,
        ewma_long_span=ewma_long_span,
        correlation_window=correlation_window,
        zscore_window=zscore_window,
    )

    risky_assets = _risky_assets(returns)
    momentum_features, momentum_12w, risk_adjusted_momentum_12w = (
        _build_momentum_trend_block(
            returns=returns,
            risky_assets=risky_assets,
            short_window=short_window,
            medium_window=medium_window,
            long_window=long_window,
            ewma_long_span=ewma_long_span,
        )
    )
    regime_features = _build_risk_regime_probability_block(
        returns=returns,
        risky_assets=risky_assets,
        market_asset=market_asset,
        medium_window=medium_window,
        long_window=long_window,
        correlation_window=correlation_window,
        zscore_window=zscore_window,
    )
    volatility_features = _build_volatility_proxy_block(
        returns=returns,
        risky_assets=risky_assets,
        short_window=short_window,
        medium_window=medium_window,
        ewma_short_span=ewma_short_span,
        ewma_long_span=ewma_long_span,
    )
    defensive_features = _build_defensive_attractiveness_block(
        returns=returns,
        momentum_12w=momentum_12w,
        risk_adjusted_momentum_12w=risk_adjusted_momentum_12w,
        regime_features=regime_features,
        market_asset=market_asset,
    )

    features = pd.concat(
        [
            momentum_features,
            regime_features,
            volatility_features,
            defensive_features,
        ],
        axis=1,
        sort=False,
    )
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        raise ValueError(
            "Feature Set V6 output is empty after dropping rolling-window NaNs; "
            "provide more return observations or shorter windows."
        )

    probability_columns = [column for column in PROBABILITY_COLUMNS if column in features]
    features.loc[:, probability_columns] = features.loc[:, probability_columns].clip(
        lower=0.0,
        upper=1.0,
    )

    return features


def _build_momentum_trend_block(
    returns: pd.DataFrame,
    risky_assets: list[str],
    short_window: int,
    medium_window: int,
    long_window: int,
    ewma_long_span: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(index=returns.index)
    momentum_12w = pd.DataFrame(index=returns.index)
    risk_adjusted_momentum_12w = pd.DataFrame(index=returns.index)

    for asset in risky_assets:
        asset_returns = returns[asset]
        ret_short = _rolling_compounded_return(asset_returns, short_window)
        ret_medium = _rolling_compounded_return(asset_returns, medium_window)
        ret_long = _rolling_compounded_return(asset_returns, long_window)
        realized_vol_medium = _rolling_volatility(asset_returns, medium_window)
        realized_vol_long = _rolling_volatility(asset_returns, long_window)

        features[f"{asset}_ret_{short_window}w"] = ret_short
        features[f"{asset}_ret_{medium_window}w"] = ret_medium
        features[f"{asset}_ret_{long_window}w"] = ret_long
        features[f"{asset}_ewma_ret_{medium_window}w"] = asset_returns.ewm(
            span=ewma_long_span,
            adjust=False,
        ).mean()
        # Trend strength is cumulative return scaled by realized volatility,
        # which keeps the signal interpretable as risk-adjusted trend.
        features[f"{asset}_trend_strength_{medium_window}w"] = ret_medium / (
            realized_vol_medium.replace(0.0, np.nan)
        )
        features[f"{asset}_trend_strength_{long_window}w"] = ret_long / (
            realized_vol_long.replace(0.0, np.nan)
        )

        momentum_12w[asset] = ret_medium
        risk_adjusted_momentum_12w[asset] = features[
            f"{asset}_trend_strength_{medium_window}w"
        ]

    momentum_ranks = momentum_12w.rank(axis=1, method="average", pct=True)
    risk_adjusted_ranks = risk_adjusted_momentum_12w.rank(
        axis=1,
        method="average",
        pct=True,
    )
    momentum_available = momentum_12w.notna().all(axis=1)
    risk_adjusted_available = risk_adjusted_momentum_12w.notna().all(axis=1)
    momentum_winners = pd.Series(index=returns.index, dtype=object)
    risk_adjusted_winners = pd.Series(index=returns.index, dtype=object)
    momentum_winners.loc[momentum_available] = momentum_12w.loc[
        momentum_available
    ].idxmax(axis=1)
    risk_adjusted_winners.loc[risk_adjusted_available] = (
        risk_adjusted_momentum_12w.loc[risk_adjusted_available].idxmax(axis=1)
    )

    for asset in risky_assets:
        features[f"{asset}_momentum_rank_{medium_window}w"] = momentum_ranks[asset]
        features[f"{asset}_risk_adjusted_momentum_rank_{medium_window}w"] = (
            risk_adjusted_ranks[asset]
        )
        features[f"{asset}_winner_{medium_window}w_one_hot"] = (
            momentum_winners == asset
        ).astype(float).mask(~momentum_available)
        features[f"{asset}_winner_risk_adjusted_{medium_window}w_one_hot"] = (
            risk_adjusted_winners == asset
        ).astype(float).mask(~risk_adjusted_available)

    return features, momentum_12w, risk_adjusted_momentum_12w


def _build_risk_regime_probability_block(
    returns: pd.DataFrame,
    risky_assets: list[str],
    market_asset: str,
    medium_window: int,
    long_window: int,
    correlation_window: int,
    zscore_window: int,
) -> pd.DataFrame:
    features = pd.DataFrame(index=returns.index)
    market_returns = returns[market_asset]
    market_ret_medium = _rolling_compounded_return(market_returns, medium_window)
    market_ret_long = _rolling_compounded_return(market_returns, long_window)
    market_vol_medium = _rolling_volatility(market_returns, medium_window)
    market_ewma_vol = _ewma_volatility(market_returns, medium_window)
    market_drawdown = _rolling_drawdown(market_returns, long_window)
    avg_pairwise_corr = _average_pairwise_correlation(
        returns=returns,
        assets=risky_assets,
        window=correlation_window,
    )

    trend_distance = ((market_ret_medium + market_ret_long) / 2.0) / (
        market_vol_medium.replace(0.0, np.nan) * np.sqrt(medium_window)
    )
    volatility_zscore = _rolling_zscore(market_ewma_vol, zscore_window)
    correlation_zscore = _rolling_zscore(avg_pairwise_corr, zscore_window)

    features["p_market_trend_positive"] = _sigmoid(trend_distance)
    features["p_market_drawdown_stress"] = _sigmoid((-market_drawdown - 0.10) / 0.05)
    features["p_market_high_vol"] = _sigmoid(volatility_zscore)
    features["p_correlation_stress"] = _sigmoid(correlation_zscore)
    features["p_risk_off"] = (
        0.35 * (1.0 - features["p_market_trend_positive"])
        + 0.25 * features["p_market_drawdown_stress"]
        + 0.25 * features["p_market_high_vol"]
        + 0.15 * features["p_correlation_stress"]
    )

    return features.clip(lower=0.0, upper=1.0)


def _build_volatility_proxy_block(
    returns: pd.DataFrame,
    risky_assets: list[str],
    short_window: int,
    medium_window: int,
    ewma_short_span: int,
    ewma_long_span: int,
) -> pd.DataFrame:
    features = pd.DataFrame(index=returns.index)
    for asset in risky_assets:
        asset_returns = returns[asset]
        ewma_vol_short = _ewma_volatility(asset_returns, ewma_short_span)
        ewma_vol_medium = _ewma_volatility(asset_returns, ewma_long_span)
        realized_vol_medium = _rolling_volatility(asset_returns, medium_window)
        features[f"{asset}_ewma_vol_{short_window}w"] = ewma_vol_short
        features[f"{asset}_ewma_vol_{medium_window}w"] = ewma_vol_medium
        features[f"{asset}_realized_vol_{medium_window}w"] = realized_vol_medium
        features[f"{asset}_vol_ratio_{short_window}w_{medium_window}w"] = (
            ewma_vol_short / ewma_vol_medium.replace(0.0, np.nan)
        )

    return features


def _build_defensive_attractiveness_block(
    returns: pd.DataFrame,
    momentum_12w: pd.DataFrame,
    risk_adjusted_momentum_12w: pd.DataFrame,
    regime_features: pd.DataFrame,
    market_asset: str,
) -> pd.DataFrame:
    features = pd.DataFrame(index=returns.index)
    market_momentum = momentum_12w[market_asset]
    market_risk_adjusted_momentum = risk_adjusted_momentum_12w[market_asset]

    for defensive_asset in ("GLD", "TLT"):
        if defensive_asset not in returns.columns:
            continue
        relative_momentum = momentum_12w[defensive_asset] - market_momentum
        relative_risk_adjusted = (
            risk_adjusted_momentum_12w[defensive_asset]
            - market_risk_adjusted_momentum
        )
        features[f"{defensive_asset}_vs_{market_asset}_momentum_12w"] = (
            relative_momentum
        )
        features[
            f"{defensive_asset}_vs_{market_asset}_risk_adjusted_momentum_12w"
        ] = relative_risk_adjusted
        features[f"defensive_asset_score_{defensive_asset}"] = (
            0.55 * _sigmoid(relative_risk_adjusted)
            + 0.25 * _sigmoid(relative_momentum)
            + 0.20 * regime_features["p_risk_off"]
        )

    features["cash_permission_score"] = (
        0.65 * regime_features["p_risk_off"]
        + 0.35 * regime_features["p_market_drawdown_stress"]
    )

    return features


def _rolling_compounded_return(returns: pd.Series, window: int) -> pd.Series:
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def _rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std()


def _ewma_volatility(returns: pd.Series, span: int) -> pd.Series:
    return returns.ewm(span=span, adjust=False).std()


def _rolling_drawdown(returns: pd.Series, window: int) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    rolling_peak = wealth.rolling(window).max()

    return wealth / rolling_peak - 1.0


def _average_pairwise_correlation(
    returns: pd.DataFrame,
    assets: list[str],
    window: int,
) -> pd.Series:
    correlations = [
        returns[first_asset].rolling(window).corr(returns[second_asset])
        for first_asset, second_asset in combinations(assets, 2)
    ]

    return pd.concat(correlations, axis=1, sort=False).mean(axis=1)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    zscore = (series - rolling_mean) / rolling_std.replace(0.0, np.nan)

    return zscore.mask((rolling_std == 0.0) & series.notna(), 0.0)


def _sigmoid(value: pd.Series) -> pd.Series:
    clipped = value.clip(lower=-20.0, upper=20.0)

    return 1.0 / (1.0 + np.exp(-clipped))


def _risky_assets(returns: pd.DataFrame) -> list[str]:
    return [asset for asset in returns.columns if asset != "CASH"]


def _validate_inputs(
    returns: pd.DataFrame,
    market_asset: str,
    short_window: int,
    medium_window: int,
    long_window: int,
    ewma_short_span: int,
    ewma_long_span: int,
    correlation_window: int,
    zscore_window: int,
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

    for value, name in (
        (short_window, "short_window"),
        (medium_window, "medium_window"),
        (long_window, "long_window"),
        (ewma_short_span, "ewma_short_span"),
        (ewma_long_span, "ewma_long_span"),
        (correlation_window, "correlation_window"),
        (zscore_window, "zscore_window"),
    ):
        _validate_window(value, name)

    if medium_window < short_window:
        raise ValueError("medium_window must be greater than or equal to short_window.")
    if long_window < medium_window:
        raise ValueError("long_window must be greater than or equal to medium_window.")
    if len(_risky_assets(returns)) < 2:
        raise ValueError(
            "At least two risky assets excluding CASH are required for V6 features."
        )


def _validate_window(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer greater than or equal to 2.")
    if value < 2:
        raise ValueError(f"{name} must be an integer greater than or equal to 2.")
