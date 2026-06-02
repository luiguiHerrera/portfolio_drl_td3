"""Feature Set V2 for return-derived portfolio state construction.

This module is additive to the original V1 feature pipeline. It builds richer
asset-level and market-regime features from the supplied returns only, and it
does not shift features internally. Dataset preparation remains responsible for
shifting features by one period before alignment with realized returns.
"""

import numpy as np
import pandas as pd


def build_features_v2(
    returns: pd.DataFrame,
    market_asset: str = "SPY",
    short_window: int = 4,
    long_window: int = 12,
    ewma_span: int = 12,
) -> pd.DataFrame:
    """Build return-derived asset and regime features without look-ahead shifts."""
    _validate_inputs(
        returns=returns,
        market_asset=market_asset,
        short_window=short_window,
        long_window=long_window,
        ewma_span=ewma_span,
    )

    market_returns = returns[market_asset]
    feature_frames = []
    for asset in returns.columns:
        asset_returns = returns[asset]
        asset_features = pd.DataFrame(index=returns.index)
        asset_features[f"{asset}_ret_1p"] = asset_returns
        asset_features[f"{asset}_mom_{short_window}p"] = _rolling_compounded_return(
            asset_returns,
            short_window,
        )
        asset_features[f"{asset}_mom_{long_window}p"] = _rolling_compounded_return(
            asset_returns,
            long_window,
        )
        asset_features[f"{asset}_vol_{short_window}p"] = asset_returns.rolling(
            short_window
        ).std()
        asset_features[f"{asset}_vol_{long_window}p"] = asset_returns.rolling(
            long_window
        ).std()
        asset_features[f"{asset}_ewma_vol_{ewma_span}p"] = ewma_volatility(
            asset_returns,
            ewma_span,
        )
        if asset != market_asset:
            asset_features[f"{asset}_beta_vs_{market_asset}_{long_window}p"] = rolling_beta(
                asset_returns,
                market_returns,
                long_window,
            )
            asset_features[f"{asset}_corr_vs_{market_asset}_{long_window}p"] = (
                rolling_correlation(asset_returns, market_returns, long_window)
            )
        asset_features[f"{asset}_rolling_drawdown_{long_window}p"] = rolling_drawdown(
            asset_returns,
            long_window,
        )
        feature_frames.append(
            _fill_static_asset_market_exposure(
                asset_features=asset_features,
                asset_returns=asset_returns,
                market_returns=market_returns,
                market_asset=market_asset,
                long_window=long_window,
            )
        )

    regime_features = _build_regime_features(
        returns=returns,
        market_asset=market_asset,
        long_window=long_window,
    )
    features = pd.concat([*feature_frames, regime_features], axis=1)
    features = features.dropna()
    if features.empty:
        raise ValueError(
            "Feature Set V2 output is empty after dropping rolling-window NaNs; "
            "provide more return observations or shorter windows."
        )

    return features


def rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling beta as covariance(asset, market) / variance(market)."""
    _validate_window(window, "window")
    market_variance = market_returns.rolling(window).var()
    beta = asset_returns.rolling(window).cov(market_returns) / market_variance

    return beta.mask(market_variance == 0.0)


def rolling_correlation(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling correlation between asset and market returns."""
    _validate_window(window, "window")

    return asset_returns.rolling(window).corr(market_returns)


def ewma_volatility(
    asset_returns: pd.Series,
    span: int,
) -> pd.Series:
    """Compute exponentially weighted moving volatility."""
    _validate_window(span, "span")

    return asset_returns.ewm(span=span, adjust=False).std()


def rolling_drawdown(
    asset_returns: pd.Series,
    window: int,
) -> pd.Series:
    """Compute drawdown from the rolling wealth peak."""
    _validate_window(window, "window")
    wealth = (1.0 + asset_returns).cumprod()
    rolling_peak = wealth.rolling(window).max()

    return wealth / rolling_peak - 1.0


def _build_regime_features(
    returns: pd.DataFrame,
    market_asset: str,
    long_window: int,
) -> pd.DataFrame:
    market_returns = returns[market_asset]
    market_momentum = _rolling_compounded_return(market_returns, long_window)
    market_volatility = market_returns.rolling(long_window).std()
    market_volatility_median = market_volatility.rolling(long_window).median()
    regime_features = pd.DataFrame(index=returns.index)
    regime_features["market_high_vol_regime"] = (
        market_volatility > market_volatility_median
    ).astype(float)
    regime_features["market_risk_off_regime"] = _risk_off_regime(
        returns=returns,
        market_asset=market_asset,
    )
    regime_features["market_trend_regime"] = (market_momentum > 0.0).astype(float)
    regime_features["market_defensive_regime"] = _defensive_regime(
        returns=returns,
        market_momentum=market_momentum,
        long_window=long_window,
    )

    return regime_features


def _risk_off_regime(
    returns: pd.DataFrame,
    market_asset: str,
) -> pd.Series:
    market_returns = returns[market_asset]
    defensive_assets = [asset for asset in ("GLD", "CASH") if asset in returns.columns]
    if defensive_assets:
        defensive_outperformance = pd.concat(
            [returns[asset] > market_returns for asset in defensive_assets],
            axis=1,
        ).any(axis=1)
        risk_off = (market_returns < 0.0) & defensive_outperformance
    else:
        risk_off = market_returns < 0.0

    return risk_off.astype(float)


def _defensive_regime(
    returns: pd.DataFrame,
    market_momentum: pd.Series,
    long_window: int,
) -> pd.Series:
    defensive_assets = [asset for asset in ("GLD", "TLT") if asset in returns.columns]
    if not defensive_assets:
        return pd.Series(0.0, index=returns.index)

    defensive_momentum = pd.concat(
        [
            _rolling_compounded_return(returns[asset], long_window) > market_momentum
            for asset in defensive_assets
        ],
        axis=1,
    )

    return defensive_momentum.any(axis=1).astype(float)


def _fill_static_asset_market_exposure(
    asset_features: pd.DataFrame,
    asset_returns: pd.Series,
    market_returns: pd.Series,
    market_asset: str,
    long_window: int,
) -> pd.DataFrame:
    market_variance = market_returns.rolling(long_window).var()
    asset_variance = asset_returns.rolling(long_window).var()
    static_asset_mask = (asset_variance == 0.0) & (market_variance > 0.0)
    beta_columns = [
        column
        for column in asset_features.columns
        if column.endswith(f"_beta_vs_{market_asset}_{long_window}p")
    ]
    corr_columns = [
        column
        for column in asset_features.columns
        if column.endswith(f"_corr_vs_{market_asset}_{long_window}p")
    ]
    if beta_columns:
        asset_features.loc[static_asset_mask, beta_columns[0]] = 0.0
    if corr_columns:
        asset_features.loc[static_asset_mask, corr_columns[0]] = 0.0

    return asset_features


def _rolling_compounded_return(returns: pd.Series, window: int) -> pd.Series:
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def _validate_inputs(
    returns: pd.DataFrame,
    market_asset: str,
    short_window: int,
    long_window: int,
    ewma_span: int,
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
    if long_window < short_window:
        raise ValueError("long_window must be greater than or equal to short_window.")


def _validate_window(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer greater than or equal to 2.")
    if value < 2:
        raise ValueError(f"{name} must be an integer greater than or equal to 2.")
