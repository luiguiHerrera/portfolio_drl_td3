"""Feature engineering utilities for portfolio state construction.

This module builds a first set of transparent return-based features from asset
returns. It intentionally does not normalize features or create train,
validation, or test splits.
"""

import numpy as np
import pandas as pd


def build_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Build rolling return, momentum, and volatility features from returns."""
    feature_frames = []

    for asset in returns.columns:
        asset_returns = returns[asset]
        asset_features = pd.DataFrame(index=returns.index)
        asset_features[f"{asset}_ret_1w"] = asset_returns
        asset_features[f"{asset}_mom_4w"] = _rolling_compounded_return(asset_returns, 4)
        asset_features[f"{asset}_mom_12w"] = _rolling_compounded_return(asset_returns, 12)
        asset_features[f"{asset}_vol_4w"] = asset_returns.rolling(4).std()
        asset_features[f"{asset}_vol_12w"] = asset_returns.rolling(12).std()
        feature_frames.append(asset_features)

    features = pd.concat(feature_frames, axis=1)

    return features.dropna()


def _rolling_compounded_return(returns: pd.Series, window: int) -> pd.Series:
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0
