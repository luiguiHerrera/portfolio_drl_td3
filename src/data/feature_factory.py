"""Feature builder selection for configured dataset preparation."""

import pandas as pd

from src.data.features import build_features
from src.data.features_v2 import build_features_v2


def build_configured_features(
    returns: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build features using the configured feature set version."""
    features_config = config.get("features", {})
    feature_version = features_config.get("version", "v1")
    if feature_version == "v1":
        return build_features(returns)
    if feature_version == "v2":
        return build_features_v2(
            returns,
            market_asset=features_config.get("market_asset", "SPY"),
            short_window=features_config.get("short_window", 4),
            long_window=features_config.get("long_window", 12),
            ewma_span=features_config.get("ewma_span", 12),
        )

    raise ValueError(f"Unsupported feature version: {feature_version}.")
