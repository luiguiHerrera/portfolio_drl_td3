"""Feature builder selection for configured dataset preparation."""

import pandas as pd

from src.data.features import build_features
from src.data.features_v2 import build_features_v2
from src.data.features_v3 import build_features_v3
from src.data.features_v4 import build_features_v4
from src.data.features_v5 import build_features_v5
from src.data.features_v6 import build_features_v6
from src.data.macro_loader import load_macro_data_from_csv


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
    if feature_version == "v3":
        macro_data = None
        if "macro_path" in features_config:
            macro_data = load_macro_data_from_csv(
                features_config["macro_path"],
                date_column=features_config.get("macro_date_column", "date"),
            )

        return build_features_v3(
            returns,
            macro_data=macro_data,
            market_asset=features_config.get("market_asset", "SPY"),
            short_window=features_config.get("short_window", 4),
            long_window=features_config.get("long_window", 12),
            ewma_span=features_config.get("ewma_span", 12),
        )
    if feature_version == "v4":
        return build_features_v4(
            returns,
            market_asset=features_config.get("market_asset", "SPY"),
            short_window=features_config.get("short_window", 4),
            long_window=features_config.get("long_window", 12),
            ewma_span=features_config.get("ewma_span", 12),
            include_garch_features=features_config.get("include_garch_features", True),
            garch_include_relative=features_config.get(
                "garch_include_relative",
                True,
            ),
            garch_omega=features_config.get("garch_omega", 1e-6),
            garch_alpha=features_config.get("garch_alpha", 0.05),
            garch_beta=features_config.get("garch_beta", 0.90),
            garch_periods_per_year=features_config.get("garch_periods_per_year", 52),
            garch_mode=features_config.get("garch_mode", "deterministic_filter"),
            garch_min_history=features_config.get("garch_min_history", 104),
            garch_window=features_config.get("garch_window", 156),
            garch_annualize=features_config.get("garch_annualize", False),
            garch_exclude_cash=features_config.get("garch_exclude_cash", False),
            garch_fallback=features_config.get("garch_fallback", "rolling_realized_vol"),
        )
    if feature_version == "v5":
        return build_features_v5(
            returns,
            market_asset=features_config.get("market_asset", "SPY"),
            short_window=features_config.get("short_window", 4),
            long_window=features_config.get("long_window", 12),
            ewma_span=features_config.get("ewma_span", 12),
            correlation_window=features_config.get("correlation_window", 12),
            drawdown_window=features_config.get("drawdown_window", 12),
            risk_off_threshold=features_config.get("risk_off_threshold", 2.0),
        )
    if feature_version == "v6":
        return build_features_v6(
            returns,
            market_asset=features_config.get("market_asset", "SPY"),
            short_window=features_config.get("short_window", 4),
            medium_window=features_config.get("medium_window", 12),
            long_window=features_config.get("long_window", 26),
            ewma_short_span=features_config.get("ewma_short_span", 4),
            ewma_long_span=features_config.get("ewma_long_span", 12),
            correlation_window=features_config.get("correlation_window", 12),
            zscore_window=features_config.get("zscore_window", 52),
        )

    raise ValueError(f"Unsupported feature version: {feature_version}.")
