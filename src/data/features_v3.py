"""Feature Set V3 with optional externally supplied macro features.

V3 extends Feature Set V2 without downloading macro data. Macro observations
are expected to be supplied by the caller and are aligned to the feature index
using forward fill only. Dataset preparation remains responsible for shifting
features by one period before alignment with realized returns.
"""

import pandas as pd

from src.data.features_v2 import build_features_v2


def build_features_v3(
    returns: pd.DataFrame,
    macro_data: pd.DataFrame | None = None,
    market_asset: str = "SPY",
    short_window: int = 4,
    long_window: int = 12,
    ewma_span: int = 12,
) -> pd.DataFrame:
    """Build V2 features and optionally append forward-filled macro features."""
    v2_features = build_features_v2(
        returns,
        market_asset=market_asset,
        short_window=short_window,
        long_window=long_window,
        ewma_span=ewma_span,
    )
    if macro_data is None:
        return v2_features

    macro_features = build_macro_features(
        macro_data=macro_data,
        target_index=v2_features.index,
    )
    combined_features = pd.concat([v2_features, macro_features], axis=1)

    return combined_features.dropna()


def build_macro_features(
    macro_data: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Align external macro data and build simple macro regime features.

    DXY and CPI momentum use 12-period percentage change on the aligned macro
    series.
    """
    _validate_macro_inputs(macro_data, target_index)
    sorted_macro_data = macro_data.sort_index()
    aligned_macro_data = sorted_macro_data.reindex(target_index, method="ffill")
    macro_features = aligned_macro_data.add_prefix("macro_")

    if {"DGS10", "DGS2"}.issubset(aligned_macro_data.columns):
        yield_curve = aligned_macro_data["DGS10"] - aligned_macro_data["DGS2"]
        macro_features["macro_yield_curve_10y_2y"] = yield_curve
        macro_features["macro_inverted_yield_curve_regime"] = (
            yield_curve < 0.0
        ).astype(float)

    if "VIX" in aligned_macro_data.columns:
        vix = aligned_macro_data["VIX"]
        rolling_median = vix.rolling(12).median()
        high_vix_regime = (vix > rolling_median).astype(float)
        high_vix_regime = high_vix_regime.mask(rolling_median.isna())
        macro_features["macro_high_vix_regime"] = high_vix_regime

    if "DXY" in aligned_macro_data.columns:
        dollar_momentum = aligned_macro_data["DXY"].pct_change(12)
        macro_features["macro_dollar_momentum_12p"] = dollar_momentum
        strong_dollar_regime = (dollar_momentum > 0.0).astype(float)
        strong_dollar_regime = strong_dollar_regime.mask(dollar_momentum.isna())
        macro_features["macro_strong_dollar_regime"] = strong_dollar_regime

    if "CPI" in aligned_macro_data.columns:
        cpi_momentum = aligned_macro_data["CPI"].pct_change(12)
        macro_features["macro_cpi_momentum_12p"] = cpi_momentum
        inflation_pressure_regime = (cpi_momentum > 0.0).astype(float)
        inflation_pressure_regime = inflation_pressure_regime.mask(cpi_momentum.isna())
        macro_features["macro_inflation_pressure_regime"] = inflation_pressure_regime

    return macro_features


def _validate_macro_inputs(
    macro_data: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> None:
    if not isinstance(macro_data, pd.DataFrame):
        raise TypeError("macro_data must be a pandas DataFrame.")
    if macro_data.empty:
        raise ValueError("macro_data must be a non-empty DataFrame.")
    if not isinstance(macro_data.index, pd.DatetimeIndex):
        raise TypeError("macro_data index must be a pandas DatetimeIndex.")
    if not isinstance(target_index, pd.DatetimeIndex):
        raise TypeError("target_index must be a pandas DatetimeIndex.")
