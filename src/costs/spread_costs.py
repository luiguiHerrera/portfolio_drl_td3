"""Bid-ask spread cost helpers for reporting-only robustness checks.

The functions in this module do not touch the portfolio environment. They are
intended for post-training execution-friction sensitivity analysis.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd


REQUIRED_QUOTE_COLUMNS = ["timestamp", "asset", "bid", "ask"]


def compute_half_spread(quotes: pd.DataFrame) -> pd.DataFrame:
    """Compute proportional top-of-book half-spreads from bid/ask quotes.

    Expected input columns are ``timestamp``, ``asset``, ``bid`` and ``ask``.
    The returned frame preserves ``timestamp`` and ``asset`` and adds
    ``mid`` and ``half_spread``.
    """
    _require_columns(quotes, REQUIRED_QUOTE_COLUMNS, "quotes")
    result = quotes.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["bid"] = pd.to_numeric(result["bid"], errors="coerce")
    result["ask"] = pd.to_numeric(result["ask"], errors="coerce")
    result["mid"] = (result["bid"] + result["ask"]) / 2.0
    invalid = (
        result["bid"].isna()
        | result["ask"].isna()
        | (result["bid"] < 0.0)
        | (result["ask"] < 0.0)
        | (result["ask"] < result["bid"])
        | (result["mid"] <= 0.0)
    )
    if invalid.any():
        raise ValueError("Quotes contain invalid bid/ask values.")
    result["half_spread"] = (result["ask"] - result["bid"]) / (2.0 * result["mid"])
    result.loc[result["asset"] == "CASH", "half_spread"] = 0.0
    return result


def aggregate_weekly_spreads(spreads: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quote half-spreads to weekly mean spreads by asset."""
    return _aggregate_weekly(spreads, method="mean")


def aggregate_weekly_close_spreads(spreads: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quote half-spreads to weekly last-observed close spreads."""
    return _aggregate_weekly(spreads, method="last")


def compute_spread_cost(
    target_weights: pd.Series | pd.DataFrame | np.ndarray,
    drifted_weights: pd.Series | pd.DataFrame | np.ndarray,
    half_spreads: pd.Series | Mapping[str, float] | pd.DataFrame | np.ndarray,
) -> float | pd.Series:
    """Compute spread cost from one-way traded notional and half-spreads.

    Formula:
        sum(abs(target_weight - drifted_weight) * asset_half_spread)
    """
    if isinstance(target_weights, pd.DataFrame):
        target = target_weights.astype(float)
        drifted = _coerce_dataframe_like(drifted_weights, target, "drifted_weights")
        spreads = _coerce_spreads_for_dataframe(half_spreads, target)
        return (target.sub(drifted).abs() * spreads).sum(axis=1)

    target_series = _coerce_series(target_weights, "target_weights")
    drifted_series = _coerce_series(drifted_weights, "drifted_weights")
    if not target_series.index.equals(drifted_series.index):
        drifted_series = drifted_series.reindex(target_series.index)
    if drifted_series.isna().any():
        raise ValueError("drifted_weights are missing assets present in target_weights.")

    spread_series = _coerce_spread_series(half_spreads, target_series.index)
    return float((target_series.sub(drifted_series).abs() * spread_series).sum())


def estimate_dynamic_spread_from_volatility(
    base_half_spread: float | pd.Series,
    rolling_vol: float | pd.Series,
    beta: float = 0.5,
) -> float | pd.Series:
    """Estimate proxy half-spread under a volatility regime.

    This is a robustness proxy, not calibrated execution truth. ``rolling_vol``
    is converted to a relative multiplier around its median when a series is
    provided. Scalar volatility is treated as a direct nonnegative multiplier
    input through ``1 + beta * rolling_vol``.
    """
    if beta < 0:
        raise ValueError("beta must be nonnegative.")

    if isinstance(rolling_vol, pd.Series):
        vol = pd.to_numeric(rolling_vol, errors="coerce").fillna(0.0).clip(lower=0.0)
        median = float(vol[vol > 0].median()) if (vol > 0).any() else 0.0
        if median <= 0.0:
            multiplier = pd.Series(1.0, index=vol.index)
        else:
            multiplier = 1.0 + beta * ((vol / median) - 1.0)
            multiplier = multiplier.clip(lower=0.0)
        return pd.Series(base_half_spread, index=vol.index, dtype=float) * multiplier

    base = float(base_half_spread)
    vol_value = float(rolling_vol)
    if base < 0.0 or vol_value < 0.0:
        raise ValueError("base_half_spread and rolling_vol must be nonnegative.")
    return base * (1.0 + beta * vol_value)


def build_proxy_weekly_spreads(
    dates: pd.Series | pd.DatetimeIndex | list[str],
    assets: list[str],
    base_half_spreads: Mapping[str, float],
    rolling_vol: pd.Series | pd.DataFrame | None = None,
    beta: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    """Build explicit proxy weekly spreads and warnings for missing quote data."""
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    frame = pd.DataFrame(index=index)
    warnings_list = [
        "Using proxy spread scenario assumptions because quote-level bid/ask data were not provided."
    ]
    for asset in assets:
        if asset == "CASH":
            frame[asset] = 0.0
            continue
        if asset not in base_half_spreads:
            raise ValueError(f"Missing proxy base half-spread for asset: {asset}")
        base = float(base_half_spreads[asset])
        if base < 0.0:
            raise ValueError(f"Negative proxy base half-spread for asset: {asset}")
        if rolling_vol is None:
            frame[asset] = base
        elif isinstance(rolling_vol, pd.DataFrame) and asset in rolling_vol.columns:
            frame[asset] = estimate_dynamic_spread_from_volatility(
                base,
                rolling_vol[asset].reindex(index),
                beta=beta,
            )
        elif isinstance(rolling_vol, pd.Series):
            frame[asset] = estimate_dynamic_spread_from_volatility(base, rolling_vol.reindex(index), beta=beta)
        else:
            warnings_list.append(f"No rolling volatility supplied for {asset}; using static proxy spread.")
            frame[asset] = base
    frame["CASH"] = 0.0
    warnings.warn(warnings_list[0], RuntimeWarning, stacklevel=2)
    return frame, warnings_list


def _aggregate_weekly(spreads: pd.DataFrame, method: str) -> pd.DataFrame:
    _require_columns(spreads, ["timestamp", "asset", "half_spread"], "spreads")
    frame = spreads.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["half_spread"] = pd.to_numeric(frame["half_spread"], errors="coerce")
    if frame["half_spread"].isna().any() or (frame["half_spread"] < 0.0).any():
        raise ValueError("Spreads contain missing or negative half_spread values.")
    frame["date"] = frame["timestamp"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    if method == "mean":
        weekly = frame.groupby(["date", "asset"], as_index=False)["half_spread"].mean()
    elif method == "last":
        weekly = (
            frame.sort_values("timestamp")
            .groupby(["date", "asset"], as_index=False)
            .tail(1)
            .loc[:, ["date", "asset", "half_spread"]]
        )
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    wide = weekly.pivot(index="date", columns="asset", values="half_spread").sort_index()
    wide.index.name = "date"
    wide["CASH"] = 0.0
    return wide


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _coerce_series(value: pd.Series | np.ndarray, label: str) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(float)
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    return pd.Series(array)


def _coerce_dataframe_like(value: pd.DataFrame | np.ndarray, target: pd.DataFrame, label: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.reindex(index=target.index, columns=target.columns).astype(float)
    else:
        frame = pd.DataFrame(np.asarray(value, dtype=float), index=target.index, columns=target.columns)
    if frame.isna().any().any():
        raise ValueError(f"{label} is missing target rows or columns.")
    return frame


def _coerce_spreads_for_dataframe(
    value: pd.Series | Mapping[str, float] | pd.DataFrame | np.ndarray,
    target: pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.reindex(index=target.index, columns=target.columns).astype(float)
    elif isinstance(value, (pd.Series, Mapping)):
        series = _coerce_spread_series(value, target.columns)
        frame = pd.DataFrame([series.to_numpy()] * len(target), index=target.index, columns=target.columns)
    else:
        array = np.asarray(value, dtype=float)
        if array.ndim == 1:
            frame = pd.DataFrame([array] * len(target), index=target.index, columns=target.columns)
        else:
            frame = pd.DataFrame(array, index=target.index, columns=target.columns)
    if frame.isna().any().any() or (frame < 0.0).any().any():
        raise ValueError("half_spreads contain missing or negative values.")
    if "CASH" in frame.columns:
        frame["CASH"] = 0.0
    return frame


def _coerce_spread_series(value: pd.Series | Mapping[str, float] | np.ndarray, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.reindex(index).astype(float)
    elif isinstance(value, Mapping):
        series = pd.Series(value, dtype=float).reindex(index)
    else:
        series = pd.Series(np.asarray(value, dtype=float), index=index)
    if series.isna().any() or (series < 0.0).any():
        raise ValueError("half_spreads contain missing or negative values.")
    if "CASH" in series.index:
        series.loc["CASH"] = 0.0
    return series
