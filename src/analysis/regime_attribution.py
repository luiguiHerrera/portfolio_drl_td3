"""Regime attribution helpers for policy behavior diagnostics.

This module joins exported policy history with raw, unnormalized feature values
so regime indicators remain interpretable during behavior analysis.
"""

from pathlib import Path

import pandas as pd

from src.analysis.policy_behavior_diagnostics import build_policy_behavior_report
from src.data.build_dataset import build_returns_dataset
from src.data.feature_factory import build_configured_features
from src.data.walk_forward_split import (
    _validate_chronological_folds,
    _validate_fold,
    slice_dataset_by_date,
)
from src.utils.config import load_config


REGIME_COLUMN_PATTERNS = (
    "regime",
    "yield_curve",
    "momentum",
    "vol",
    "drawdown",
    "beta",
    "corr",
)


def get_candidate_regime_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric feature columns that are useful for regime attribution."""
    candidates = []
    for column in features.columns:
        column_name = str(column)
        lower_name = column_name.lower()
        matches_pattern = column_name.startswith("macro_") or any(
            pattern in lower_name for pattern in REGIME_COLUMN_PATTERNS
        )
        if not matches_pattern:
            continue
        if not pd.api.types.is_numeric_dtype(features[column]):
            continue
        if pd.api.types.is_bool_dtype(features[column]):
            continue
        if features[column].isna().all():
            continue

        candidates.append(column)

    return candidates


def merge_policy_history_with_features(
    policy_history: pd.DataFrame,
    features: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """Left join policy history to raw features by exact dates."""
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features index must be a DatetimeIndex.")

    policy = policy_history.copy()
    if date_column in policy.columns:
        policy[date_column] = pd.to_datetime(policy[date_column], errors="coerce")
        merge_dates = policy[date_column]
    elif isinstance(policy.index, pd.DatetimeIndex):
        merge_dates = policy.index
        policy[date_column] = merge_dates
    else:
        raise ValueError(
            "policy_history must have a date column or a DatetimeIndex."
        )

    shared_dates = pd.DatetimeIndex(merge_dates).intersection(features.index)
    if shared_dates.empty:
        raise ValueError("policy_history and features have no shared dates.")

    feature_frame = features.copy()
    feature_frame = feature_frame.loc[~feature_frame.index.duplicated(keep="last")]
    feature_frame[date_column] = feature_frame.index

    return policy.merge(feature_frame, how="left", on=date_column)


def build_raw_walk_forward_features(
    config_path: str,
    fold: dict,
) -> dict:
    """Build raw shifted walk-forward features without normalization."""
    _validate_fold(fold)
    config = load_config(config_path)
    returns = build_returns_dataset(config_path)
    raw_features = build_configured_features(returns, config)
    features_available_before_return = raw_features.shift(1).dropna()

    train_returns, train_features_raw = slice_dataset_by_date(
        returns,
        features_available_before_return,
        fold["train_start"],
        fold["train_end"],
    )
    validation_returns, validation_features_raw = slice_dataset_by_date(
        returns,
        features_available_before_return,
        fold["validation_start"],
        fold["validation_end"],
    )
    test_returns, test_features_raw = slice_dataset_by_date(
        returns,
        features_available_before_return,
        fold["test_start"],
        fold["test_end"],
    )

    _validate_chronological_folds(
        train_returns.index,
        validation_returns.index,
        test_returns.index,
    )

    return {
        "train_features_raw": train_features_raw,
        "validation_features_raw": validation_features_raw,
        "test_features_raw": test_features_raw,
        "train_returns": train_returns,
        "validation_returns": validation_returns,
        "test_returns": test_returns,
    }


def build_regime_attribution_report(
    policy_history_path: str,
    features: pd.DataFrame,
    output_dir: str | None = None,
    report_name: str = "regime_attribution",
    date_column: str = "date",
    return_column: str = "portfolio_return",
    regime_columns: list[str] | None = None,
) -> dict:
    """Build a regime attribution report from policy history and raw features."""
    policy_history = pd.read_csv(policy_history_path)
    selected_regime_columns = (
        get_candidate_regime_columns(features)
        if regime_columns is None
        else regime_columns
    )
    merged = merge_policy_history_with_features(
        policy_history,
        features,
        date_column=date_column,
    )
    behavior_report = build_policy_behavior_report(
        merged,
        return_column=return_column,
        regime_columns=selected_regime_columns,
    )

    outputs = {
        "merged_observations": behavior_report["observations"],
        "regime_attribution": behavior_report["regime_attribution"],
        "dominant_asset_distribution": behavior_report[
            "dominant_asset_distribution"
        ],
        "concentration_summary": behavior_report["concentration_summary"],
        "regime_columns": selected_regime_columns,
        "merged_observations_path": None,
        "regime_attribution_path": None,
        "dominant_asset_distribution_path": None,
        "concentration_summary_path": None,
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        save_targets = {
            "merged_observations_path": outputs["merged_observations"],
            "regime_attribution_path": outputs["regime_attribution"],
            "dominant_asset_distribution_path": outputs[
                "dominant_asset_distribution"
            ],
            "concentration_summary_path": outputs["concentration_summary"],
        }
        for key, frame in save_targets.items():
            file_name = key.removesuffix("_path")
            path = destination / f"{report_name}_{file_name}.csv"
            frame.to_csv(path, index=False)
            outputs[key] = str(path)

    return outputs


def build_walk_forward_regime_attribution_report(
    config_path: str,
    fold: dict,
    policy_history_path: str,
    split: str = "test",
    output_dir: str | None = None,
    report_name: str = "walk_forward_regime_attribution",
    regime_columns: list[str] | None = None,
) -> dict:
    """Build regime attribution for one walk-forward validation/test split."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")

    raw_datasets = build_raw_walk_forward_features(config_path, fold)
    features = raw_datasets[f"{split}_features_raw"]

    return build_regime_attribution_report(
        policy_history_path=policy_history_path,
        features=features,
        output_dir=output_dir,
        report_name=report_name,
        regime_columns=regime_columns,
    )
