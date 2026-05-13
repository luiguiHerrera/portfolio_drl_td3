"""Explicit chronological walk-forward dataset construction utilities."""

import pandas as pd

from src.data.build_dataset import build_returns_dataset
from src.data.feature_factory import build_configured_features
from src.data.normalize import normalize_train_validation_test
from src.utils.config import load_config


def slice_dataset_by_date(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align returns/features and slice an inclusive chronological date range."""
    _validate_dataframe(returns, "returns")
    _validate_dataframe(features, "features")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date must be less than or equal to end_date.")

    shared_index = returns.index.intersection(features.index)
    if shared_index.empty:
        raise ValueError("returns and features must have at least one shared index.")

    aligned_returns = returns.loc[shared_index].sort_index()
    aligned_features = features.loc[shared_index].sort_index()
    sliced_returns = aligned_returns.loc[start:end]
    sliced_features = aligned_features.loc[start:end]

    if sliced_returns.empty or sliced_features.empty:
        raise ValueError(
            f"empty dataset slice for date range {start_date} through {end_date}."
        )
    if not sliced_returns.index.equals(sliced_features.index):
        raise ValueError("sliced returns and features indexes must match.")

    return sliced_returns, sliced_features


def build_walk_forward_datasets(
    config_path: str,
    fold: dict,
) -> dict:
    """Build train/validation/test datasets from explicit fold date ranges."""
    _validate_fold(fold)
    config = load_config(config_path)
    returns = build_returns_dataset(config_path)
    raw_features = build_configured_features(returns, config)
    features_available_before_return = raw_features.shift(1).dropna()

    train_returns, train_features = slice_dataset_by_date(
        returns,
        features_available_before_return,
        fold["train_start"],
        fold["train_end"],
    )
    validation_returns, validation_features = slice_dataset_by_date(
        returns,
        features_available_before_return,
        fold["validation_start"],
        fold["validation_end"],
    )
    test_returns, test_features = slice_dataset_by_date(
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

    (
        train_features_normalized,
        validation_features_normalized,
        test_features_normalized,
        feature_scaler,
    ) = normalize_train_validation_test(
        train_features,
        validation_features,
        test_features,
    )

    return {
        "train_returns": train_returns,
        "validation_returns": validation_returns,
        "test_returns": test_returns,
        "train_features": train_features_normalized,
        "validation_features": validation_features_normalized,
        "test_features": test_features_normalized,
        "feature_scaler": feature_scaler,
    }


def _validate_dataframe(dataframe: pd.DataFrame, name: str) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError(f"{name} must not be empty.")
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError(f"{name} index must be a pandas DatetimeIndex.")


def _validate_fold(fold: dict) -> None:
    required_keys = {
        "fold_id",
        "description",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    }
    missing_keys = required_keys.difference(fold)
    if missing_keys:
        raise KeyError(f"fold is missing required keys: {sorted(missing_keys)}")


def _validate_chronological_folds(
    train_index: pd.DatetimeIndex,
    validation_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
) -> None:
    if train_index.max() >= validation_index.min():
        raise ValueError("train fold must end before validation fold starts.")
    if validation_index.max() >= test_index.min():
        raise ValueError("validation fold must end before test fold starts.")
