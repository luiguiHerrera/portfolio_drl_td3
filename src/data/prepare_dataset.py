"""Model dataset preparation utilities.

This module prepares aligned realized returns and normalized feature matrices
for train, validation, and test periods. Returns are never normalized because
they are used for realized portfolio rewards.
"""

import pandas as pd

from src.data.build_dataset import build_returns_dataset
from src.data.feature_factory import build_configured_features
from src.data.normalize import normalize_train_validation_test
from src.data.split import chronological_split
from src.utils.config import load_config


def prepare_train_validation_test_datasets(config_path: str) -> dict:
    """Prepare aligned returns and train-normalized features for model experiments."""
    config = load_config(config_path)
    training_config = config["training"]

    returns = build_returns_dataset(config_path)
    raw_features = build_configured_features(returns, config)
    features_available_before_return = raw_features.shift(1).dropna()
    aligned_returns, aligned_features = _align_returns_and_features(
        returns,
        features_available_before_return,
    )

    train_features, validation_features, test_features = chronological_split(
        aligned_features,
        training_config["train_ratio"],
        training_config["validation_ratio"],
        training_config["test_ratio"],
    )
    train_returns = aligned_returns.loc[train_features.index]
    validation_returns = aligned_returns.loc[validation_features.index]
    test_returns = aligned_returns.loc[test_features.index]

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


def _align_returns_and_features(
    returns: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared_index = returns.index[returns.index.isin(features.index)]
    if shared_index.empty:
        raise ValueError("returns and features must have at least one shared index.")

    aligned_returns = returns.loc[shared_index]
    aligned_features = features.loc[shared_index]

    if aligned_returns.empty or aligned_features.empty:
        raise ValueError("aligned returns and features must not be empty.")

    return aligned_returns, aligned_features
