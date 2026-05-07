"""Feature normalization utilities.

This module implements standard scaling for train, validation, and test
features. The scaler is fit only on the training partition to avoid data leakage.
It uses pandas' sample standard deviation (`ddof=1`).
"""

import pandas as pd


def fit_standard_scaler(train_features: pd.DataFrame) -> dict[str, pd.Series]:
    """Fit a standard scaler using training features only."""
    if train_features.empty:
        raise ValueError("train_features must not be empty.")

    if train_features.columns.duplicated().any():
        raise ValueError("train_features must not contain duplicate columns.")

    mean = train_features.mean()
    std = train_features.std(ddof=1)
    std = std.mask((std == 0.0) | std.isna(), 1.0)

    return {
        "mean": mean,
        "std": std,
    }


def apply_standard_scaler(features: pd.DataFrame, scaler: dict[str, pd.Series]) -> pd.DataFrame:
    """Apply a previously fit standard scaler to a features DataFrame."""
    _validate_scaler(scaler)

    if features.empty:
        raise ValueError("features must not be empty.")

    if features.columns.duplicated().any():
        raise ValueError("features must not contain duplicate columns.")

    mean = scaler["mean"]
    std = scaler["std"]

    if not features.columns.equals(mean.index) or not features.columns.equals(std.index):
        raise ValueError("features columns must match scaler mean and std indexes.")

    normalized = (features - mean) / std

    return normalized.loc[:, features.columns]


def normalize_train_validation_test(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """Fit on train and apply standard scaling to all chronological splits."""
    scaler = fit_standard_scaler(train)

    return (
        apply_standard_scaler(train, scaler),
        apply_standard_scaler(validation, scaler),
        apply_standard_scaler(test, scaler),
        scaler,
    )


def _validate_scaler(scaler: dict[str, pd.Series]) -> None:
    missing_keys = [key for key in ("mean", "std") if key not in scaler]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise KeyError(f"scaler is missing required keys: {missing}")

    if not isinstance(scaler["mean"], pd.Series):
        raise TypeError("scaler['mean'] must be a pandas Series.")

    if not isinstance(scaler["std"], pd.Series):
        raise TypeError("scaler['std'] must be a pandas Series.")

    if not scaler["mean"].index.equals(scaler["std"].index):
        raise ValueError("scaler mean and std indexes must match.")