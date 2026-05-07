"""Chronological dataset splitting utilities.

This module provides deterministic time-ordered splits for financial datasets.
No shuffling is performed, which avoids look-ahead leakage in time-series
experiments.
"""

import numpy as np
import pandas as pd


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into chronological train, validation, and test partitions."""
    _validate_split_inputs(data, train_ratio, validation_ratio, test_ratio)

    n_rows = len(data)
    train_end = int(n_rows * train_ratio)
    validation_end = train_end + int(n_rows * validation_ratio)

    train = data.iloc[:train_end]
    validation = data.iloc[train_end:validation_end]
    test = data.iloc[validation_end:]

    if train.empty or validation.empty or test.empty:
        raise ValueError("chronological_split produced an empty partition.")

    return train, validation, test


def _validate_split_inputs(
    data: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)

    if data.empty:
        raise ValueError("data must not be empty.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("data index must be sorted in increasing chronological order.")
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("split ratios must be positive.")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to 1.0.")
