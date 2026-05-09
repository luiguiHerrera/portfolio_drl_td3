"""Allocation diagnostics for portfolio weight histories."""

import numpy as np
import pandas as pd


def allocation_diagnostics(
    weights: pd.DataFrame,
    turnover: pd.Series | None = None,
    transaction_costs: pd.Series | None = None,
) -> dict:
    """Compute concentration and allocation diagnostics from portfolio weights."""
    _validate_weights(weights)

    max_weights = weights.max(axis=1)
    hhi = (weights**2).sum(axis=1)
    effective_number_of_assets = 1.0 / hhi
    entropy = weights.apply(_row_entropy, axis=1)

    diagnostics = {
        "average_max_weight": float(max_weights.mean()),
        "final_max_weight": float(max_weights.iloc[-1]),
        "average_cash_weight": float(weights["CASH"].mean()) if "CASH" in weights else 0.0,
        "final_cash_weight": float(weights["CASH"].iloc[-1]) if "CASH" in weights else 0.0,
        "average_herfindahl_index": float(hhi.mean()),
        "final_herfindahl_index": float(hhi.iloc[-1]),
        "average_effective_number_of_assets": float(effective_number_of_assets.mean()),
        "final_effective_number_of_assets": float(effective_number_of_assets.iloc[-1]),
        "average_entropy": float(entropy.mean()),
        "final_entropy": float(entropy.iloc[-1]),
    }

    if turnover is not None:
        _validate_optional_series(turnover, weights.index, "turnover")
        diagnostics["average_turnover"] = float(turnover.mean())
        diagnostics["final_turnover"] = float(turnover.iloc[-1])

    if transaction_costs is not None:
        _validate_optional_series(
            transaction_costs,
            weights.index,
            "transaction_costs",
        )
        diagnostics["average_transaction_cost"] = float(transaction_costs.mean())
        diagnostics["final_transaction_cost"] = float(transaction_costs.iloc[-1])

    return diagnostics


def _validate_weights(weights: pd.DataFrame) -> None:
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame.")
    if weights.empty:
        raise ValueError("weights must not be empty.")
    if (weights < 0.0).any().any():
        raise ValueError("weights must not contain negative values.")
    if not np.allclose(weights.sum(axis=1).to_numpy(), 1.0):
        raise ValueError("each row of weights must sum approximately to 1.0.")


def _validate_optional_series(series: pd.Series, expected_index: pd.Index, name: str) -> None:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if not series.index.equals(expected_index):
        raise ValueError(f"{name} index must match weights index.")


def _row_entropy(row: pd.Series) -> float:
    positive_weights = row[row > 0.0]
    return float(-(positive_weights * np.log(positive_weights)).sum())
