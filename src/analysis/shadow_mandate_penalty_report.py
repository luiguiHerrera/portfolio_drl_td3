"""Ex-post mandate penalty reports for saved strategy histories.

This module applies mandate penalty components to saved TD3 or benchmark
history CSVs without retraining, changing rewards, or changing environment
dynamics.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.risk.mandate_penalties import compute_mandate_penalty
from src.risk.mandate_profiles import get_mandate_limits


DEFAULT_MANDATE_PROFILES = ["conservative", "moderate", "aggressive"]
BREACH_COLUMNS = [
    "drawdown_breach",
    "volatility_breach",
    "max_weight_breach",
    "effective_assets_breach",
    "turnover_breach",
]


def infer_weight_columns(data: pd.DataFrame) -> list[str]:
    """Return non-flag weight columns from a history DataFrame."""
    weight_columns = [
        column
        for column in data.columns
        if column.startswith("weight_")
        and not column.endswith("_ok")
        and not pd.api.types.is_bool_dtype(data[column])
    ]
    if not weight_columns:
        raise ValueError("No weight_ columns found.")

    return weight_columns


def compute_effective_assets_from_weights(weights: pd.DataFrame) -> pd.Series:
    """Compute effective number of assets from weight columns."""
    _validate_numeric_frame(weights, "weights")
    herfindahl = (weights**2).sum(axis=1)

    return (1.0 / herfindahl.replace(0.0, np.nan)).rename("effective_assets")


def compute_trailing_volatility(
    returns: pd.Series,
    window: int = 12,
    periods_per_year: int = 52,
) -> pd.Series:
    """Compute annualized rolling volatility using observations through each row."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer greater than or equal to 2.")
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year < 1
    ):
        raise ValueError("periods_per_year must be an integer greater than or equal to 1.")
    if not pd.api.types.is_numeric_dtype(returns):
        raise ValueError("returns must be numeric.")

    return (returns.rolling(window).std() * np.sqrt(periods_per_year)).rename(
        "trailing_volatility"
    )


def build_shadow_mandate_penalty_observations(
    history: pd.DataFrame,
    mandate_profile: str = "moderate",
    penalty_weights: dict | None = None,
    return_column: str = "net_return",
    drawdown_column: str = "drawdown",
    turnover_column: str = "turnover",
    volatility_window: int = 12,
) -> pd.DataFrame:
    """Compute per-period mandate penalty observations from a strategy history."""
    _require_column(history, return_column)
    _require_column(history, drawdown_column)
    _require_column(history, turnover_column)
    weight_columns = infer_weight_columns(history)

    result = history.copy()
    weights = result.loc[:, weight_columns]
    _validate_numeric_frame(weights, "weights")
    _validate_numeric_series(result[return_column], return_column)
    _validate_numeric_series(result[drawdown_column], drawdown_column)
    _validate_numeric_series(result[turnover_column], turnover_column)

    result["max_weight"] = weights.max(axis=1)
    result["effective_assets"] = compute_effective_assets_from_weights(weights)
    result["mandate_drawdown"] = _as_negative_drawdown(result[drawdown_column])
    result["trailing_volatility"] = compute_trailing_volatility(
        result[return_column],
        window=volatility_window,
    )
    result = result.dropna(subset=["trailing_volatility", "effective_assets"]).copy()

    mandate_limits = get_mandate_limits(mandate_profile)
    penalty_rows = []
    for _, row in result.iterrows():
        penalty_result = compute_mandate_penalty(
            current_drawdown=float(row["mandate_drawdown"]),
            current_volatility=float(row["trailing_volatility"]),
            max_weight=float(row["max_weight"]),
            effective_assets=float(row["effective_assets"]),
            turnover=float(row[turnover_column]),
            mandate_limits=mandate_limits,
            penalty_weights=penalty_weights,
        )
        penalty_rows.append(
            {
                **penalty_result["breaches"],
                "mandate_penalty": penalty_result["penalty"],
            }
        )

    penalty_frame = pd.DataFrame(penalty_rows, index=result.index)
    result = pd.concat([result, penalty_frame], axis=1)
    result["mandate_profile"] = mandate_profile

    return result


def summarize_shadow_mandate_penalties(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize shadow mandate penalties and breach rates."""
    required_columns = [
        "mandate_penalty",
        "max_weight",
        "effective_assets",
        "trailing_volatility",
        "turnover",
        *BREACH_COLUMNS,
    ]
    missing_columns = [
        column for column in required_columns if column not in observations.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing observation columns: {missing_columns}")

    summary = {
        "n_observations": len(observations),
        "mean_mandate_penalty": observations["mandate_penalty"].mean(),
        "max_mandate_penalty": observations["mandate_penalty"].max(),
        "penalty_positive_rate": (observations["mandate_penalty"] > 0.0).mean(),
        "mean_drawdown_breach": observations["drawdown_breach"].mean(),
        "mean_volatility_breach": observations["volatility_breach"].mean(),
        "mean_max_weight_breach": observations["max_weight_breach"].mean(),
        "mean_effective_assets_breach": observations[
            "effective_assets_breach"
        ].mean(),
        "mean_turnover_breach": observations["turnover_breach"].mean(),
        "drawdown_breach_rate": (observations["drawdown_breach"] > 0.0).mean(),
        "volatility_breach_rate": (observations["volatility_breach"] > 0.0).mean(),
        "max_weight_breach_rate": (observations["max_weight_breach"] > 0.0).mean(),
        "effective_assets_breach_rate": (
            observations["effective_assets_breach"] > 0.0
        ).mean(),
        "turnover_breach_rate": (observations["turnover_breach"] > 0.0).mean(),
        "mean_max_weight": observations["max_weight"].mean(),
        "mean_effective_assets": observations["effective_assets"].mean(),
        "mean_trailing_volatility": observations["trailing_volatility"].mean(),
        "mean_turnover": observations["turnover"].mean(),
    }

    return pd.DataFrame([summary])


def build_shadow_mandate_penalty_report(
    history_paths: list[str],
    strategy_names: list[str] | None = None,
    mandate_profiles: list[str] | None = None,
    output_dir: str | None = None,
    return_column: str = "net_return",
    drawdown_column: str = "drawdown",
    turnover_column: str = "turnover",
    volatility_window: int = 12,
) -> dict:
    """Build a shadow mandate penalty report for histories and profiles."""
    if not history_paths:
        raise ValueError("history_paths must be non-empty.")
    if strategy_names is not None and len(strategy_names) != len(history_paths):
        raise ValueError("strategy_names must have the same length as history_paths.")

    selected_profiles = DEFAULT_MANDATE_PROFILES if mandate_profiles is None else mandate_profiles
    observations = []
    summaries = []
    for history_index, history_path in enumerate(history_paths):
        file_path = Path(history_path)
        if not file_path.exists():
            raise FileNotFoundError(f"History file not found: {history_path}")
        history = pd.read_csv(file_path)
        strategy_name = (
            strategy_names[history_index]
            if strategy_names is not None
            else _infer_strategy_name(file_path)
        )

        for mandate_profile in selected_profiles:
            profile_observations = build_shadow_mandate_penalty_observations(
                history=history,
                mandate_profile=mandate_profile,
                return_column=return_column,
                drawdown_column=drawdown_column,
                turnover_column=turnover_column,
                volatility_window=volatility_window,
            )
            profile_observations.insert(0, "strategy_name", strategy_name)
            profile_observations.insert(1, "history_path", str(file_path))
            observations.append(profile_observations)

            profile_summary = summarize_shadow_mandate_penalties(profile_observations)
            profile_summary.insert(0, "strategy_name", strategy_name)
            profile_summary.insert(1, "history_path", str(file_path))
            profile_summary.insert(2, "mandate_profile", mandate_profile)
            summaries.append(profile_summary)

    observations_frame = pd.concat(observations, ignore_index=True)
    summary_frame = pd.concat(summaries, ignore_index=True)

    observations_path = None
    summary_path = None
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        observations_path = str(destination / "shadow_mandate_penalty_observations.csv")
        summary_path = str(destination / "shadow_mandate_penalty_summary.csv")
        observations_frame.to_csv(observations_path, index=False)
        summary_frame.to_csv(summary_path, index=False)

    return {
        "observations": observations_frame,
        "summary": summary_frame,
        "observations_path": observations_path,
        "summary_path": summary_path,
    }


def _infer_strategy_name(file_path: Path) -> str:
    stem = file_path.stem
    if stem.endswith("_history"):
        return stem.removesuffix("_history")
    return file_path.parent.name


def _as_negative_drawdown(drawdown: pd.Series) -> pd.Series:
    """Normalize drawdown conventions to negative-or-zero for mandate penalties."""
    negative_drawdown = -drawdown.abs()
    if (negative_drawdown <= -1.0).any():
        raise ValueError("drawdown values must imply drawdowns greater than -1.")

    return negative_drawdown.rename("mandate_drawdown")


def _require_column(data: pd.DataFrame, column: str) -> None:
    if column not in data.columns:
        raise ValueError(f"Missing required column: {column}")


def _validate_numeric_frame(data: pd.DataFrame, name: str) -> None:
    non_numeric_columns = [
        column for column in data.columns if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"{name} columns must be numeric: {non_numeric_columns}")


def _validate_numeric_series(series: pd.Series, name: str) -> None:
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"{name} must be numeric.")
