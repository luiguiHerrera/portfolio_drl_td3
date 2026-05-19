"""Diagnostics for whether high CASH allocation was defensively justified.

These functions are pure ex-post analysis helpers. They distinguish ordinary
cash bands from high cash during risk-off states and measure the forward
opportunity cost of holding cash.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.concentration_quality_diagnostics import (
    compute_forward_asset_returns,
)


def infer_cash_weight_column(
    data: pd.DataFrame,
    cash_asset: str = "CASH",
) -> str:
    """Return the cash weight column name after validating it is numeric."""
    column = f"weight_{cash_asset}"
    if column not in data.columns:
        raise ValueError(f"Missing cash weight column: {column}")
    if (
        not pd.api.types.is_numeric_dtype(data[column])
        or pd.api.types.is_bool_dtype(data[column])
    ):
        raise ValueError(f"Cash weight column must be numeric: {column}")

    return column


def compute_risk_off_score(
    data: pd.DataFrame,
    signal_columns: list[str],
) -> pd.Series:
    """Compute row-wise risk-off score from selected signal columns."""
    if not signal_columns:
        raise ValueError("signal_columns must be non-empty.")
    missing_columns = [column for column in signal_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing risk-off signal columns: {missing_columns}")
    non_numeric_columns = [
        column
        for column in signal_columns
        if not (
            pd.api.types.is_numeric_dtype(data[column])
            or pd.api.types.is_bool_dtype(data[column])
        )
    ]
    if non_numeric_columns:
        raise ValueError(f"Risk-off signal columns must be numeric or bool: {non_numeric_columns}")

    return data.loc[:, signal_columns].astype(float).sum(axis=1).rename("risk_off_score")


def add_cash_state_diagnostics(
    data: pd.DataFrame,
    normal_cash_max: float = 0.10,
    risk_off_score_column: str | None = None,
    risk_off_threshold: float = 1.0,
    cash_asset: str = "CASH",
) -> pd.DataFrame:
    """Add cash band and risk-off justification diagnostics."""
    _validate_normal_cash_max(normal_cash_max)
    _validate_risk_off_threshold(risk_off_threshold)
    cash_column = infer_cash_weight_column(data, cash_asset)
    if risk_off_score_column is not None and risk_off_score_column not in data.columns:
        raise ValueError(f"Missing risk-off score column: {risk_off_score_column}")

    result = data.copy()
    result["cash_weight"] = result[cash_column]
    result["cash_above_normal_max"] = result["cash_weight"] > normal_cash_max
    result["cash_excess_normal"] = (
        result["cash_weight"] - normal_cash_max
    ).clip(lower=0.0)
    if risk_off_score_column is None:
        result["risk_off_state"] = False
    else:
        if (
            not pd.api.types.is_numeric_dtype(result[risk_off_score_column])
            and not pd.api.types.is_bool_dtype(result[risk_off_score_column])
        ):
            raise ValueError(
                f"Risk-off score column must be numeric or bool: {risk_off_score_column}"
            )
        result["risk_off_state"] = (
            result[risk_off_score_column].astype(float) >= risk_off_threshold
        )

    result["cash_allowed_by_state"] = (
        result["cash_weight"] <= normal_cash_max
    ) | result["risk_off_state"]
    result["unjustified_cash_excess"] = result["cash_excess_normal"].where(
        ~result["risk_off_state"],
        0.0,
    )

    return result


def attach_forward_cash_opportunity_cost(
    data: pd.DataFrame,
    asset_returns: pd.DataFrame,
    horizon: int = 1,
    date_column: str = "date",
    cash_asset: str = "CASH",
) -> pd.DataFrame:
    """Attach forward cash return and opportunity-cost fields."""
    infer_cash_weight_column(data, cash_asset)
    _validate_returns(asset_returns)
    if cash_asset not in asset_returns.columns:
        raise ValueError(f"Missing cash asset returns column: {cash_asset}")
    risky_assets = [asset for asset in asset_returns.columns if asset != cash_asset]
    if not risky_assets:
        raise ValueError("asset_returns must contain at least one risky asset.")

    policy = _prepare_dates(data, date_column)
    forward_returns = compute_forward_asset_returns(asset_returns, horizon=horizon)
    forward_returns = forward_returns.loc[
        ~forward_returns.index.duplicated(keep="last")
    ].copy()
    forward_returns[date_column] = forward_returns.index
    forward_columns = {
        asset: f"forward_return_{asset}"
        for asset in asset_returns.columns
    }
    forward_returns = forward_returns.rename(columns=forward_columns)

    merged = policy.merge(forward_returns, how="left", on=date_column)
    needed_columns = list(forward_columns.values())
    merged = merged.dropna(subset=needed_columns).copy()
    if merged.empty:
        return merged

    cash_forward_column = forward_columns[cash_asset]
    risky_forward_columns = [forward_columns[asset] for asset in risky_assets]
    all_forward_columns = [forward_columns[asset] for asset in asset_returns.columns]

    merged["cash_forward_return"] = merged[cash_forward_column]
    merged["equal_weight_forward_return"] = merged.loc[:, all_forward_columns].mean(axis=1)
    merged["best_risky_asset_forward_return"] = merged.loc[
        :,
        risky_forward_columns,
    ].max(axis=1)
    merged["cash_excess_vs_equal_weight"] = (
        merged["cash_forward_return"] - merged["equal_weight_forward_return"]
    )
    merged["cash_excess_vs_best_risky_asset"] = (
        merged["cash_forward_return"] - merged["best_risky_asset_forward_return"]
    )
    merged["cash_underperforms_equal_weight"] = (
        merged["cash_forward_return"] < merged["equal_weight_forward_return"]
    )
    merged["cash_underperforms_best_risky_asset"] = (
        merged["cash_forward_return"] < merged["best_risky_asset_forward_return"]
    )

    return merged


def summarize_cash_allocation_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize high-cash behavior and its forward opportunity cost."""
    _require_summary_columns(data)
    summary = {
        "n_observations": len(data),
        "mean_cash_weight": data["cash_weight"].mean(),
        "max_cash_weight": data["cash_weight"].max(),
        "cash_above_normal_rate": data["cash_above_normal_max"].mean(),
        "risk_off_rate": data["risk_off_state"].mean(),
        "cash_allowed_rate": data["cash_allowed_by_state"].mean(),
        "unjustified_cash_rate": (data["unjustified_cash_excess"] > 0.0).mean(),
        "mean_unjustified_cash_excess": data["unjustified_cash_excess"].mean(),
        "mean_cash_excess_normal": data["cash_excess_normal"].mean(),
        "mean_cash_forward_return": data["cash_forward_return"].mean(),
        "mean_equal_weight_forward_return": data[
            "equal_weight_forward_return"
        ].mean(),
        "mean_best_risky_asset_forward_return": data[
            "best_risky_asset_forward_return"
        ].mean(),
        "mean_cash_excess_vs_equal_weight": data[
            "cash_excess_vs_equal_weight"
        ].mean(),
        "mean_cash_excess_vs_best_risky_asset": data[
            "cash_excess_vs_best_risky_asset"
        ].mean(),
        "cash_underperforms_equal_weight_rate": data[
            "cash_underperforms_equal_weight"
        ].mean(),
        "cash_underperforms_best_risky_asset_rate": data[
            "cash_underperforms_best_risky_asset"
        ].mean(),
    }

    return pd.DataFrame([summary])


def build_cash_allocation_report(
    policy_history_paths: list[str],
    asset_returns_path: str,
    strategy_names: list[str] | None = None,
    horizons: list[int] | None = None,
    normal_cash_max: float = 0.10,
    risk_off_signal_paths: list[str] | None = None,
    risk_off_signal_columns: list[str] | None = None,
    risk_off_threshold: float = 1.0,
    output_dir: str | None = None,
) -> dict:
    """Build cash allocation diagnostics for policy histories."""
    if not policy_history_paths:
        raise ValueError("policy_history_paths must be non-empty.")
    if strategy_names is not None and len(strategy_names) != len(policy_history_paths):
        raise ValueError("strategy_names must have the same length as policy_history_paths.")

    selected_horizons = [1, 4, 12] if horizons is None else horizons
    _validate_normal_cash_max(normal_cash_max)
    asset_returns = _load_asset_returns(asset_returns_path)
    signal_frame = _load_risk_off_signals(
        risk_off_signal_paths,
        risk_off_signal_columns,
    )

    observations = []
    summaries = []
    for index, policy_history_path in enumerate(policy_history_paths):
        file_path = Path(policy_history_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Policy history file not found: {policy_history_path}")
        policy_history = pd.read_csv(file_path)
        strategy_name = (
            strategy_names[index]
            if strategy_names is not None
            else _infer_strategy_name(file_path)
        )
        policy_with_signals = _merge_signals(policy_history, signal_frame)
        if signal_frame is not None:
            policy_with_signals["risk_off_score"] = compute_risk_off_score(
                policy_with_signals,
                risk_off_signal_columns or [],
            )

        for horizon in selected_horizons:
            cash_state = add_cash_state_diagnostics(
                policy_with_signals,
                normal_cash_max=normal_cash_max,
                risk_off_score_column=(
                    "risk_off_score" if signal_frame is not None else None
                ),
                risk_off_threshold=risk_off_threshold,
            )
            cash_observations = attach_forward_cash_opportunity_cost(
                cash_state,
                asset_returns,
                horizon=horizon,
            )
            cash_observations.insert(0, "strategy_name", strategy_name)
            cash_observations.insert(1, "policy_history_path", str(file_path))
            cash_observations.insert(2, "horizon", horizon)
            observations.append(cash_observations)

            summary = summarize_cash_allocation_diagnostics(cash_observations)
            summary.insert(0, "strategy_name", strategy_name)
            summary.insert(1, "policy_history_path", str(file_path))
            summary.insert(2, "horizon", horizon)
            summaries.append(summary)

    observations_frame = pd.concat(observations, ignore_index=True)
    summary_frame = pd.concat(summaries, ignore_index=True)
    observations_path = None
    summary_path = None
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        observations_path = str(destination / "cash_allocation_observations.csv")
        summary_path = str(destination / "cash_allocation_summary.csv")
        observations_frame.to_csv(observations_path, index=False)
        summary_frame.to_csv(summary_path, index=False)

    return {
        "observations": observations_frame,
        "summary": summary_frame,
        "observations_path": observations_path,
        "summary_path": summary_path,
    }


def _prepare_dates(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    result = data.copy()
    if date_column not in result.columns:
        raise ValueError(f"Missing date column: {date_column}")
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
    result = result.dropna(subset=[date_column])
    if result.empty:
        raise ValueError("data has no usable dates.")

    return result


def _load_asset_returns(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Asset returns file not found: {path}")
    data = pd.read_csv(file_path)
    if "date" not in data.columns:
        raise KeyError("Asset returns CSV must include a date column.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    data = data.sort_values("date")
    data = data.drop_duplicates(subset=["date"], keep="last")
    data = data.set_index("date")
    data.index.name = None
    returns = data.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if returns.empty:
        raise ValueError("Asset returns CSV has no usable return rows.")

    return returns


def _load_risk_off_signals(
    risk_off_signal_paths: list[str] | None,
    risk_off_signal_columns: list[str] | None,
) -> pd.DataFrame | None:
    if risk_off_signal_paths is None:
        return None
    if not risk_off_signal_paths:
        raise ValueError("risk_off_signal_paths must be non-empty when provided.")
    if not risk_off_signal_columns:
        raise ValueError("risk_off_signal_columns must be provided with signal paths.")

    frames = []
    for path in risk_off_signal_paths:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Risk-off signal file not found: {path}")
        frame = pd.read_csv(file_path)
        if "date" not in frame.columns:
            raise KeyError("Risk-off signal CSV must include a date column.")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"])
        frames.append(frame)

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, how="outer", on="date")
    combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    return combined


def _merge_signals(
    policy_history: pd.DataFrame,
    signal_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if signal_frame is None:
        return policy_history.copy()

    policy = _prepare_dates(policy_history, "date")

    return policy.merge(signal_frame, how="left", on="date")


def _validate_returns(asset_returns: pd.DataFrame) -> None:
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame.")
    if asset_returns.empty:
        raise ValueError("asset_returns must be non-empty.")
    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise TypeError("asset_returns index must be a DatetimeIndex.")
    non_numeric_columns = [
        column
        for column in asset_returns.columns
        if not pd.api.types.is_numeric_dtype(asset_returns[column])
        or pd.api.types.is_bool_dtype(asset_returns[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"asset_returns columns must be numeric: {non_numeric_columns}")


def _validate_normal_cash_max(normal_cash_max: float) -> None:
    if (
        isinstance(normal_cash_max, bool)
        or not isinstance(normal_cash_max, (int, float))
        or normal_cash_max < 0.0
        or normal_cash_max > 1.0
    ):
        raise ValueError("normal_cash_max must be numeric and between 0 and 1.")


def _validate_risk_off_threshold(risk_off_threshold: float) -> None:
    if isinstance(risk_off_threshold, bool) or not isinstance(
        risk_off_threshold,
        (int, float),
    ):
        raise ValueError("risk_off_threshold must be numeric.")


def _require_summary_columns(data: pd.DataFrame) -> None:
    required_columns = [
        "cash_weight",
        "cash_above_normal_max",
        "cash_excess_normal",
        "risk_off_state",
        "cash_allowed_by_state",
        "unjustified_cash_excess",
        "cash_forward_return",
        "equal_weight_forward_return",
        "best_risky_asset_forward_return",
        "cash_excess_vs_equal_weight",
        "cash_excess_vs_best_risky_asset",
        "cash_underperforms_equal_weight",
        "cash_underperforms_best_risky_asset",
    ]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing cash diagnostic columns: {missing_columns}")


def _infer_strategy_name(path: Path) -> str:
    if path.stem.endswith("_policy_history"):
        return path.stem.removesuffix("_policy_history")

    return path.parent.name or path.stem
