"""Ex-post diagnostics for whether concentration was useful.

These helpers evaluate concentrated policy choices against subsequent realized
asset returns. They are descriptive diagnostics only; they do not change the
reward, environment, training loop, or portfolio constraints.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def infer_weight_columns(data: pd.DataFrame) -> list[str]:
    """Return valid numeric asset-weight columns."""
    weight_columns = [
        column
        for column in data.columns
        if str(column).startswith("weight_")
        and str(column) != "weight_sum"
        and not str(column).endswith("_ok")
        and pd.api.types.is_numeric_dtype(data[column])
        and not pd.api.types.is_bool_dtype(data[column])
    ]
    if not weight_columns:
        raise ValueError("No valid weight_ columns found.")

    return weight_columns


def add_dominant_asset_state(data: pd.DataFrame) -> pd.DataFrame:
    """Add dominant-asset and concentration state columns to a policy history."""
    weight_columns = infer_weight_columns(data)
    weights = data.loc[:, weight_columns]
    if (weights.abs().sum(axis=1) == 0.0).any():
        raise ValueError("Weight rows must not have total absolute weight equal to 0.")

    result = data.copy()
    result["weight_sum"] = weights.sum(axis=1)
    dominant_columns = weights.idxmax(axis=1)
    result["dominant_asset"] = dominant_columns.str.removeprefix("weight_")
    result["dominant_weight"] = weights.max(axis=1)
    result["herfindahl_index"] = (weights**2).sum(axis=1)
    result["effective_number_of_assets"] = 1.0 / result[
        "herfindahl_index"
    ].replace(0.0, np.nan)

    return result


def compute_forward_asset_returns(
    returns: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """Compute next-horizon compounded asset returns for each decision date."""
    _validate_returns_frame(returns)
    _validate_horizon(horizon)

    compounded_through_current = (1.0 + returns).rolling(horizon).apply(
        np.prod,
        raw=True,
    ) - 1.0

    return compounded_through_current.shift(-horizon)


def attach_forward_dominant_asset_performance(
    policy_history: pd.DataFrame,
    asset_returns: pd.DataFrame,
    horizon: int = 1,
    date_column: str = "date",
) -> pd.DataFrame:
    """Attach forward realized performance of the dominant asset choice."""
    _validate_horizon(horizon)
    _validate_returns_frame(asset_returns)
    observations = add_dominant_asset_state(policy_history)
    policy = _prepare_policy_dates(observations, date_column)

    assets = [column.removeprefix("weight_") for column in infer_weight_columns(policy)]
    missing_assets = [asset for asset in assets if asset not in asset_returns.columns]
    if missing_assets:
        raise ValueError(f"Missing asset return columns: {missing_assets}")

    forward_returns = compute_forward_asset_returns(
        asset_returns.loc[:, assets],
        horizon=horizon,
    )
    forward_columns = {
        asset: f"forward_return_{asset}"
        for asset in forward_returns.columns
    }
    forward_returns = forward_returns.rename(columns=forward_columns)
    forward_returns = forward_returns.loc[
        ~forward_returns.index.duplicated(keep="last")
    ].copy()
    forward_returns[date_column] = forward_returns.index

    merged = policy.merge(forward_returns, how="left", on=date_column)
    forward_column_names = list(forward_columns.values())
    merged = merged.dropna(subset=forward_column_names).copy()

    if merged.empty:
        return merged

    forward_matrix = merged.loc[:, forward_column_names]
    dominant_forward_returns = []
    dominant_ranks = []
    for row_index, row in merged.iterrows():
        dominant_column = f"forward_return_{row['dominant_asset']}"
        dominant_return = row[dominant_column]
        descending_returns = (
            forward_matrix.loc[row_index]
            .sort_values(ascending=False, kind="mergesort")
        )
        dominant_ranks.append(
            int(descending_returns.index.get_loc(dominant_column)) + 1
        )
        dominant_forward_returns.append(dominant_return)

    merged["dominant_forward_return"] = dominant_forward_returns
    merged["best_asset_forward_return"] = forward_matrix.max(axis=1)
    merged["worst_asset_forward_return"] = forward_matrix.min(axis=1)
    merged["equal_weight_forward_return"] = forward_matrix.mean(axis=1)
    merged["dominant_asset_rank"] = dominant_ranks
    merged["dominant_is_best_asset"] = (
        merged["dominant_forward_return"] == merged["best_asset_forward_return"]
    )
    merged["dominant_beats_equal_weight"] = (
        merged["dominant_forward_return"] > merged["equal_weight_forward_return"]
    )
    merged["dominant_forward_excess_vs_equal_weight"] = (
        merged["dominant_forward_return"] - merged["equal_weight_forward_return"]
    )
    merged["dominant_forward_excess_vs_best_asset"] = (
        merged["dominant_forward_return"] - merged["best_asset_forward_return"]
    )

    return merged


def summarize_concentration_quality(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether concentrated choices were justified ex post."""
    _require_quality_columns(data)
    summary = {
        "n_observations": len(data),
        "mean_dominant_weight": data["dominant_weight"].mean(),
        "mean_effective_number_of_assets": data[
            "effective_number_of_assets"
        ].mean(),
        "dominant_best_asset_rate": data["dominant_is_best_asset"].mean(),
        "dominant_beats_equal_weight_rate": data[
            "dominant_beats_equal_weight"
        ].mean(),
        "mean_dominant_forward_return": data["dominant_forward_return"].mean(),
        "mean_equal_weight_forward_return": data[
            "equal_weight_forward_return"
        ].mean(),
        "mean_best_asset_forward_return": data["best_asset_forward_return"].mean(),
        "mean_dominant_forward_excess_vs_equal_weight": data[
            "dominant_forward_excess_vs_equal_weight"
        ].mean(),
        "mean_dominant_forward_excess_vs_best_asset": data[
            "dominant_forward_excess_vs_best_asset"
        ].mean(),
        "mean_dominant_asset_rank": data["dominant_asset_rank"].mean(),
    }

    return pd.DataFrame([summary])


def summarize_concentration_quality_by_asset(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize concentration quality conditional on dominant asset."""
    _require_quality_columns(data)

    return (
        data.groupby("dominant_asset", as_index=False)
        .agg(
            n_observations=("dominant_asset", "size"),
            mean_dominant_weight=("dominant_weight", "mean"),
            mean_effective_number_of_assets=(
                "effective_number_of_assets",
                "mean",
            ),
            dominant_best_asset_rate=("dominant_is_best_asset", "mean"),
            dominant_beats_equal_weight_rate=(
                "dominant_beats_equal_weight",
                "mean",
            ),
            mean_dominant_forward_return=("dominant_forward_return", "mean"),
            mean_dominant_forward_excess_vs_equal_weight=(
                "dominant_forward_excess_vs_equal_weight",
                "mean",
            ),
            mean_dominant_asset_rank=("dominant_asset_rank", "mean"),
        )
        .sort_values(["n_observations", "dominant_asset"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_concentration_quality_report(
    policy_history_paths: list[str],
    asset_returns_path: str,
    strategy_names: list[str] | None = None,
    horizons: list[int] | None = None,
    output_dir: str | None = None,
) -> dict:
    """Build concentration quality reports for one or more policy histories."""
    if not policy_history_paths:
        raise ValueError("policy_history_paths must be non-empty.")
    if strategy_names is not None and len(strategy_names) != len(policy_history_paths):
        raise ValueError("strategy_names must have the same length as policy_history_paths.")

    selected_horizons = [1, 4, 12] if horizons is None else horizons
    for horizon in selected_horizons:
        _validate_horizon(horizon)

    asset_returns = _load_asset_returns(asset_returns_path)
    observation_frames = []
    summary_frames = []
    by_asset_frames = []

    for history_index, history_path in enumerate(policy_history_paths):
        file_path = Path(history_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Policy history file not found: {history_path}")
        policy_history = pd.read_csv(file_path)
        strategy_name = (
            strategy_names[history_index]
            if strategy_names is not None
            else _infer_strategy_name(file_path)
        )

        for horizon in selected_horizons:
            observations = attach_forward_dominant_asset_performance(
                policy_history,
                asset_returns,
                horizon=horizon,
            )
            observations.insert(0, "strategy_name", strategy_name)
            observations.insert(1, "policy_history_path", str(file_path))
            observations.insert(2, "horizon", horizon)
            observation_frames.append(observations)

            summary = summarize_concentration_quality(observations)
            summary.insert(0, "strategy_name", strategy_name)
            summary.insert(1, "policy_history_path", str(file_path))
            summary.insert(2, "horizon", horizon)
            summary_frames.append(summary)

            by_asset_summary = summarize_concentration_quality_by_asset(observations)
            by_asset_summary.insert(0, "strategy_name", strategy_name)
            by_asset_summary.insert(1, "policy_history_path", str(file_path))
            by_asset_summary.insert(2, "horizon", horizon)
            by_asset_frames.append(by_asset_summary)

    observations_frame = pd.concat(observation_frames, ignore_index=True)
    summary_frame = pd.concat(summary_frames, ignore_index=True)
    by_asset_frame = pd.concat(by_asset_frames, ignore_index=True)

    observations_path = None
    summary_path = None
    by_asset_summary_path = None
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        observations_path = str(destination / "concentration_quality_observations.csv")
        summary_path = str(destination / "concentration_quality_summary.csv")
        by_asset_summary_path = str(
            destination / "concentration_quality_by_asset_summary.csv"
        )
        observations_frame.to_csv(observations_path, index=False)
        summary_frame.to_csv(summary_path, index=False)
        by_asset_frame.to_csv(by_asset_summary_path, index=False)

    return {
        "observations": observations_frame,
        "summary": summary_frame,
        "by_asset_summary": by_asset_frame,
        "observations_path": observations_path,
        "summary_path": summary_path,
        "by_asset_summary_path": by_asset_summary_path,
    }


def _prepare_policy_dates(
    policy_history: pd.DataFrame,
    date_column: str,
) -> pd.DataFrame:
    policy = policy_history.copy()
    if date_column in policy.columns:
        policy[date_column] = pd.to_datetime(policy[date_column], errors="coerce")
    elif isinstance(policy.index, pd.DatetimeIndex):
        policy[date_column] = policy.index
    else:
        raise ValueError(
            f"policy_history must include {date_column!r} or use a DatetimeIndex."
        )

    policy = policy.dropna(subset=[date_column])
    if policy.empty:
        raise ValueError("policy_history has no usable dates.")

    return policy


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

    returns = data.apply(pd.to_numeric, errors="coerce")
    returns = returns.dropna(how="all")
    if returns.empty:
        raise ValueError("Asset returns CSV has no usable return rows.")

    return returns


def _validate_returns_frame(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns must be non-empty.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex.")
    non_numeric_columns = [
        column
        for column in returns.columns
        if not pd.api.types.is_numeric_dtype(returns[column])
        or pd.api.types.is_bool_dtype(returns[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"returns columns must be numeric: {non_numeric_columns}")


def _validate_horizon(horizon: int) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be an integer greater than or equal to 1.")


def _require_quality_columns(data: pd.DataFrame) -> None:
    required_columns = [
        "dominant_asset",
        "dominant_weight",
        "effective_number_of_assets",
        "dominant_is_best_asset",
        "dominant_beats_equal_weight",
        "dominant_forward_return",
        "equal_weight_forward_return",
        "best_asset_forward_return",
        "dominant_forward_excess_vs_equal_weight",
        "dominant_forward_excess_vs_best_asset",
        "dominant_asset_rank",
    ]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing concentration quality columns: {missing_columns}")


def _infer_strategy_name(path: Path) -> str:
    if path.stem.endswith("_policy_history"):
        return path.stem.removesuffix("_policy_history")

    return path.parent.name or path.stem
