"""Attribution of V5 high-CASH exposure to raw V5 risk-off state.

These helpers are pure ex-post diagnostics. They rebuild raw, unnormalized V5
features from a local returns snapshot and merge them with saved policy
histories to test whether high CASH allocation occurred during V5 risk-off
states.
"""

from pathlib import Path

import pandas as pd

from src.data.features_v5 import build_features_v5


REQUIRED_V5_REGIME_COLUMNS = (
    "regime_market_drawdown_stress",
    "regime_market_high_vol",
    "correlation_stress",
    "risk_off_score",
    "risk_off_state",
)


def load_policy_history(path: str) -> pd.DataFrame:
    """Load a saved policy history CSV with parsed, sorted dates."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Policy history file not found: {path}")
    policy_history = pd.read_csv(file_path)
    if "date" not in policy_history.columns:
        raise ValueError("Policy history must include a date column.")
    policy_history["date"] = pd.to_datetime(policy_history["date"], errors="coerce")
    policy_history = policy_history.dropna(subset=["date"]).sort_values("date")
    if policy_history.empty:
        raise ValueError("Policy history has no usable dated rows.")

    return policy_history.reset_index(drop=True)


def build_raw_v5_features_for_returns(
    returns_path: str,
    config_path: str | None = None,
) -> pd.DataFrame:
    """Build raw V5 features directly from a local returns CSV.

    config_path is accepted for future compatibility, but this first diagnostic
    uses the standard V5 defaults so the resulting regime flags remain raw and
    interpretable.
    """
    del config_path
    file_path = Path(returns_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Returns file not found: {returns_path}")
    returns = pd.read_csv(file_path)
    if "date" not in returns.columns:
        raise ValueError("Returns file must include a date column.")
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    returns = returns.dropna(subset=["date"]).sort_values("date")
    if returns.empty:
        raise ValueError("Returns file has no usable dated rows.")
    returns = returns.set_index("date")
    returns = returns.apply(pd.to_numeric, errors="coerce")
    if returns.isna().any().any():
        raise ValueError("Returns file contains missing or non-numeric returns.")

    return build_features_v5(returns)


def merge_policy_with_v5_regime(
    policy_history: pd.DataFrame,
    raw_v5_features: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """Merge policy history with raw V5 regime columns by exact date."""
    if date_column not in policy_history.columns:
        raise ValueError(f"Policy history must include a {date_column} column.")
    if not isinstance(raw_v5_features.index, pd.DatetimeIndex):
        raise TypeError("raw_v5_features index must be a DatetimeIndex.")
    missing_columns = [
        column for column in REQUIRED_V5_REGIME_COLUMNS if column not in raw_v5_features.columns
    ]
    if missing_columns:
        raise ValueError(f"Raw V5 features missing required columns: {missing_columns}")

    policy = policy_history.copy()
    policy[date_column] = pd.to_datetime(policy[date_column], errors="coerce")
    policy = policy.dropna(subset=[date_column])
    features = raw_v5_features.loc[:, REQUIRED_V5_REGIME_COLUMNS].copy()
    features = features.loc[~features.index.duplicated(keep="last")]
    features[date_column] = features.index
    features = features.reset_index(drop=True)
    merged = policy.merge(features, how="inner", on=date_column)
    if merged.empty:
        raise ValueError("No overlapping dates between policy history and V5 features.")

    return merged


def add_cash_risk_off_attribution(
    merged: pd.DataFrame,
    normal_cash_max: float = 0.10,
    cash_column: str = "weight_CASH",
) -> pd.DataFrame:
    """Add high-CASH and V5 risk-off attribution fields."""
    _validate_normal_cash_max(normal_cash_max)
    if cash_column not in merged.columns:
        raise ValueError(f"Missing cash weight column: {cash_column}")
    if (
        not pd.api.types.is_numeric_dtype(merged[cash_column])
        or pd.api.types.is_bool_dtype(merged[cash_column])
    ):
        raise ValueError(f"Cash weight column must be numeric: {cash_column}")
    if "risk_off_state" not in merged.columns:
        raise ValueError("Merged data must include risk_off_state.")

    result = merged.copy()
    result["cash_weight"] = result[cash_column]
    risk_off_state = result["risk_off_state"].astype(float) == 1.0
    result["high_cash"] = result["cash_weight"] > normal_cash_max
    result["high_cash_and_risk_off"] = result["high_cash"] & risk_off_state
    result["high_cash_without_risk_off"] = result["high_cash"] & ~risk_off_state
    result["cash_excess_normal"] = (
        result["cash_weight"] - normal_cash_max
    ).clip(lower=0.0)
    result["unjustified_cash_excess"] = result["cash_excess_normal"].where(
        ~risk_off_state,
        0.0,
    )
    result["risk_off_justified_cash_excess"] = result["cash_excess_normal"].where(
        risk_off_state,
        0.0,
    )

    return result


def summarize_cash_risk_off_attribution(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether high CASH exposure coincided with V5 risk-off state."""
    _require_attribution_columns(data)
    high_cash_count = int(data["high_cash"].sum())
    high_cash_in_risk_off = int(data["high_cash_and_risk_off"].sum())
    share_high_cash_in_risk_off = (
        high_cash_in_risk_off / high_cash_count if high_cash_count > 0 else 0.0
    )
    risk_off_mask = data["risk_off_state"].astype(float) == 1.0
    high_cash_mask = data["high_cash"]

    return pd.DataFrame(
        [
            {
                "n_observations": len(data),
                "mean_cash_weight": data["cash_weight"].mean(),
                "max_cash_weight": data["cash_weight"].max(),
                "high_cash_rate": data["high_cash"].mean(),
                "risk_off_rate": risk_off_mask.mean(),
                "high_cash_justified_rate": data["high_cash_and_risk_off"].mean(),
                "high_cash_unjustified_rate": data[
                    "high_cash_without_risk_off"
                ].mean(),
                "share_high_cash_observations_in_risk_off": share_high_cash_in_risk_off,
                "mean_unjustified_cash_excess": data[
                    "unjustified_cash_excess"
                ].mean(),
                "mean_risk_off_justified_cash_excess": data[
                    "risk_off_justified_cash_excess"
                ].mean(),
                "mean_cash_when_risk_off": data.loc[
                    risk_off_mask,
                    "cash_weight",
                ].mean(),
                "mean_cash_when_not_risk_off": data.loc[
                    ~risk_off_mask,
                    "cash_weight",
                ].mean(),
                "mean_risk_off_score_when_high_cash": data.loc[
                    high_cash_mask,
                    "risk_off_score",
                ].mean(),
                "mean_risk_off_score_when_low_cash": data.loc[
                    ~high_cash_mask,
                    "risk_off_score",
                ].mean(),
            }
        ]
    )


def summarize_by_risk_off_state(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize cash behavior conditional on V5 risk_off_state."""
    _require_attribution_columns(data)

    return (
        data.assign(risk_off_state=data["risk_off_state"].astype(float))
        .groupby("risk_off_state", as_index=False)
        .agg(
            n_observations=("risk_off_state", "size"),
            mean_cash_weight=("cash_weight", "mean"),
            max_cash_weight=("cash_weight", "max"),
            high_cash_rate=("high_cash", "mean"),
            mean_cash_excess_normal=("cash_excess_normal", "mean"),
            mean_risk_off_score=("risk_off_score", "mean"),
            mean_regime_market_drawdown_stress=(
                "regime_market_drawdown_stress",
                "mean",
            ),
            mean_regime_market_high_vol=("regime_market_high_vol", "mean"),
            mean_correlation_stress=("correlation_stress", "mean"),
        )
        .sort_values("risk_off_state")
        .reset_index(drop=True)
    )


def build_v5_cash_risk_off_report(
    policy_history_paths: list[str],
    returns_path: str,
    strategy_names: list[str] | None = None,
    output_dir: str | None = None,
    normal_cash_max: float = 0.10,
) -> dict:
    """Build V5 CASH/risk-off attribution reports for policy histories."""
    if not policy_history_paths:
        raise ValueError("policy_history_paths must be non-empty.")
    if strategy_names is not None and len(strategy_names) != len(policy_history_paths):
        raise ValueError("strategy_names must have the same length as policy_history_paths.")
    _validate_normal_cash_max(normal_cash_max)
    raw_v5_features = build_raw_v5_features_for_returns(returns_path)

    observation_frames = []
    summary_frames = []
    by_state_frames = []
    for index, path in enumerate(policy_history_paths):
        strategy_name = (
            strategy_names[index]
            if strategy_names is not None
            else _infer_strategy_name(Path(path))
        )
        policy_history = load_policy_history(path)
        merged = merge_policy_with_v5_regime(policy_history, raw_v5_features)
        observations = add_cash_risk_off_attribution(
            merged,
            normal_cash_max=normal_cash_max,
        )
        observations.insert(0, "strategy_name", strategy_name)
        observations.insert(1, "policy_history_path", path)
        observation_frames.append(observations)

        summary = summarize_cash_risk_off_attribution(observations)
        summary.insert(0, "strategy_name", strategy_name)
        summary.insert(1, "policy_history_path", path)
        summary_frames.append(summary)

        by_state = summarize_by_risk_off_state(observations)
        by_state.insert(0, "strategy_name", strategy_name)
        by_state.insert(1, "policy_history_path", path)
        by_state_frames.append(by_state)

    observations_frame = pd.concat(observation_frames, ignore_index=True)
    summary_frame = pd.concat(summary_frames, ignore_index=True)
    by_state_frame = pd.concat(by_state_frames, ignore_index=True)

    observations_path = None
    summary_path = None
    by_state_path = None
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        observations_path = str(destination / "v5_cash_risk_off_observations.csv")
        summary_path = str(destination / "v5_cash_risk_off_summary.csv")
        by_state_path = str(destination / "v5_cash_risk_off_by_state.csv")
        observations_frame.to_csv(observations_path, index=False)
        summary_frame.to_csv(summary_path, index=False)
        by_state_frame.to_csv(by_state_path, index=False)

    return {
        "observations": observations_frame,
        "summary": summary_frame,
        "by_risk_off_state": by_state_frame,
        "observations_path": observations_path,
        "summary_path": summary_path,
        "by_risk_off_state_path": by_state_path,
    }


def _validate_normal_cash_max(normal_cash_max: float) -> None:
    if (
        isinstance(normal_cash_max, bool)
        or not isinstance(normal_cash_max, (int, float))
        or normal_cash_max < 0.0
        or normal_cash_max > 1.0
    ):
        raise ValueError("normal_cash_max must be numeric and in the range [0, 1].")


def _require_attribution_columns(data: pd.DataFrame) -> None:
    required_columns = [
        "cash_weight",
        "high_cash",
        "high_cash_and_risk_off",
        "high_cash_without_risk_off",
        "cash_excess_normal",
        "unjustified_cash_excess",
        "risk_off_justified_cash_excess",
        *REQUIRED_V5_REGIME_COLUMNS,
    ]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required attribution columns: {missing_columns}")


def _infer_strategy_name(path: Path) -> str:
    if path.parent.name:
        return path.parent.name

    return path.stem
