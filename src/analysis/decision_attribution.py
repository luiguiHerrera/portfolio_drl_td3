"""Decision attribution diagnostics for TD3 policies versus simple rules.

These diagnostics are ex-post only. They compare saved policy choices against
future realized asset returns and transparent dynamic allocation rules without
changing training, reward, or environment behavior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.dynamic_allocation_benchmarks import (
    build_defensive_risk_off_weights,
    build_momentum_winner_weights,
    build_risk_adjusted_momentum_winner_weights,
    build_trend_following_spy_cash_weights,
)
from src.analysis.v5_cash_risk_off_attribution import build_raw_v5_features_for_returns


DEFAULT_RULES = (
    "momentum_winner_12p",
    "risk_adjusted_momentum_winner_12p_12p",
    "trend_spy_cash_12p",
    "defensive_risk_off_12p",
)

DEFAULT_REGIME_COLUMNS = (
    "risk_off_state",
    "correlation_stress",
    "regime_market_high_vol",
    "market_trend_regime",
)


def dominant_asset_from_policy(policy_history: pd.DataFrame) -> pd.DataFrame:
    """Extract dominant asset and dominant weight from policy weight columns."""
    if "date" not in policy_history.columns:
        raise ValueError("policy_history must include a date column.")
    weight_columns = _infer_weight_columns(policy_history)
    policy = policy_history.copy()
    policy["date"] = pd.to_datetime(policy["date"], errors="coerce")
    policy = policy.dropna(subset=["date"]).sort_values("date")
    if policy.empty:
        raise ValueError("policy_history has no usable dated rows.")

    weights = policy.loc[:, weight_columns]
    dominant_columns = weights.idxmax(axis=1)
    return pd.DataFrame(
        {
            "date": policy["date"].to_numpy(),
            "dominant_asset": dominant_columns.str.removeprefix("weight_").to_numpy(),
            "dominant_weight": weights.max(axis=1).to_numpy(dtype=float),
        },
    )


def compute_forward_asset_returns(
    returns: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute forward compounded asset returns for each horizon.

    Horizon 1 at date t is return at t+1, so the current-period return is not
    included in the forward diagnostic window.
    """
    returns_frame = _prepare_returns(returns)
    selected_horizons = [1, 4, 12] if horizons is None else horizons
    rows = []
    for horizon in selected_horizons:
        _validate_horizon(horizon)
        forward = _forward_returns_for_horizon(returns_frame, horizon)
        renamed = forward.add_prefix("forward_return_")
        frame = renamed.copy()
        frame.insert(0, "date", frame.index)
        frame.insert(1, "horizon", horizon)
        rows.append(frame.reset_index(drop=True))

    return pd.concat(rows, ignore_index=True).dropna()


def compute_dominant_asset_regret(
    policy_history: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: list[int] | None = None,
) -> dict:
    """Compare TD3 dominant choices against the best future asset."""
    returns_frame = _prepare_returns(returns)
    assets = _assets_from_policy(policy_history)
    missing_assets = [asset for asset in assets if asset not in returns_frame.columns]
    if missing_assets:
        raise ValueError(f"Missing return columns for assets: {missing_assets}")
    dominant = dominant_asset_from_policy(policy_history)
    selected_horizons = [1, 4, 12] if horizons is None else horizons
    observation_frames = []
    summary_rows = []

    for horizon in selected_horizons:
        _validate_horizon(horizon)
        forward = _forward_returns_for_horizon(returns_frame.loc[:, assets], horizon)
        forward = forward.dropna(how="any").copy()
        if forward.empty:
            continue
        observations = _merge_dominant_with_forward_returns(dominant, forward, horizon)
        if observations.empty:
            continue
        observation_frames.append(observations)
        summary_rows.append(_summarize_regret(observations, horizon))

    observations_frame = (
        pd.concat(observation_frames, ignore_index=True)
        if observation_frames
        else _empty_regret_observations()
    )
    summary_frame = (
        pd.DataFrame(summary_rows)
        if summary_rows
        else _empty_regret_summary()
    )
    return {"observations": observations_frame, "summary": summary_frame}


def compare_td3_to_rule_choices(
    policy_history: pd.DataFrame,
    returns: pd.DataFrame,
    rule_name: str,
    horizons: list[int] | None = None,
) -> dict:
    """Compare TD3 dominant choices with a simple dynamic rule."""
    returns_frame = _prepare_returns(returns)
    td3_dominant = dominant_asset_from_policy(policy_history)
    rule_weights = _build_rule_weights(returns_frame, rule_name)
    rule_policy = rule_weights.reset_index().rename(columns={"index": "date"})
    if "date" not in rule_policy.columns:
        rule_policy = rule_policy.rename(columns={rule_policy.columns[0]: "date"})
    rule_dominant = dominant_asset_from_policy(
        rule_policy.rename(columns={asset: f"weight_{asset}" for asset in returns_frame.columns}),
    )
    rule_dominant = rule_dominant.rename(
        columns={
            "dominant_asset": "rule_dominant_asset",
            "dominant_weight": "rule_dominant_weight",
        },
    )

    selected_horizons = [1, 4, 12] if horizons is None else horizons
    observation_frames = []
    summary_rows = []
    for horizon in selected_horizons:
        _validate_horizon(horizon)
        forward = _forward_returns_for_horizon(returns_frame, horizon).dropna(how="any")
        if forward.empty:
            continue
        observations = _merge_td3_rule_with_forward_returns(
            td3_dominant,
            rule_dominant,
            forward,
            horizon,
            rule_name,
        )
        if observations.empty:
            continue
        observation_frames.append(observations)
        summary_rows.append(_summarize_rule_comparison(observations, horizon, rule_name))

    observations_frame = (
        pd.concat(observation_frames, ignore_index=True)
        if observation_frames
        else _empty_rule_observations()
    )
    summary_frame = (
        pd.DataFrame(summary_rows)
        if summary_rows
        else _empty_rule_summary()
    )
    return {"observations": observations_frame, "summary": summary_frame}


def summarize_regime_error_attribution(
    regret_observations: pd.DataFrame,
    raw_regime_features: pd.DataFrame,
    regime_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Group dominant-asset regret by raw V5 regime columns when available."""
    if regret_observations.empty:
        return _empty_regime_summary()
    selected_columns = list(DEFAULT_REGIME_COLUMNS if regime_columns is None else regime_columns)
    if not isinstance(raw_regime_features.index, pd.DatetimeIndex):
        raise TypeError("raw_regime_features index must be a DatetimeIndex.")
    missing = [column for column in selected_columns if column not in raw_regime_features.columns]
    if missing:
        raise ValueError(f"Missing regime columns: {missing}")

    features = raw_regime_features.loc[:, selected_columns].copy()
    features = features.loc[~features.index.duplicated(keep="last")]
    features.index.name = None
    features["date"] = features.index
    observations = regret_observations.copy()
    observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
    merged = observations.merge(features, how="inner", on="date")
    if merged.empty:
        return _empty_regime_summary()

    rows = []
    group_keys = ["strategy", "horizon"] if "strategy" in merged.columns else ["horizon"]
    for regime_column in selected_columns:
        for keys, group in merged.groupby([*group_keys, regime_column], dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_values = dict(zip([*group_keys, regime_column], keys))
            row = {
                **{key: key_values[key] for key in group_keys},
                "regime_column": regime_column,
                "regime_value": key_values[regime_column],
                "n_observations": len(group),
                "mean_regret": group["regret"].mean(),
                "td3_best_asset_hit_rate": group["td3_is_best_asset"].mean(),
                "td3_beats_equal_weight_rate": group["td3_beats_equal_weight"].mean(),
                "mean_td3_excess_vs_equal_weight": group[
                    "td3_excess_vs_equal_weight"
                ].mean(),
            }
            rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [*group_keys, "regime_column", "regime_value"],
    ).reset_index(drop=True)


def build_decision_attribution_report(
    comparison_dir: str,
    returns_path: str = "data/processed/returns_weekly_latest.csv",
    strategies: list[str] | None = None,
    horizons: list[int] | None = None,
    rules: list[str] | None = None,
    output_dir: str | None = None,
) -> dict:
    """Build decision attribution reports for saved DRL policy histories."""
    comparison_path = Path(comparison_dir)
    if not comparison_path.exists():
        raise FileNotFoundError(f"Comparison directory not found: {comparison_dir}")
    returns = _load_returns_csv(returns_path)
    selected_strategies = (
        ["V2_reference", "V5_no_cash_penalty", "V5_dynamic_cash_025"]
        if strategies is None
        else strategies
    )
    selected_horizons = [1, 4, 12] if horizons is None else horizons
    selected_rules = list(DEFAULT_RULES if rules is None else rules)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else comparison_path / "decision_attribution"
    )
    destination.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    regret_summary_frames = []
    regret_observation_frames = []
    hit_rate_frames = []
    rule_summary_frames = []
    rule_observation_frames = []

    for strategy in selected_strategies:
        paths = sorted(comparison_path.glob(f"F*_{strategy}_seed_*/test_policy_history.csv"))
        if not paths:
            warnings.append(f"{strategy}: no test policy histories found.")
            continue
        for path in paths:
            fold, seed = _infer_fold_seed(path, strategy)
            policy_history = pd.read_csv(path)
            regret = compute_dominant_asset_regret(
                policy_history,
                returns,
                horizons=selected_horizons,
            )
            regret_observations = regret["observations"].copy()
            regret_observations.insert(0, "strategy", strategy)
            regret_observations.insert(1, "fold", fold)
            regret_observations.insert(2, "seed", seed)
            regret_observations.insert(3, "policy_history_path", str(path))
            regret_observation_frames.append(regret_observations)

            regret_summary = regret["summary"].copy()
            regret_summary.insert(0, "strategy", strategy)
            regret_summary.insert(1, "fold", fold)
            regret_summary.insert(2, "seed", seed)
            regret_summary.insert(3, "policy_history_path", str(path))
            regret_summary_frames.append(regret_summary)
            hit_rate_frames.append(
                regret_summary.loc[
                    :,
                    [
                        "strategy",
                        "fold",
                        "seed",
                        "horizon",
                        "td3_best_asset_hit_rate",
                        "td3_beats_equal_weight_rate",
                        "mean_td3_excess_vs_equal_weight",
                    ],
                ],
            )

            for rule_name in selected_rules:
                try:
                    comparison = compare_td3_to_rule_choices(
                        policy_history,
                        returns,
                        rule_name=rule_name,
                        horizons=selected_horizons,
                    )
                except ValueError as exc:
                    warnings.append(f"{strategy} {path.name} {rule_name}: {exc}")
                    continue
                rule_observations = comparison["observations"].copy()
                rule_observations.insert(0, "strategy", strategy)
                rule_observations.insert(1, "fold", fold)
                rule_observations.insert(2, "seed", seed)
                rule_observations.insert(3, "policy_history_path", str(path))
                rule_observation_frames.append(rule_observations)

                rule_summary = comparison["summary"].copy()
                rule_summary.insert(0, "strategy", strategy)
                rule_summary.insert(1, "fold", fold)
                rule_summary.insert(2, "seed", seed)
                rule_summary.insert(3, "policy_history_path", str(path))
                rule_summary_frames.append(rule_summary)

    regret_observations_frame = _concat_or_empty(
        regret_observation_frames,
        _empty_regret_observations(with_metadata=True),
    )
    regret_summary_raw = _concat_or_empty(regret_summary_frames, _empty_regret_summary())
    rule_summary_raw = _concat_or_empty(rule_summary_frames, _empty_rule_summary())
    rule_observations_frame = _concat_or_empty(
        rule_observation_frames,
        _empty_rule_observations(with_metadata=True),
    )
    hit_rate_raw = _concat_or_empty(hit_rate_frames, pd.DataFrame())

    regret_summary = _aggregate_regret_summary(regret_summary_raw)
    rule_summary = _aggregate_rule_summary(rule_summary_raw)
    hit_rate = _aggregate_hit_rate_summary(hit_rate_raw)

    regime_summary = _empty_regime_summary()
    try:
        raw_v5_features = build_raw_v5_features_for_returns(returns_path)
        regime_summary = summarize_regime_error_attribution(
            regret_observations_frame,
            raw_v5_features,
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        warnings.append(f"regime attribution skipped: {exc}")

    regret_summary.to_csv(destination / "dominant_asset_regret_summary.csv", index=False)
    rule_summary.to_csv(destination / "td3_vs_rule_choice_summary.csv", index=False)
    hit_rate.to_csv(destination / "dominant_asset_hit_rate_by_horizon.csv", index=False)
    if not regime_summary.empty:
        regime_summary.to_csv(destination / "regret_by_regime.csv", index=False)
    warnings_text = "\n".join(warnings) if warnings else "No decision attribution warnings."
    (destination / "decision_attribution_warnings.txt").write_text(warnings_text + "\n")

    return {
        "dominant_asset_regret_summary": regret_summary,
        "td3_vs_rule_choice_summary": rule_summary,
        "dominant_asset_hit_rate_by_horizon": hit_rate,
        "regret_by_regime": regime_summary,
        "warnings": warnings_text,
        "output_dir": str(destination),
        "dominant_asset_regret_summary_path": str(
            destination / "dominant_asset_regret_summary.csv",
        ),
        "td3_vs_rule_choice_summary_path": str(
            destination / "td3_vs_rule_choice_summary.csv",
        ),
        "dominant_asset_hit_rate_by_horizon_path": str(
            destination / "dominant_asset_hit_rate_by_horizon.csv",
        ),
        "regret_by_regime_path": str(destination / "regret_by_regime.csv")
        if not regime_summary.empty
        else None,
        "warnings_path": str(destination / "decision_attribution_warnings.txt"),
    }


def _infer_weight_columns(data: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in data.columns
        if str(column).startswith("weight_")
        and not str(column).endswith("_ok")
        and pd.api.types.is_numeric_dtype(data[column])
        and not pd.api.types.is_bool_dtype(data[column])
    ]
    if not columns:
        raise ValueError("No valid weight_ columns found.")
    return columns


def _assets_from_policy(policy_history: pd.DataFrame) -> list[str]:
    return [column.removeprefix("weight_") for column in _infer_weight_columns(policy_history)]


def _prepare_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    result = returns.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result = result.dropna(subset=["date"]).set_index("date")
    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex or date column.")
    result = result.sort_index()
    result.index.name = None
    result = result.apply(pd.to_numeric, errors="coerce")
    if result.isna().any().any():
        raise ValueError("returns must contain only numeric, non-missing values.")
    return result


def _load_returns_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Returns file not found: {path}")
    return _prepare_returns(pd.read_csv(file_path))


def _validate_horizon(horizon: int) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("horizon must be a non-bool integer.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")


def _forward_returns_for_horizon(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    compounded_through_current = (1.0 + returns).rolling(horizon).apply(
        np.prod,
        raw=True,
    ) - 1.0
    return compounded_through_current.shift(-horizon)


def _merge_dominant_with_forward_returns(
    dominant: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    forward = forward_returns.copy()
    forward = forward.loc[~forward.index.duplicated(keep="last")]
    forward.index.name = None
    forward["date"] = forward.index
    merged = dominant.merge(forward, how="inner", on="date")
    if merged.empty:
        return _empty_regret_observations()

    rows = []
    asset_columns = [column for column in forward_returns.columns]
    for _, row in merged.iterrows():
        td3_asset = row["dominant_asset"]
        if td3_asset not in asset_columns:
            continue
        asset_returns = row.loc[asset_columns].astype(float)
        td3_return = float(asset_returns[td3_asset])
        best_asset = str(asset_returns.idxmax())
        best_return = float(asset_returns.max())
        equal_weight_return = float(asset_returns.mean())
        rows.append(
            {
                "date": row["date"],
                "horizon": horizon,
                "td3_dominant_asset": td3_asset,
                "td3_dominant_weight": row["dominant_weight"],
                "td3_forward_return": td3_return,
                "best_forward_asset": best_asset,
                "best_forward_return": best_return,
                "equal_weight_forward_return": equal_weight_return,
                "regret": best_return - td3_return,
                "td3_is_best_asset": td3_asset == best_asset,
                "td3_beats_equal_weight": td3_return > equal_weight_return,
                "td3_excess_vs_equal_weight": td3_return - equal_weight_return,
            },
        )
    return pd.DataFrame(rows)


def _summarize_regret(observations: pd.DataFrame, horizon: int) -> dict:
    return {
        "horizon": horizon,
        "n_observations": len(observations),
        "mean_regret": observations["regret"].mean(),
        "median_regret": observations["regret"].median(),
        "regret_positive_rate": (observations["regret"] > 0.0).mean(),
        "td3_best_asset_hit_rate": observations["td3_is_best_asset"].mean(),
        "td3_beats_equal_weight_rate": observations["td3_beats_equal_weight"].mean(),
        "mean_td3_forward_return": observations["td3_forward_return"].mean(),
        "mean_best_forward_return": observations["best_forward_return"].mean(),
        "mean_equal_weight_forward_return": observations[
            "equal_weight_forward_return"
        ].mean(),
        "mean_td3_excess_vs_equal_weight": observations[
            "td3_excess_vs_equal_weight"
        ].mean(),
    }


def _build_rule_weights(returns: pd.DataFrame, rule_name: str) -> pd.DataFrame:
    if rule_name == "momentum_winner_12p":
        return build_momentum_winner_weights(returns, window=12)
    if rule_name == "risk_adjusted_momentum_winner_12p_12p":
        return build_risk_adjusted_momentum_winner_weights(
            returns,
            momentum_window=12,
            volatility_window=12,
        )
    if rule_name == "trend_spy_cash_12p":
        return build_trend_following_spy_cash_weights(returns, window=12)
    if rule_name == "defensive_risk_off_12p":
        return build_defensive_risk_off_weights(returns, window=12)
    raise ValueError(f"Unsupported rule_name: {rule_name}")


def _merge_td3_rule_with_forward_returns(
    td3_dominant: pd.DataFrame,
    rule_dominant: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizon: int,
    rule_name: str,
) -> pd.DataFrame:
    forward = forward_returns.copy()
    forward = forward.loc[~forward.index.duplicated(keep="last")]
    forward.index.name = None
    forward["date"] = forward.index
    merged = td3_dominant.merge(rule_dominant, how="inner", on="date")
    merged = merged.merge(forward, how="inner", on="date")
    if merged.empty:
        return _empty_rule_observations()

    rows = []
    asset_columns = [column for column in forward_returns.columns]
    for _, row in merged.iterrows():
        td3_asset = row["dominant_asset"]
        rule_asset = row["rule_dominant_asset"]
        if td3_asset not in asset_columns or rule_asset not in asset_columns:
            continue
        td3_return = float(row[td3_asset])
        rule_return = float(row[rule_asset])
        rows.append(
            {
                "date": row["date"],
                "horizon": horizon,
                "rule_name": rule_name,
                "td3_dominant_asset": td3_asset,
                "rule_dominant_asset": rule_asset,
                "td3_forward_return": td3_return,
                "rule_forward_return": rule_return,
                "td3_minus_rule_forward_return": td3_return - rule_return,
                "td3_rule_overlap": td3_asset == rule_asset,
                "td3_beats_rule": td3_return > rule_return,
            },
        )
    return pd.DataFrame(rows)


def _summarize_rule_comparison(
    observations: pd.DataFrame,
    horizon: int,
    rule_name: str,
) -> dict:
    return {
        "horizon": horizon,
        "rule_name": rule_name,
        "n_observations": len(observations),
        "overlap_rate": observations["td3_rule_overlap"].mean(),
        "mean_td3_forward_return": observations["td3_forward_return"].mean(),
        "mean_rule_forward_return": observations["rule_forward_return"].mean(),
        "mean_td3_minus_rule": observations["td3_minus_rule_forward_return"].mean(),
        "td3_win_rate_vs_rule": observations["td3_beats_rule"].mean(),
    }


def _infer_fold_seed(path: Path, strategy: str) -> tuple[str, int | None]:
    name = path.parent.name
    prefix, _, seed_text = name.partition(f"_{strategy}_seed_")
    fold = prefix if prefix else "unknown"
    try:
        seed = int(seed_text)
    except ValueError:
        seed = None
    return fold, seed


def _aggregate_regret_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    return (
        summary.groupby(["strategy", "horizon"], as_index=False)
        .agg(
            n_runs=("policy_history_path", "nunique"),
            total_observations=("n_observations", "sum"),
            mean_regret=("mean_regret", "mean"),
            median_regret=("median_regret", "mean"),
            regret_positive_rate=("regret_positive_rate", "mean"),
            td3_best_asset_hit_rate=("td3_best_asset_hit_rate", "mean"),
            td3_beats_equal_weight_rate=("td3_beats_equal_weight_rate", "mean"),
            mean_td3_forward_return=("mean_td3_forward_return", "mean"),
            mean_best_forward_return=("mean_best_forward_return", "mean"),
            mean_equal_weight_forward_return=("mean_equal_weight_forward_return", "mean"),
            mean_td3_excess_vs_equal_weight=("mean_td3_excess_vs_equal_weight", "mean"),
        )
        .sort_values(["strategy", "horizon"])
        .reset_index(drop=True)
    )


def _aggregate_rule_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    return (
        summary.groupby(["strategy", "rule_name", "horizon"], as_index=False)
        .agg(
            n_runs=("policy_history_path", "nunique"),
            total_observations=("n_observations", "sum"),
            overlap_rate=("overlap_rate", "mean"),
            mean_td3_forward_return=("mean_td3_forward_return", "mean"),
            mean_rule_forward_return=("mean_rule_forward_return", "mean"),
            mean_td3_minus_rule=("mean_td3_minus_rule", "mean"),
            td3_win_rate_vs_rule=("td3_win_rate_vs_rule", "mean"),
        )
        .sort_values(["strategy", "rule_name", "horizon"])
        .reset_index(drop=True)
    )


def _aggregate_hit_rate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    return (
        summary.groupby(["strategy", "horizon"], as_index=False)
        .agg(
            td3_best_asset_hit_rate=("td3_best_asset_hit_rate", "mean"),
            td3_beats_equal_weight_rate=("td3_beats_equal_weight_rate", "mean"),
            mean_td3_excess_vs_equal_weight=("mean_td3_excess_vs_equal_weight", "mean"),
        )
        .sort_values(["strategy", "horizon"])
        .reset_index(drop=True)
    )


def _concat_or_empty(frames: list[pd.DataFrame], empty: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else empty


def _empty_regret_observations(with_metadata: bool = False) -> pd.DataFrame:
    columns = [
        "date",
        "horizon",
        "td3_dominant_asset",
        "td3_dominant_weight",
        "td3_forward_return",
        "best_forward_asset",
        "best_forward_return",
        "equal_weight_forward_return",
        "regret",
        "td3_is_best_asset",
        "td3_beats_equal_weight",
        "td3_excess_vs_equal_weight",
    ]
    if with_metadata:
        columns = ["strategy", "fold", "seed", "policy_history_path", *columns]
    return pd.DataFrame(columns=columns)


def _empty_regret_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "horizon",
            "n_observations",
            "mean_regret",
            "median_regret",
            "regret_positive_rate",
            "td3_best_asset_hit_rate",
            "td3_beats_equal_weight_rate",
            "mean_td3_forward_return",
            "mean_best_forward_return",
            "mean_equal_weight_forward_return",
            "mean_td3_excess_vs_equal_weight",
        ],
    )


def _empty_rule_observations(with_metadata: bool = False) -> pd.DataFrame:
    columns = [
        "date",
        "horizon",
        "rule_name",
        "td3_dominant_asset",
        "rule_dominant_asset",
        "td3_forward_return",
        "rule_forward_return",
        "td3_minus_rule_forward_return",
        "td3_rule_overlap",
        "td3_beats_rule",
    ]
    if with_metadata:
        columns = ["strategy", "fold", "seed", "policy_history_path", *columns]
    return pd.DataFrame(columns=columns)


def _empty_rule_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "horizon",
            "rule_name",
            "n_observations",
            "overlap_rate",
            "mean_td3_forward_return",
            "mean_rule_forward_return",
            "mean_td3_minus_rule",
            "td3_win_rate_vs_rule",
        ],
    )


def _empty_regime_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "strategy",
            "horizon",
            "regime_column",
            "regime_value",
            "n_observations",
            "mean_regret",
            "td3_best_asset_hit_rate",
            "td3_beats_equal_weight_rate",
            "mean_td3_excess_vs_equal_weight",
        ],
    )
