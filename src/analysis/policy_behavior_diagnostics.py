"""Policy behavior diagnostics for saved or in-memory portfolio weights.

These utilities describe what a learned policy is doing before changing reward
terms, max-weight constraints, or portfolio restrictions. Concentration is
measured as behavior to understand, not assumed to be wrong.
"""

from pathlib import Path

import pandas as pd


def infer_weight_columns(
    data: pd.DataFrame,
    prefix: str = "weight_",
) -> list[str]:
    """Return valid numeric asset-weight columns."""
    weight_columns = [
        column
        for column in data.columns
        if _is_valid_weight_column(data, column, prefix)
    ]
    if not weight_columns:
        raise ValueError(f"No valid {prefix} columns found.")

    return weight_columns


def add_policy_state_columns(
    data: pd.DataFrame,
    weight_prefix: str = "weight_",
) -> pd.DataFrame:
    """Add concentration and dominant-asset state columns to policy data."""
    weight_columns = infer_weight_columns(data, weight_prefix)
    weights = data.loc[:, weight_columns]
    weight_abs_sum = weights.abs().sum(axis=1)
    if (weight_abs_sum == 0).any():
        raise ValueError("Weight rows must not have total absolute weight equal to 0.")

    result = data.copy()
    result["weight_sum"] = weights.sum(axis=1)

    dominant_columns = weights.idxmax(axis=1)
    result["dominant_asset"] = dominant_columns.str.removeprefix(weight_prefix)
    result["dominant_weight"] = weights.max(axis=1)
    result["herfindahl_index"] = (weights**2).sum(axis=1)
    result["effective_number_of_assets"] = 1.0 / result["herfindahl_index"]
    result["is_highly_concentrated_80"] = result["dominant_weight"] >= 0.80
    result["is_highly_concentrated_90"] = result["dominant_weight"] >= 0.90

    return result


def summarize_policy_concentration(
    data: pd.DataFrame,
    weight_prefix: str = "weight_",
) -> pd.DataFrame:
    """Summarize concentration intensity across observations."""
    observations = add_policy_state_columns(data, weight_prefix)
    summary = {
        "n_observations": len(observations),
        "mean_dominant_weight": observations["dominant_weight"].mean(),
        "median_dominant_weight": observations["dominant_weight"].median(),
        "max_dominant_weight": observations["dominant_weight"].max(),
        "min_dominant_weight": observations["dominant_weight"].min(),
        "mean_herfindahl_index": observations["herfindahl_index"].mean(),
        "mean_effective_number_of_assets": observations[
            "effective_number_of_assets"
        ].mean(),
        "min_effective_number_of_assets": observations[
            "effective_number_of_assets"
        ].min(),
        "high_concentration_80_rate": observations[
            "is_highly_concentrated_80"
        ].mean(),
        "high_concentration_90_rate": observations[
            "is_highly_concentrated_90"
        ].mean(),
        "mean_weight_sum": observations["weight_sum"].mean(),
        "min_weight_sum": observations["weight_sum"].min(),
        "max_weight_sum": observations["weight_sum"].max(),
    }

    return pd.DataFrame([summary])


def summarize_dominant_asset_distribution(
    data: pd.DataFrame,
    weight_prefix: str = "weight_",
) -> pd.DataFrame:
    """Summarize how often each asset is dominant."""
    observations = add_policy_state_columns(data, weight_prefix)
    summary = (
        observations.groupby("dominant_asset", as_index=False)
        .agg(
            count=("dominant_asset", "size"),
            mean_dominant_weight=("dominant_weight", "mean"),
            mean_effective_number_of_assets=("effective_number_of_assets", "mean"),
        )
        .sort_values(["count", "dominant_asset"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["rate"] = summary["count"] / len(observations)

    return summary.loc[
        :,
        [
            "dominant_asset",
            "count",
            "rate",
            "mean_dominant_weight",
            "mean_effective_number_of_assets",
        ],
    ]


def summarize_dominant_asset_transitions(
    data: pd.DataFrame,
    weight_prefix: str = "weight_",
) -> pd.DataFrame:
    """Count switches from one dominant asset to another."""
    observations = add_policy_state_columns(data, weight_prefix)
    previous = observations["dominant_asset"].shift(1)
    current = observations["dominant_asset"]
    switches = pd.DataFrame(
        {
            "from_asset": previous,
            "to_asset": current,
        }
    ).iloc[1:]
    switches = switches.loc[switches["from_asset"] != switches["to_asset"]]
    columns = ["from_asset", "to_asset", "count", "rate_of_all_switches"]
    if switches.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        switches.groupby(["from_asset", "to_asset"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "from_asset", "to_asset"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    summary["rate_of_all_switches"] = summary["count"] / summary["count"].sum()

    return summary.loc[:, columns]


def summarize_holding_periods(
    data: pd.DataFrame,
    weight_prefix: str = "weight_",
) -> pd.DataFrame:
    """Return consecutive dominant-asset holding periods."""
    observations = add_policy_state_columns(data, weight_prefix)
    group_id = observations["dominant_asset"].ne(
        observations["dominant_asset"].shift(1)
    ).cumsum()
    index_values = observations.index.to_series()
    holding_periods = (
        observations.assign(_group_id=group_id, _index_value=index_values)
        .groupby("_group_id", as_index=False)
        .agg(
            dominant_asset=("dominant_asset", "first"),
            start_index=("_index_value", "first"),
            end_index=("_index_value", "last"),
            holding_period_length=("dominant_asset", "size"),
            mean_dominant_weight=("dominant_weight", "mean"),
            mean_effective_number_of_assets=("effective_number_of_assets", "mean"),
        )
    )

    return holding_periods.loc[
        :,
        [
            "dominant_asset",
            "start_index",
            "end_index",
            "holding_period_length",
            "mean_dominant_weight",
            "mean_effective_number_of_assets",
        ],
    ]


def summarize_asset_conditional_performance(
    data: pd.DataFrame,
    return_column: str,
    weight_prefix: str = "weight_",
) -> pd.DataFrame:
    """Summarize realized returns conditional on dominant asset."""
    if return_column not in data.columns:
        raise KeyError(f"Missing return column: {return_column}")

    observations = add_policy_state_columns(data, weight_prefix)
    summary = (
        observations.groupby("dominant_asset", as_index=False)
        .agg(
            n_observations=(return_column, "size"),
            mean_return=(return_column, "mean"),
            volatility=(return_column, "std"),
            cumulative_return=(return_column, _compound_return),
            hit_rate=(return_column, lambda returns: (returns > 0).mean()),
            mean_dominant_weight=("dominant_weight", "mean"),
            mean_effective_number_of_assets=("effective_number_of_assets", "mean"),
        )
        .sort_values(["n_observations", "dominant_asset"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return summary


def summarize_regime_attribution(
    data: pd.DataFrame,
    regime_columns: list[str],
    weight_prefix: str = "weight_",
) -> pd.DataFrame:
    """Summarize regime feature averages by dominant asset."""
    missing_columns = [column for column in regime_columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Missing regime columns: {missing}")

    non_numeric_columns = [
        column
        for column in regime_columns
        if not pd.api.types.is_numeric_dtype(data[column])
        or pd.api.types.is_bool_dtype(data[column])
    ]
    if non_numeric_columns:
        invalid = ", ".join(non_numeric_columns)
        raise ValueError(f"Regime columns must be numeric: {invalid}")

    observations = add_policy_state_columns(data, weight_prefix)
    aggregations = {
        column: (column, "mean")
        for column in regime_columns
    }
    summary = (
        observations.groupby("dominant_asset", as_index=False)
        .agg(
            n_observations=("dominant_asset", "size"),
            **{
                f"mean_{column}": aggregation
                for column, aggregation in aggregations.items()
            },
        )
        .sort_values(["n_observations", "dominant_asset"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return summary


def build_policy_behavior_report(
    data: pd.DataFrame,
    return_column: str | None = None,
    regime_columns: list[str] | None = None,
    output_dir: str | None = None,
    report_name: str = "policy_behavior_diagnostics",
    weight_prefix: str = "weight_",
) -> dict:
    """Build a complete policy behavior diagnostics report."""
    observations = add_policy_state_columns(data, weight_prefix)
    concentration_summary = summarize_policy_concentration(data, weight_prefix)
    dominant_asset_distribution = summarize_dominant_asset_distribution(
        data,
        weight_prefix,
    )
    dominant_asset_transitions = summarize_dominant_asset_transitions(
        data,
        weight_prefix,
    )
    holding_periods = summarize_holding_periods(data, weight_prefix)
    conditional_performance = None
    if return_column is not None:
        conditional_performance = summarize_asset_conditional_performance(
            data,
            return_column=return_column,
            weight_prefix=weight_prefix,
        )
    regime_attribution = None
    if regime_columns is not None:
        regime_attribution = summarize_regime_attribution(
            data,
            regime_columns=regime_columns,
            weight_prefix=weight_prefix,
        )

    paths = _save_policy_behavior_outputs(
        output_dir=output_dir,
        report_name=report_name,
        observations=observations,
        concentration_summary=concentration_summary,
        dominant_asset_distribution=dominant_asset_distribution,
        dominant_asset_transitions=dominant_asset_transitions,
        holding_periods=holding_periods,
        conditional_performance=conditional_performance,
        regime_attribution=regime_attribution,
    )

    return {
        "observations": observations,
        "concentration_summary": concentration_summary,
        "dominant_asset_distribution": dominant_asset_distribution,
        "dominant_asset_transitions": dominant_asset_transitions,
        "holding_periods": holding_periods,
        "conditional_performance": conditional_performance,
        "regime_attribution": regime_attribution,
        **paths,
    }


def _is_valid_weight_column(
    data: pd.DataFrame,
    column: str,
    prefix: str,
) -> bool:
    column_name = str(column)
    if not column_name.startswith(prefix):
        return False
    if column_name.endswith("_ok"):
        return False
    if pd.api.types.is_bool_dtype(data[column]):
        return False

    return pd.api.types.is_numeric_dtype(data[column])


def _compound_return(returns: pd.Series) -> float:
    return (1.0 + returns).prod() - 1.0


def _save_policy_behavior_outputs(
    output_dir: str | None,
    report_name: str,
    observations: pd.DataFrame,
    concentration_summary: pd.DataFrame,
    dominant_asset_distribution: pd.DataFrame,
    dominant_asset_transitions: pd.DataFrame,
    holding_periods: pd.DataFrame,
    conditional_performance: pd.DataFrame | None,
    regime_attribution: pd.DataFrame | None,
) -> dict:
    path_keys = {
        "observations_path": None,
        "concentration_summary_path": None,
        "dominant_asset_distribution_path": None,
        "dominant_asset_transitions_path": None,
        "holding_periods_path": None,
        "conditional_performance_path": None,
        "regime_attribution_path": None,
    }
    if output_dir is None:
        return path_keys

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    outputs = {
        "observations_path": observations,
        "concentration_summary_path": concentration_summary,
        "dominant_asset_distribution_path": dominant_asset_distribution,
        "dominant_asset_transitions_path": dominant_asset_transitions,
        "holding_periods_path": holding_periods,
    }
    if conditional_performance is not None:
        outputs["conditional_performance_path"] = conditional_performance
    if regime_attribution is not None:
        outputs["regime_attribution_path"] = regime_attribution

    for key, frame in outputs.items():
        file_name = key.removesuffix("_path")
        path = destination / f"{report_name}_{file_name}.csv"
        frame.to_csv(path, index=False)
        path_keys[key] = str(path)

    return path_keys
