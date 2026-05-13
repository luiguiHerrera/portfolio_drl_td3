"""Mandate risk diagnostics for saved portfolio experiment outputs.

Concentration is treated as a mandate exposure, not as an automatic error.
The limits used here are explicit function arguments so future suitability or
appropriateness workflows can map client profiles into these same controls.
"""

from pathlib import Path

import pandas as pd


REQUIRED_METRIC_COLUMNS = [
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "sortino_ratio",
    "calmar_ratio",
    "information_ratio_vs_equal_weight_rebalanced_net",
    "capm_alpha_vs_SPY",
]

REQUIRED_DIAGNOSTIC_COLUMNS = [
    "average_max_weight",
    "final_max_weight",
    "average_effective_number_of_assets",
    "final_effective_number_of_assets",
    "average_herfindahl_index",
    "final_herfindahl_index",
    "average_turnover",
    "final_turnover",
    "average_transaction_cost",
    "final_transaction_cost",
]


def load_metrics_and_diagnostics(
    metrics_paths: list[str],
    diagnostics_paths: list[str],
) -> pd.DataFrame:
    """Load agent metrics and diagnostics CSVs into one observation table."""
    if not metrics_paths or not diagnostics_paths:
        raise ValueError("metrics_paths and diagnostics_paths must be non-empty.")
    if len(metrics_paths) != len(diagnostics_paths):
        raise ValueError("metrics_paths and diagnostics_paths must have the same length.")

    rows = []
    for metrics_path, diagnostics_path in zip(metrics_paths, diagnostics_paths):
        metrics_file = Path(metrics_path)
        diagnostics_file = Path(diagnostics_path)
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        if not diagnostics_file.exists():
            raise FileNotFoundError(f"Diagnostics file not found: {diagnostics_path}")

        metrics = pd.read_csv(metrics_file)
        diagnostics = pd.read_csv(diagnostics_file)
        if diagnostics.empty:
            raise ValueError(f"Diagnostics CSV is empty: {diagnostics_path}")

        agent_row = _extract_agent_row(metrics, metrics_path)
        diagnostics_row = diagnostics.iloc[0]
        combined = pd.concat([agent_row, diagnostics_row])
        combined["metrics_path"] = str(metrics_file)
        combined["diagnostics_path"] = str(diagnostics_file)
        rows.append(combined)

    return pd.DataFrame(rows).reset_index(drop=True)


def add_mandate_flags(
    data: pd.DataFrame,
    max_drawdown_limit: float = -0.20,
    max_volatility_limit: float = 0.25,
    max_weight_limit: float = 0.80,
    min_effective_assets: float = 1.25,
    max_turnover_limit: float = 0.75,
) -> pd.DataFrame:
    """Return a copy of data with per-observation mandate compliance flags."""
    _validate_mandate_limits(
        max_drawdown_limit=max_drawdown_limit,
        max_volatility_limit=max_volatility_limit,
        max_weight_limit=max_weight_limit,
        min_effective_assets=min_effective_assets,
        max_turnover_limit=max_turnover_limit,
    )
    _require_columns(data, REQUIRED_METRIC_COLUMNS + REQUIRED_DIAGNOSTIC_COLUMNS)

    flagged = data.copy()
    flagged["drawdown_ok"] = flagged["max_drawdown"] >= max_drawdown_limit
    flagged["volatility_ok"] = flagged["annualized_volatility"] <= max_volatility_limit
    flagged["final_weight_ok"] = flagged["final_max_weight"] <= max_weight_limit
    flagged["average_weight_ok"] = flagged["average_max_weight"] <= max_weight_limit
    flagged["effective_assets_ok"] = (
        flagged["average_effective_number_of_assets"] >= min_effective_assets
    )
    flagged["turnover_ok"] = flagged["average_turnover"] <= max_turnover_limit
    flag_columns = [
        "drawdown_ok",
        "volatility_ok",
        "final_weight_ok",
        "average_weight_ok",
        "effective_assets_ok",
        "turnover_ok",
    ]
    flagged["mandate_pass"] = flagged[flag_columns].all(axis=1)

    return flagged


def summarize_mandate_risk(
    data: pd.DataFrame,
    max_drawdown_limit: float = -0.20,
    max_volatility_limit: float = 0.25,
    max_weight_limit: float = 0.80,
    min_effective_assets: float = 1.25,
    max_turnover_limit: float = 0.75,
) -> pd.DataFrame:
    """Summarize performance, risk, concentration, and mandate compliance."""
    flagged = add_mandate_flags(
        data,
        max_drawdown_limit=max_drawdown_limit,
        max_volatility_limit=max_volatility_limit,
        max_weight_limit=max_weight_limit,
        min_effective_assets=min_effective_assets,
        max_turnover_limit=max_turnover_limit,
    )

    summary = {
        "n_observations": len(flagged),
        "max_drawdown_limit": max_drawdown_limit,
        "max_volatility_limit": max_volatility_limit,
        "max_weight_limit": max_weight_limit,
        "min_effective_assets": min_effective_assets,
        "max_turnover_limit": max_turnover_limit,
        "mean_cumulative_return": flagged["cumulative_return"].mean(),
        "mean_annualized_return": flagged["annualized_return"].mean(),
        "mean_annualized_volatility": flagged["annualized_volatility"].mean(),
        "mean_sharpe_ratio": flagged["sharpe_ratio"].mean(),
        "mean_sortino_ratio": flagged["sortino_ratio"].mean(),
        "mean_calmar_ratio": flagged["calmar_ratio"].mean(),
        "mean_max_drawdown": flagged["max_drawdown"].mean(),
        "worst_max_drawdown": flagged["max_drawdown"].min(),
        "mean_information_ratio_vs_equal_weight_rebalanced_net": flagged[
            "information_ratio_vs_equal_weight_rebalanced_net"
        ].mean(),
        "mean_capm_alpha_vs_SPY": flagged["capm_alpha_vs_SPY"].mean(),
        "mean_average_max_weight": flagged["average_max_weight"].mean(),
        "mean_final_max_weight": flagged["final_max_weight"].mean(),
        "max_final_max_weight": flagged["final_max_weight"].max(),
        "mean_average_effective_number_of_assets": flagged[
            "average_effective_number_of_assets"
        ].mean(),
        "min_average_effective_number_of_assets": flagged[
            "average_effective_number_of_assets"
        ].min(),
        "mean_final_effective_number_of_assets": flagged[
            "final_effective_number_of_assets"
        ].mean(),
        "min_final_effective_number_of_assets": flagged[
            "final_effective_number_of_assets"
        ].min(),
        "mean_average_herfindahl_index": flagged["average_herfindahl_index"].mean(),
        "mean_final_herfindahl_index": flagged["final_herfindahl_index"].mean(),
        "mean_average_turnover": flagged["average_turnover"].mean(),
        "mean_final_turnover": flagged["final_turnover"].mean(),
        "mean_average_transaction_cost": flagged["average_transaction_cost"].mean(),
        "mean_final_transaction_cost": flagged["final_transaction_cost"].mean(),
        "drawdown_pass_rate": flagged["drawdown_ok"].mean(),
        "volatility_pass_rate": flagged["volatility_ok"].mean(),
        "final_weight_pass_rate": flagged["final_weight_ok"].mean(),
        "average_weight_pass_rate": flagged["average_weight_ok"].mean(),
        "effective_assets_pass_rate": flagged["effective_assets_ok"].mean(),
        "turnover_pass_rate": flagged["turnover_ok"].mean(),
        "mandate_pass_rate": flagged["mandate_pass"].mean(),
    }

    return pd.DataFrame([summary])


def summarize_dominant_assets(
    diagnostics: pd.DataFrame,
    asset_columns_prefix: str = "final_weight_",
) -> pd.DataFrame:
    """Summarize which asset has the highest final portfolio weight."""
    asset_columns = [
        column
        for column in diagnostics.columns
        if _is_asset_weight_column(diagnostics, column, asset_columns_prefix)
    ]
    if not asset_columns:
        raise ValueError(f"No {asset_columns_prefix} columns found.")

    final_weights = diagnostics.loc[:, asset_columns].copy()
    dominant_columns = final_weights.idxmax(axis=1)
    dominant_assets = dominant_columns.str.removeprefix(asset_columns_prefix)
    dominant_weights = final_weights.max(axis=1)

    dominant = pd.DataFrame(
        {
            "dominant_asset": dominant_assets,
            "dominant_weight": dominant_weights,
        }
    )
    summary = (
        dominant.groupby("dominant_asset", as_index=False)
        .agg(
            count=("dominant_asset", "size"),
            mean_final_weight=("dominant_weight", "mean"),
        )
        .sort_values(["count", "dominant_asset"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["rate"] = summary["count"] / len(diagnostics)

    return summary.loc[:, ["dominant_asset", "count", "rate", "mean_final_weight"]]


def build_mandate_risk_report(
    metrics_paths: list[str],
    diagnostics_paths: list[str],
    output_dir: str | None = None,
    report_name: str = "mandate_risk_diagnostics",
    max_drawdown_limit: float = -0.20,
    max_volatility_limit: float = 0.25,
    max_weight_limit: float = 0.80,
    min_effective_assets: float = 1.25,
    max_turnover_limit: float = 0.75,
) -> dict:
    """Build a mandate risk report from saved metrics and diagnostics CSVs."""
    combined = load_metrics_and_diagnostics(metrics_paths, diagnostics_paths)
    observations = add_mandate_flags(
        combined,
        max_drawdown_limit=max_drawdown_limit,
        max_volatility_limit=max_volatility_limit,
        max_weight_limit=max_weight_limit,
        min_effective_assets=min_effective_assets,
        max_turnover_limit=max_turnover_limit,
    )
    summary = summarize_mandate_risk(
        combined,
        max_drawdown_limit=max_drawdown_limit,
        max_volatility_limit=max_volatility_limit,
        max_weight_limit=max_weight_limit,
        min_effective_assets=min_effective_assets,
        max_turnover_limit=max_turnover_limit,
    )
    dominant_assets = summarize_dominant_assets(observations)

    observations_path = None
    summary_path = None
    dominant_assets_path = None
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        observations_path = str(destination / f"{report_name}_observations.csv")
        summary_path = str(destination / f"{report_name}_summary.csv")
        dominant_assets_path = str(destination / f"{report_name}_dominant_assets.csv")

        observations.to_csv(observations_path, index=False)
        summary.to_csv(summary_path, index=False)
        dominant_assets.to_csv(dominant_assets_path, index=False)

    return {
        "observations": observations,
        "summary": summary,
        "dominant_assets": dominant_assets,
        "observations_path": observations_path,
        "summary_path": summary_path,
        "dominant_assets_path": dominant_assets_path,
    }


def _extract_agent_row(metrics: pd.DataFrame, metrics_path: str) -> pd.Series:
    for column in metrics.columns:
        if metrics[column].astype(str).eq("agent").any():
            row = metrics.loc[metrics[column].astype(str).eq("agent")].iloc[0]
            return row.drop(labels=[column])

    if metrics.index.astype(str).str.contains("agent").any():
        return metrics.loc[metrics.index.astype(str) == "agent"].iloc[0]

    raise ValueError(f"Agent row not found in metrics CSV: {metrics_path}")


def _validate_mandate_limits(
    max_drawdown_limit: float,
    max_volatility_limit: float,
    max_weight_limit: float,
    min_effective_assets: float,
    max_turnover_limit: float,
) -> None:
    if max_drawdown_limit > 0:
        raise ValueError("max_drawdown_limit must be <= 0.")
    if max_volatility_limit < 0:
        raise ValueError("max_volatility_limit must be >= 0.")
    if max_weight_limit < 0 or max_weight_limit > 1:
        raise ValueError("max_weight_limit must be between 0 and 1 inclusive.")
    if min_effective_assets < 1:
        raise ValueError("min_effective_assets must be >= 1.")
    if max_turnover_limit < 0:
        raise ValueError("max_turnover_limit must be >= 0.")


def _require_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")


def _is_asset_weight_column(
    diagnostics: pd.DataFrame,
    column: str,
    asset_columns_prefix: str,
) -> bool:
    column_name = str(column)
    if not column_name.startswith(asset_columns_prefix):
        return False
    if column_name == "final_weight_ok" or column_name.endswith("_ok"):
        return False
    if pd.api.types.is_bool_dtype(diagnostics[column]):
        return False

    return pd.api.types.is_numeric_dtype(diagnostics[column])
