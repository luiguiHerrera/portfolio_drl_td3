"""Robust strategy scoring utilities for ex-post evaluation.

This module is intentionally reporting-only. It does not participate in
training, reward calculation, or environment dynamics.
"""

from __future__ import annotations

from math import e, erf, isfinite, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

DEFAULT_COMPOSITE_WEIGHTS = {
    "dsr_score": 0.30,
    "sortino_score": 0.20,
    "calmar_score": 0.20,
    "drawdown_score": 0.15,
    "stability_score": 0.10,
    "discipline_score": 0.05,
}

EULER_MASCHERONI = 0.5772156649
POOLED_DSR_WARNING = (
    "Pooled DSR can overstate evidence when folds/seeds contain overlapping dates. "
    "Composite robust_score uses median run-level DSR when available."
)


def compute_annualized_sharpe(returns, periods_per_year: int = 52) -> float:
    """Compute annualized Sharpe from periodic returns, with safe edge handling."""
    returns_series = _clean_returns(returns)
    _validate_periods_per_year(periods_per_year)
    if len(returns_series) < 2:
        return 0.0
    volatility = returns_series.std(ddof=1)
    if volatility <= 1e-12 or not np.isfinite(volatility):
        return 0.0
    return float(returns_series.mean() / volatility * sqrt(periods_per_year))


def compute_probabilistic_sharpe_ratio(
    returns,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """Compute a Probabilistic Sharpe Ratio in [0, 1].

    Public-facing Sharpe inputs are annualized. Internally the annualized
    observed and benchmark Sharpe ratios are converted to matching periodic
    units before applying the Bailey/Lopez de Prado PSR statistic.
    """
    returns_series = _clean_returns(returns)
    if len(returns_series) < 3:
        return 0.0
    _validate_periods_per_year(periods_per_year)
    _validate_number(benchmark_sharpe, "benchmark_sharpe")

    volatility = returns_series.std(ddof=1)
    if volatility <= 1e-12 or not np.isfinite(volatility):
        return 0.0

    observed_annualized_sharpe = compute_annualized_sharpe(
        returns_series,
        periods_per_year=periods_per_year,
    )
    observed_periodic_sharpe = observed_annualized_sharpe / sqrt(periods_per_year)
    benchmark_periodic_sharpe = float(benchmark_sharpe / sqrt(periods_per_year))
    skewness = float(returns_series.skew())
    kurtosis = float(returns_series.kurt()) + 3.0
    if not all(np.isfinite([observed_periodic_sharpe, skewness, kurtosis])):
        return 0.0

    denominator = sqrt(
        max(
            1.0
            - skewness * observed_periodic_sharpe
            + ((kurtosis - 1.0) / 4.0) * observed_periodic_sharpe**2,
            1e-12,
        ),
    )
    statistic = (
        (observed_periodic_sharpe - benchmark_periodic_sharpe)
        * sqrt(len(returns_series) - 1.0)
        / denominator
    )
    return _clip_probability(_standard_normal_cdf(statistic))


def estimate_expected_max_sharpe(
    sharpe_std: float,
    n_trials: int = 1,
    sharpe_mean: float = 0.0,
) -> float:
    """Estimate expected maximum Sharpe across trials.

    Uses the Bailey/Lopez de Prado approximation:
    E[max SR_N] ~= SR_mean + SR_std * [
        (1 - gamma) * Phi^-1(1 - 1/N)
        + gamma * Phi^-1(1 - 1/(N * e))
    ].
    """
    _validate_number(sharpe_std, "sharpe_std")
    _validate_number(sharpe_mean, "sharpe_mean")
    if sharpe_std < 0:
        raise ValueError("sharpe_std must be >= 0.")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int):
        raise ValueError("n_trials must be a non-bool integer.")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if n_trials == 1:
        return float(sharpe_mean)

    first_quantile = _standard_normal_ppf(1.0 - 1.0 / n_trials)
    second_quantile = _standard_normal_ppf(1.0 - 1.0 / (n_trials * e))
    expected_max = sharpe_mean + sharpe_std * (
        (1.0 - EULER_MASCHERONI) * first_quantile
        + EULER_MASCHERONI * second_quantile
    )
    return float(expected_max)


def compute_deflated_sharpe_ratio(
    returns,
    n_trials: int = 1,
    periods_per_year: int = 52,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Compute Deflated Sharpe Ratio using expected maximum Sharpe adjustment."""
    returns_series = _clean_returns(returns)
    if len(returns_series) < 3:
        return 0.0
    _validate_trials(n_trials)
    _validate_number(benchmark_sharpe, "benchmark_sharpe")
    if n_trials == 1:
        return compute_probabilistic_sharpe_ratio(
            returns_series,
            benchmark_sharpe=benchmark_sharpe,
            periods_per_year=periods_per_year,
        )

    sharpe_std = _estimate_annualized_sharpe_standard_error(
        returns_series,
        periods_per_year=periods_per_year,
    )
    expected_max_sharpe = estimate_expected_max_sharpe(
        sharpe_std=sharpe_std,
        n_trials=n_trials,
        sharpe_mean=0.0,
    )
    benchmark_for_dsr = max(float(benchmark_sharpe), expected_max_sharpe)
    return compute_probabilistic_sharpe_ratio(
        returns_series,
        benchmark_sharpe=benchmark_for_dsr,
        periods_per_year=periods_per_year,
    )


def normalize_metric_series(values, higher_is_better: bool = True) -> pd.Series:
    """Robust min-max normalize numeric values to [0, 1]."""
    series = pd.Series(values).apply(pd.to_numeric, errors="coerce")
    finite = series.replace([np.inf, -np.inf], np.nan)
    result = pd.Series(0.5, index=series.index, dtype=float)
    valid = finite.dropna()
    if valid.empty:
        return result

    min_value = float(valid.min())
    max_value = float(valid.max())
    if max_value == min_value:
        result.loc[valid.index] = 0.5
        return result.clip(0.0, 1.0)

    normalized = (finite - min_value) / (max_value - min_value)
    if not higher_is_better:
        normalized = 1.0 - normalized
    result.loc[valid.index] = normalized.loc[valid.index]
    return result.clip(0.0, 1.0)


def compute_discipline_score(metrics_df: pd.DataFrame) -> pd.Series:
    """Score cash, turnover, and diversification discipline in [0, 1]."""
    if metrics_df.empty:
        return pd.Series(dtype=float)

    cash_excess = _numeric_column(metrics_df, "unjustified_cash_excess", default=np.nan)
    cash_above_band = _numeric_column(metrics_df, "cash_above_10_rate", default=0.0)
    if cash_excess.isna().all():
        cash_excess = cash_above_band
    else:
        cash_excess = cash_excess.combine_first(cash_above_band)
    cash_discipline = (1.0 - cash_excess.fillna(0.0).clip(lower=0.0, upper=1.0)).clip(
        0.0,
        1.0,
    )

    turnover = _numeric_column(metrics_df, "turnover", default=np.nan)
    if turnover.isna().all():
        turnover = _numeric_column(metrics_df, "mean_average_turnover", default=0.0)
    turnover_excess = ((turnover.fillna(0.0) - 0.50) / 0.50).clip(lower=0.0, upper=1.0)
    turnover_discipline = (1.0 - turnover_excess).clip(0.0, 1.0)

    effective_assets = _numeric_column(metrics_df, "effective_assets", default=np.nan)
    if effective_assets.isna().all():
        effective_assets = _numeric_column(
            metrics_df,
            "mean_average_effective_number_of_assets",
            default=np.nan,
        )
    diversification = ((effective_assets.fillna(1.0) - 1.0) / 1.0).clip(0.0, 1.0)

    return (
        0.50 * cash_discipline
        + 0.30 * turnover_discipline
        + 0.20 * diversification
    ).clip(0.0, 1.0)


def compute_composite_robust_score(
    metrics_df: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Add component scores and composite robust_score to a metrics table."""
    if metrics_df.empty:
        return metrics_df.copy()
    score_weights = dict(DEFAULT_COMPOSITE_WEIGHTS)
    if weights is not None:
        unknown = set(weights) - set(score_weights)
        if unknown:
            raise ValueError(f"Unknown composite score weight keys: {sorted(unknown)}")
        for key, value in weights.items():
            _validate_number(value, key)
            if value < 0:
                raise ValueError(f"{key} must be >= 0.")
            score_weights[key] = float(value)
    weight_sum = sum(score_weights.values())
    if weight_sum <= 0:
        raise ValueError("Composite score weights must sum to a positive value.")

    result = metrics_df.copy()
    result = _ensure_dsr_score_columns(result)

    result["sortino_score"] = normalize_metric_series(
        _first_available_numeric(result, ["sortino", "mean_sortino"]),
    )
    result["calmar_score"] = normalize_metric_series(
        _first_available_numeric(result, ["calmar", "mean_calmar"]),
    )
    result["drawdown_score"] = normalize_metric_series(
        _first_available_numeric(result, ["max_drawdown", "mean_max_drawdown"]),
    )
    std_sharpe = _first_available_numeric(result, ["std_sharpe"])
    worst_drawdown = _first_available_numeric(result, ["worst_drawdown", "worst_max_drawdown"])
    result["stability_score"] = (
        0.70 * normalize_metric_series(std_sharpe, higher_is_better=False)
        + 0.30 * normalize_metric_series(worst_drawdown)
    ).clip(0.0, 1.0)
    result["discipline_score"] = compute_discipline_score(result)

    result["robust_score"] = 0.0
    for component, weight in score_weights.items():
        result["robust_score"] += result[component].fillna(0.5) * weight
    result["robust_score"] = (result["robust_score"] / weight_sum).clip(0.0, 1.0)
    return result


def build_robust_score_report(
    comparison_dir: str,
    split: str = "test",
    periods_per_year: int = 52,
    benchmark_sharpe: float = 0.0,
    n_trials_effective: int = 25,
    output_dir: str | None = None,
) -> dict:
    """Build robust-score ranking files for an existing comparison output."""
    _validate_trials(n_trials_effective)
    comparison_path = Path(comparison_dir)
    if not comparison_path.exists():
        raise FileNotFoundError(f"Comparison directory not found: {comparison_dir}")
    output_path = Path(output_dir) if output_dir is not None else comparison_path
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_path = comparison_path / "overall_aggregate_by_strategy_split.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing aggregate metrics file: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[metrics["split"] == split].copy()
    if metrics.empty:
        raise ValueError(f"No aggregate metrics rows found for split: {split}")

    metrics = _rename_report_columns(metrics)
    warnings: list[str] = []
    pooled_dsr_n10_values = []
    pooled_dsr_n25_values = []
    pooled_dsr_n50_values = []
    mean_run_dsr_n25_values = []
    median_run_dsr_n25_values = []
    min_run_dsr_n25_values = []
    max_run_dsr_n25_values = []
    date_averaged_dsr_n25_values = []
    psr_values = []
    dsr_available = []
    dsr_methods = []
    for _, row in metrics.iterrows():
        observations = _load_return_observations_for_strategy(
            comparison_path=comparison_path,
            strategy=str(row["strategy"]),
            strategy_type=str(row["type"]),
            split=split,
        )
        if observations is None or observations.empty:
            pooled_dsr_n10_values.append(np.nan)
            pooled_dsr_n25_values.append(np.nan)
            pooled_dsr_n50_values.append(np.nan)
            mean_run_dsr_n25_values.append(np.nan)
            median_run_dsr_n25_values.append(np.nan)
            min_run_dsr_n25_values.append(np.nan)
            max_run_dsr_n25_values.append(np.nan)
            date_averaged_dsr_n25_values.append(np.nan)
            psr_values.append(np.nan)
            dsr_available.append(False)
            dsr_methods.append("fallback_from_sharpe")
            warnings.append(
                f"{row['strategy']}: returns unavailable; DSR falls back to Sharpe normalization.",
            )
            continue
        returns = observations["return"]
        psr_values.append(
            compute_probabilistic_sharpe_ratio(
                returns,
                benchmark_sharpe=benchmark_sharpe,
                periods_per_year=periods_per_year,
            ),
        )
        pooled_dsr_n10 = compute_deflated_sharpe_ratio(
            returns,
            n_trials=10,
            periods_per_year=periods_per_year,
            benchmark_sharpe=benchmark_sharpe,
        )
        pooled_dsr_n25 = compute_deflated_sharpe_ratio(
            returns,
            n_trials=25,
            periods_per_year=periods_per_year,
            benchmark_sharpe=benchmark_sharpe,
        )
        pooled_dsr_n50 = compute_deflated_sharpe_ratio(
            returns,
            n_trials=50,
            periods_per_year=periods_per_year,
            benchmark_sharpe=benchmark_sharpe,
        )
        run_dsr_values = _compute_run_level_dsr(
            observations=observations,
            n_trials=n_trials_effective,
            periods_per_year=periods_per_year,
            benchmark_sharpe=benchmark_sharpe,
        )
        date_averaged_dsr_n25 = _compute_date_averaged_dsr(
            observations=observations,
            n_trials=n_trials_effective,
            periods_per_year=periods_per_year,
            benchmark_sharpe=benchmark_sharpe,
        )
        pooled_dsr_n10_values.append(pooled_dsr_n10)
        pooled_dsr_n25_values.append(pooled_dsr_n25)
        pooled_dsr_n50_values.append(pooled_dsr_n50)
        mean_run_dsr_n25_values.append(_finite_mean(run_dsr_values))
        median_run_dsr_n25_values.append(_finite_median(run_dsr_values))
        min_run_dsr_n25_values.append(_finite_min(run_dsr_values))
        max_run_dsr_n25_values.append(_finite_max(run_dsr_values))
        date_averaged_dsr_n25_values.append(date_averaged_dsr_n25)
        dsr_available.append(True)
        dsr_methods.append(
            _select_dsr_method(
                median_run_dsr_n25_values[-1],
                date_averaged_dsr_n25,
                pooled_dsr_n25,
            ),
        )

    metrics["psr_score"] = psr_values
    metrics["pooled_dsr_n10"] = pooled_dsr_n10_values
    metrics["pooled_dsr_n25"] = pooled_dsr_n25_values
    metrics["pooled_dsr_n50"] = pooled_dsr_n50_values
    metrics["dsr_n10"] = metrics["pooled_dsr_n10"]
    metrics["dsr_n25"] = metrics["pooled_dsr_n25"]
    metrics["dsr_n50"] = metrics["pooled_dsr_n50"]
    metrics["mean_run_dsr_n25"] = mean_run_dsr_n25_values
    metrics["median_run_dsr_n25"] = median_run_dsr_n25_values
    metrics["min_run_dsr_n25"] = min_run_dsr_n25_values
    metrics["max_run_dsr_n25"] = max_run_dsr_n25_values
    metrics["date_averaged_dsr_n25"] = date_averaged_dsr_n25_values
    metrics["dsr_available"] = dsr_available
    metrics["dsr_method"] = dsr_methods
    metrics = _ensure_dsr_score_columns(metrics)
    metrics = _merge_diagnostic_columns(comparison_path, metrics)
    scored = compute_composite_robust_score(metrics)

    ranking_columns = [
        "strategy",
        "type",
        "robust_score",
        "dsr_score",
        "pooled_dsr_n10",
        "pooled_dsr_n25",
        "pooled_dsr_n50",
        "dsr_n10",
        "dsr_n25",
        "dsr_n50",
        "mean_run_dsr_n25",
        "median_run_dsr_n25",
        "min_run_dsr_n25",
        "max_run_dsr_n25",
        "date_averaged_dsr_n25",
        "dsr_available",
        "dsr_method",
        "sortino_score",
        "calmar_score",
        "drawdown_score",
        "stability_score",
        "discipline_score",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "worst_drawdown",
        "turnover",
        "effective_assets",
        "cash_above_10_rate",
        "unjustified_cash_excess",
    ]
    ranking = scored.sort_values("robust_score", ascending=False).reset_index(drop=True)
    ranking_path = output_path / "robust_score_ranking.csv"
    details_path = output_path / "robust_score_component_details.csv"
    warnings_path = output_path / "robust_score_warnings.txt"
    ranking.loc[:, ranking_columns].to_csv(ranking_path, index=False)
    scored.to_csv(details_path, index=False)

    warning_lines = [POOLED_DSR_WARNING]
    if warnings:
        warning_lines.extend(warnings)
        warnings_text = "\n".join(warning_lines)
    else:
        warning_lines.append(
            "DSR uses Bailey/Lopez de Prado expected maximum Sharpe adjustment "
            f"with n_trials_effective={n_trials_effective}; sensitivity computed "
            "for n_trials 10/25/50."
        )
        warnings_text = "\n".join(warning_lines)
    warnings_path.write_text(warnings_text + "\n")

    return {
        "ranking": ranking.loc[:, ranking_columns],
        "component_details": scored,
        "warnings": warnings_text,
        "ranking_path": str(ranking_path),
        "component_details_path": str(details_path),
        "warnings_path": str(warnings_path),
    }


def _clean_returns(returns) -> pd.Series:
    series = pd.Series(returns).apply(pd.to_numeric, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    return series.astype(float)


def _ensure_dsr_score_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    if "dsr_score" in result:
        if "dsr_available" not in result:
            result["dsr_available"] = _numeric_column(result, "dsr_score").notna()
        if "dsr_method" not in result:
            result["dsr_method"] = "provided"
        return result

    median_run = _numeric_column(result, "median_run_dsr_n25", default=np.nan)
    date_averaged = _numeric_column(result, "date_averaged_dsr_n25", default=np.nan)
    pooled = _numeric_column(result, "pooled_dsr_n25", default=np.nan)
    selected = median_run.combine_first(date_averaged).combine_first(pooled)
    fallback = normalize_metric_series(_numeric_column(result, "sharpe", default=np.nan))
    result["dsr_score"] = selected.combine_first(fallback)

    methods = pd.Series("fallback_from_sharpe", index=result.index, dtype=object)
    methods.loc[pooled.notna()] = "pooled"
    methods.loc[date_averaged.notna()] = "date_averaged"
    methods.loc[median_run.notna()] = "median_run"
    result["dsr_method"] = methods
    result["dsr_available"] = selected.notna()
    return result


def _select_dsr_method(
    median_run_dsr: float,
    date_averaged_dsr: float,
    pooled_dsr: float,
) -> str:
    if np.isfinite(median_run_dsr):
        return "median_run"
    if np.isfinite(date_averaged_dsr):
        return "date_averaged"
    if np.isfinite(pooled_dsr):
        return "pooled"
    return "fallback_from_sharpe"


def _compute_run_level_dsr(
    observations: pd.DataFrame,
    n_trials: int,
    periods_per_year: int,
    benchmark_sharpe: float,
) -> pd.Series:
    if observations.empty or "run_id" not in observations:
        return pd.Series(dtype=float)
    values = []
    for _, group in observations.groupby("run_id", sort=False):
        returns = _clean_returns(group["return"])
        if returns.empty:
            continue
        values.append(
            compute_deflated_sharpe_ratio(
                returns,
                n_trials=n_trials,
                periods_per_year=periods_per_year,
                benchmark_sharpe=benchmark_sharpe,
            ),
        )
    return pd.Series(values, dtype=float)


def _compute_date_averaged_dsr(
    observations: pd.DataFrame,
    n_trials: int,
    periods_per_year: int,
    benchmark_sharpe: float,
) -> float:
    if observations.empty or "date" not in observations:
        return np.nan
    dated = observations.dropna(subset=["date"]).copy()
    if dated.empty:
        return np.nan
    date_averaged_returns = (
        dated.groupby("date", sort=True)["return"].mean().replace([np.inf, -np.inf], np.nan).dropna()
    )
    if date_averaged_returns.empty:
        return np.nan
    return compute_deflated_sharpe_ratio(
        date_averaged_returns,
        n_trials=n_trials,
        periods_per_year=periods_per_year,
        benchmark_sharpe=benchmark_sharpe,
    )


def _finite_mean(values: pd.Series) -> float:
    valid = _finite_values(values)
    return float(valid.mean()) if not valid.empty else np.nan


def _finite_median(values: pd.Series) -> float:
    valid = _finite_values(values)
    return float(valid.median()) if not valid.empty else np.nan


def _finite_min(values: pd.Series) -> float:
    valid = _finite_values(values)
    return float(valid.min()) if not valid.empty else np.nan


def _finite_max(values: pd.Series) -> float:
    valid = _finite_values(values)
    return float(valid.max()) if not valid.empty else np.nan


def _finite_values(values: pd.Series) -> pd.Series:
    return pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()


def _standard_normal_cdf(value: float) -> float:
    if not isfinite(value):
        return 0.0 if value < 0 else 1.0
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _standard_normal_ppf(value: float) -> float:
    if value <= 0.0 or value >= 1.0:
        raise ValueError("normal inverse CDF input must be between 0 and 1.")
    return float(NormalDist().inv_cdf(value))


def _estimate_annualized_sharpe_standard_error(
    returns: pd.Series,
    periods_per_year: int,
) -> float:
    _validate_periods_per_year(periods_per_year)
    returns_series = _clean_returns(returns)
    if len(returns_series) < 3:
        return 0.0
    volatility = returns_series.std(ddof=1)
    if volatility <= 1e-12 or not np.isfinite(volatility):
        return 0.0

    periodic_sharpe = float(returns_series.mean() / volatility)
    skewness = float(returns_series.skew())
    kurtosis = float(returns_series.kurt()) + 3.0
    if not all(np.isfinite([periodic_sharpe, skewness, kurtosis])):
        return 0.0

    variance_term = max(
        1.0
        - skewness * periodic_sharpe
        + ((kurtosis - 1.0) / 4.0) * periodic_sharpe**2,
        0.0,
    )
    periodic_standard_error = sqrt(variance_term / (len(returns_series) - 1.0))
    return float(periodic_standard_error * sqrt(periods_per_year))


def _clip_probability(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _validate_periods_per_year(periods_per_year: int) -> None:
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int):
        raise ValueError("periods_per_year must be a non-bool integer.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")


def _validate_number(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-bool number.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _validate_trials(n_trials: int) -> None:
    if isinstance(n_trials, bool) or not isinstance(n_trials, int):
        raise ValueError("n_trials must be a non-bool integer.")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")


def _numeric_column(df: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _first_available_numeric(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in df:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _rename_report_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    renamed = metrics.rename(
        columns={
            "strategy_type": "type",
            "mean_sharpe": "sharpe",
            "mean_sortino": "sortino",
            "mean_calmar": "calmar",
            "mean_max_drawdown": "max_drawdown",
            "worst_max_drawdown": "worst_drawdown",
            "mean_average_turnover": "turnover",
            "mean_average_effective_number_of_assets": "effective_assets",
        },
    ).copy()
    if "type" not in renamed:
        renamed["type"] = "unknown"
    return renamed


def _merge_diagnostic_columns(comparison_path: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    cash_path = comparison_path / "test_cash_attribution_aggregate_by_drl_strategy.csv"
    if cash_path.exists():
        cash = pd.read_csv(cash_path).rename(
            columns={"mean_unjustified_cash_excess": "diagnostic_unjustified_cash_excess"},
        )
        cash = cash.loc[:, ["strategy", "diagnostic_unjustified_cash_excess"]]
        result = result.merge(cash, on="strategy", how="left")
        result["unjustified_cash_excess"] = result[
            "diagnostic_unjustified_cash_excess"
        ].combine_first(_numeric_column(result, "unjustified_cash_excess", default=np.nan))
        result = result.drop(columns=["diagnostic_unjustified_cash_excess"])
    if "unjustified_cash_excess" not in result:
        result["unjustified_cash_excess"] = np.nan
    return result


def _load_returns_for_strategy(
    comparison_path: Path,
    strategy: str,
    strategy_type: str,
    split: str,
) -> pd.Series | None:
    observations = _load_return_observations_for_strategy(
        comparison_path,
        strategy,
        strategy_type,
        split,
    )
    if observations is None or observations.empty:
        return None
    return observations["return"].dropna()


def _load_return_observations_for_strategy(
    comparison_path: Path,
    strategy: str,
    strategy_type: str,
    split: str,
) -> pd.DataFrame | None:
    if strategy_type == "drl":
        parts = []
        for history_path in sorted(comparison_path.glob(f"F*_{strategy}_seed_*/{split}_policy_history.csv")):
            history = pd.read_csv(history_path)
            return_column = (
                "financial_net_return"
                if "financial_net_return" in history.columns
                else "portfolio_return"
            )
            if return_column in history:
                returns = pd.to_numeric(history[return_column], errors="coerce")
                dates = _history_dates(history, returns.index)
                parts.append(
                    pd.DataFrame(
                        {
                            "run_id": history_path.parent.name,
                            "date": dates,
                            "return": returns,
                        },
                    ),
                )
        if parts:
            return _clean_return_observations(pd.concat(parts, ignore_index=True))
        return None

    equity_path = comparison_path / "benchmark_equity_curves_by_fold.csv"
    if not equity_path.exists():
        return None
    equity = pd.read_csv(equity_path)
    equity = equity.loc[(equity["split"] == split) & (equity["strategy"] == strategy)].copy()
    if equity.empty:
        return None
    parts = []
    for _, group in equity.groupby("fold"):
        group = group.sort_values("date")
        returns = pd.to_numeric(group["equity_curve"], errors="coerce").pct_change().dropna()
        if not returns.empty:
            parts.append(
                pd.DataFrame(
                    {
                        "run_id": f"{strategy}_{group['fold'].iloc[0]}",
                        "date": pd.to_datetime(group.loc[returns.index, "date"], errors="coerce"),
                        "return": returns,
                    },
                ),
            )
    if not parts:
        return None
    return _clean_return_observations(pd.concat(parts, ignore_index=True))


def _history_dates(history: pd.DataFrame, fallback_index: pd.Index) -> pd.Series:
    if "date" not in history:
        return pd.Series(pd.NaT, index=fallback_index)
    return pd.to_datetime(history["date"], errors="coerce")


def _clean_return_observations(observations: pd.DataFrame) -> pd.DataFrame:
    result = observations.copy()
    result["return"] = pd.to_numeric(result["return"], errors="coerce")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=["return"]).reset_index(drop=True)
