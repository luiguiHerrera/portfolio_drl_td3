"""Build the paper comparison from metrics computed per full-OOS TD3 seed.

This reporting-only layer starts from the already validated 228-date alignment.
For TD3, it concatenates the four disjoint test folds inside each seed, computes
all nonlinear metrics on each complete 228-date seed history, and only then
summarizes those metrics across the ten training seeds. The date-wise average
return path is retained as a synthetic diagnostic and is never substituted for
expected-seed performance.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.asset_specific_constraint_pareto_report import (
    build_constraint_pass_fail_matrix,
    build_feasible_strategy_rankings,
    build_pareto_tables,
    build_standard_metric_rankings,
)
from src.analysis.mandate_profile_comparison_report import (
    build_profile_rankings,
    build_profile_winners,
    score_strategies_for_profiles,
)
from src.analysis.paper_aligned_comparison import (
    ASSET_COST_COLUMNS,
    BASE_METRIC_COLUMNS,
    DATE_COLUMN,
    EXPECTED_BENCHMARKS,
    EXPECTED_SEEDS,
    EXPECTED_TD3_STRATEGIES,
    HISTORY_NUMERIC_COLUMNS,
    PERIODS_PER_YEAR,
    RETURN_COLUMN,
    SCORE_COMPONENT_COLUMNS,
    WEIGHT_COLUMNS,
    _json_safe,
    _selected_td3_specs,
    align_td3_candidate,
    build_td3_benchmark_deltas,
    compute_aligned_metrics,
    load_td3_candidate_histories,
    protocol_sources,
    reset_aligned_equity,
    score_aligned_universe,
    validate_output_directory,
)
from src.backtest.evaluate_policy import (
    annualized_return,
    annualized_volatility,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
)
from src.backtest.performance_metrics import calmar_ratio, sortino_ratio


PRIMARY_AGGREGATION_METHOD = "mean_of_metrics_across_ten_full_oos_seed_histories"
AVERAGE_PATH_METHOD = "datewise_mean_of_net_returns_across_ten_seed_policies"
MEDIAN_PATH_METHOD = "datewise_median_of_net_returns_across_ten_seed_policies_diagnostic_only"
CANONICAL_SHARPE_NAME = "annualized_arithmetic_mean_to_sample_standard_deviation_sharpe"
LEGACY_RATIO_NAME = "cagr_to_annualized_volatility_ratio"
WRC_STATISTIC_NAME = "max_mean_weekly_net_return_differential"

SUMMARY_METRICS = [
    *BASE_METRIC_COLUMNS,
    "dsr_score",
]


def build_paper_seed_aggregated_comparison(
    repo_root: str | Path,
    external_root: str | Path,
    aligned_dir: str | Path = "outputs/paper_aligned_comparison",
    output_dir: str | Path = "outputs/paper_seed_aggregated_comparison",
    expected_observations: int = 228,
) -> dict[str, Any]:
    """Generate a separate metrics-then-average package without overwriting alignment outputs."""
    repo = Path(repo_root).expanduser().resolve()
    external = Path(external_root).expanduser().resolve()
    aligned = _resolve_under_repo(repo, aligned_dir)
    output = _resolve_under_repo(repo, output_dir)
    _ensure_output_directories(output)

    aligned_validation = validate_output_directory(
        aligned,
        expected_observations=expected_observations,
    )
    if aligned_validation.get("status") != "PASS":
        raise ValueError("The prerequisite aligned package did not validate.")

    aligned_dates = pd.read_csv(
        aligned / "alignment/aligned_date_index.csv",
        parse_dates=[DATE_COLUMN],
    )
    aligned_histories = pd.read_csv(
        aligned / "histories/aligned_strategy_histories.csv",
        parse_dates=[DATE_COLUMN],
    )
    aligned_metrics = pd.read_csv(aligned / "metrics/aligned_strategy_metrics.csv")
    aligned_ranking = pd.read_csv(aligned / "ranking/aligned_combined_ranking.csv")
    aligned_constraints = pd.read_csv(
        aligned / "mandates/aligned_constraint_pass_fail_matrix.csv"
    )
    aligned_profiles = pd.read_csv(
        aligned / "mandates/aligned_mandate_profile_rankings.csv"
    )
    aligned_pareto = pd.read_csv(aligned / "pareto/aligned_pareto_frontier.csv")

    all_seed_histories: list[pd.DataFrame] = []
    all_seed_metrics: list[pd.DataFrame] = []
    all_metric_summaries: list[pd.DataFrame] = []
    all_primary_metrics: list[pd.DataFrame] = []
    all_average_paths: list[pd.DataFrame] = []
    all_median_metrics: list[dict[str, Any]] = []
    all_comparisons: list[pd.DataFrame] = []
    all_diversification: list[dict[str, Any]] = []
    all_cost_diagnostics: list[dict[str, Any]] = []
    all_seed_constraint_summaries: list[pd.DataFrame] = []
    all_rankings: list[pd.DataFrame] = []
    all_robust: list[pd.DataFrame] = []
    all_pairwise: list[pd.DataFrame] = []
    all_constraints: list[pd.DataFrame] = []
    all_feasible: list[pd.DataFrame] = []
    all_profiles: list[pd.DataFrame] = []
    all_profile_winners: list[pd.DataFrame] = []
    all_standard: list[pd.DataFrame] = []
    all_pareto: list[pd.DataFrame] = []
    all_dominated: list[pd.DataFrame] = []
    score_comparisons: list[pd.DataFrame] = []
    protocol_metadata: dict[str, Any] = {}

    sources = protocol_sources(repo, external)
    for source in sources:
        canonical = _canonical_index(aligned_dates, source.protocol, expected_observations)
        protocol_primary: list[dict[str, Any]] = []
        protocol_seed_metrics: list[pd.DataFrame] = []
        protocol_average_path_metrics: list[dict[str, Any]] = []

        for spec in _selected_td3_specs(source.td3_dir):
            loaded = load_td3_candidate_histories(
                protocol=source.protocol,
                td3_dir=source.td3_dir,
                base_candidate=spec["base_candidate"],
                cap_label=spec["cap_label"],
                repo_root=repo,
                external_root=external,
            )
            result = analyze_td3_seed_aggregation(loaded, canonical, source.cash_bps)
            all_seed_histories.append(result["seed_histories"])
            all_seed_metrics.append(result["seed_metrics"])
            all_metric_summaries.append(result["metric_summary"])
            all_average_paths.append(result["average_path"])
            all_median_metrics.append(result["median_path_metrics"])
            all_comparisons.append(result["aggregation_comparison"])
            all_diversification.append(result["diversification"])
            all_cost_diagnostics.append(result["cost_diagnostics"])
            protocol_primary.append(result["primary_metrics"])
            protocol_seed_metrics.append(result["seed_metrics"])
            protocol_average_path_metrics.append(result["average_path_metrics"])

        benchmark_metrics = aligned_metrics[
            (aligned_metrics["protocol"] == source.protocol)
            & (aligned_metrics["strategy_type"].str.lower() == "benchmark")
        ].copy()
        if len(benchmark_metrics) != EXPECTED_BENCHMARKS:
            raise ValueError(
                f"{source.protocol}: expected {EXPECTED_BENCHMARKS} aligned benchmarks."
            )
        benchmark_metrics["std_sharpe"] = np.nan
        benchmark_metrics["seed_dispersion_status"] = (
            "not_observed_neutral_not_zero_in_stability_component"
        )
        benchmark_metrics["aggregation_method"] = "single_deterministic_aligned_history"

        primary_td3 = pd.DataFrame(protocol_primary)
        primary_td3["seed_dispersion_status"] = (
            "observed_across_ten_full_oos_seed_histories"
        )
        primary_metrics = pd.concat(
            [primary_td3, benchmark_metrics], ignore_index=True, sort=False
        )
        if len(primary_metrics) != EXPECTED_TD3_STRATEGIES + EXPECTED_BENCHMARKS:
            raise ValueError(f"{source.protocol}: unexpected primary strategy count.")

        ranking = score_aligned_universe(primary_metrics)
        robust = ranking[
            [
                "protocol",
                "strategy_name",
                "strategy_type",
                *SCORE_COMPONENT_COLUMNS,
                "rank_robust",
                "rank_mandate_aware",
            ]
        ].copy()
        pairwise = build_td3_benchmark_deltas(ranking)

        constraint_input = ranking.copy()
        constraint_input["strategy_type"] = constraint_input["strategy_type"].str.lower()
        constraint_input["drawdown_severity"] = constraint_input["max_drawdown"].abs()
        constraints = build_constraint_pass_fail_matrix(constraint_input)
        feasible = build_feasible_strategy_rankings(constraints)
        standard = build_standard_metric_rankings(constraint_input)
        pareto, dominated = build_pareto_tables(constraint_input)
        profile_rankings = build_profile_rankings(
            score_strategies_for_profiles(constraint_input)
        )
        profile_winners = build_profile_winners(profile_rankings)
        for frame in [
            constraints,
            feasible,
            standard,
            pareto,
            dominated,
            profile_rankings,
            profile_winners,
        ]:
            frame.insert(0, "protocol", source.protocol)

        seed_constraint_summary = build_seed_constraint_summary(
            pd.concat(protocol_seed_metrics, ignore_index=True, sort=False)
        )
        seed_constraint_summary.insert(0, "protocol", source.protocol)

        average_path_td3 = pd.DataFrame(protocol_average_path_metrics)
        average_path_td3["seed_dispersion_status"] = (
            "observed_across_ten_full_oos_seed_histories"
        )
        average_path_metrics = pd.concat(
            [average_path_td3, benchmark_metrics], ignore_index=True, sort=False
        )
        controlled_average_path_ranking = score_aligned_universe(average_path_metrics)
        score_comparisons.append(
            build_score_method_comparison(
                aligned_ranking[aligned_ranking["protocol"] == source.protocol],
                controlled_average_path_ranking,
                ranking,
            )
        )

        all_primary_metrics.append(primary_metrics)
        all_rankings.append(ranking)
        all_robust.append(robust)
        all_pairwise.append(pairwise)
        all_constraints.append(constraints)
        all_feasible.append(feasible)
        all_profiles.append(profile_rankings)
        all_profile_winners.append(profile_winners)
        all_standard.append(standard)
        all_pareto.append(pareto)
        all_dominated.append(dominated)
        all_seed_constraint_summaries.append(seed_constraint_summary)

        protocol_metadata[source.protocol] = {
            "n_td3_strategies": len(primary_td3),
            "n_benchmarks": len(benchmark_metrics),
            "n_strategies": len(primary_metrics),
            "n_seeds_per_td3": len(EXPECTED_SEEDS),
            "common_observations": len(canonical),
            "common_start_date": canonical.min().date().isoformat(),
            "common_end_date": canonical.max().date().isoformat(),
            "cash_bps": source.cash_bps,
            "cash_return_assumption": source.cash_return_assumption,
        }

    seed_histories = pd.concat(all_seed_histories, ignore_index=True, sort=False)
    seed_metrics = pd.concat(all_seed_metrics, ignore_index=True, sort=False)
    metric_summary = pd.concat(all_metric_summaries, ignore_index=True, sort=False)
    primary_metrics = pd.concat(all_primary_metrics, ignore_index=True, sort=False)
    average_paths = pd.concat(all_average_paths, ignore_index=True, sort=False)
    median_metrics = pd.DataFrame(all_median_metrics)
    comparison = pd.concat(all_comparisons, ignore_index=True, sort=False)
    diversification = pd.DataFrame(all_diversification)
    cost_diagnostics = pd.DataFrame(all_cost_diagnostics)
    seed_constraint_summary = pd.concat(
        all_seed_constraint_summaries, ignore_index=True, sort=False
    )
    rankings = pd.concat(all_rankings, ignore_index=True, sort=False)
    robust = pd.concat(all_robust, ignore_index=True, sort=False)
    pairwise = pd.concat(all_pairwise, ignore_index=True, sort=False)
    constraints = pd.concat(all_constraints, ignore_index=True, sort=False)
    feasible = pd.concat(all_feasible, ignore_index=True, sort=False)
    profiles = pd.concat(all_profiles, ignore_index=True, sort=False)
    profile_winners = pd.concat(all_profile_winners, ignore_index=True, sort=False)
    standard = pd.concat(all_standard, ignore_index=True, sort=False)
    pareto = pd.concat(all_pareto, ignore_index=True, sort=False)
    dominated = pd.concat(all_dominated, ignore_index=True, sort=False)
    score_comparison = pd.concat(score_comparisons, ignore_index=True, sort=False)

    statistical_reference = load_named_statistical_reference(sources, rankings)
    audit = build_current_vs_seed_audit(
        aligned_ranking=aligned_ranking,
        seed_ranking=rankings,
        aligned_constraints=aligned_constraints,
        seed_constraints=constraints,
        aligned_profiles=aligned_profiles,
        seed_profiles=profiles,
        aligned_pareto=aligned_pareto,
        seed_pareto=pareto,
    )
    methodology = build_methodology_metadata(
        repo=repo,
        aligned=aligned,
        protocol_metadata=protocol_metadata,
    )
    paths = write_outputs(
        output=output,
        aligned=aligned,
        aligned_dates=aligned_dates,
        aligned_histories=aligned_histories,
        seed_histories=seed_histories,
        seed_metrics=seed_metrics,
        metric_summary=metric_summary,
        primary_metrics=primary_metrics,
        average_paths=average_paths,
        median_metrics=median_metrics,
        comparison=comparison,
        diversification=diversification,
        cost_diagnostics=cost_diagnostics,
        score_comparison=score_comparison,
        seed_constraint_summary=seed_constraint_summary,
        rankings=rankings,
        robust=robust,
        pairwise=pairwise,
        constraints=constraints,
        feasible=feasible,
        profiles=profiles,
        profile_winners=profile_winners,
        standard=standard,
        pareto=pareto,
        dominated=dominated,
        statistical_reference=statistical_reference,
        audit=audit,
        methodology=methodology,
        protocol_metadata=protocol_metadata,
    )
    validation = validate_seed_aggregated_output(
        output,
        expected_observations=expected_observations,
    )
    validation_path = output / "metadata/validation_summary.json"
    validation_path.write_text(json.dumps(_json_safe(validation), indent=2), encoding="utf-8")
    paths["validation_summary"] = str(validation_path)
    return {
        "output_dir": str(output),
        "paths": paths,
        "validation": validation,
        "rankings": rankings,
        "metrics": primary_metrics,
        "constraints": constraints,
        "pareto": pareto,
    }


def analyze_td3_seed_aggregation(
    loaded: dict[str, Any],
    common_index: pd.DatetimeIndex,
    cash_bps: float,
) -> dict[str, Any]:
    """Compute per-seed metrics before aggregation and all requested diagnostics."""
    seed_histories: list[pd.DataFrame] = []
    seed_metric_rows: list[dict[str, Any]] = []
    return_columns: dict[int, pd.Series] = {}
    stacked = loaded["stacked"].copy()
    for seed, group in stacked.groupby("seed", sort=True):
        history = build_full_seed_history(
            group,
            common_index,
            protocol=loaded["protocol"],
            strategy_name=loaded["strategy_name"],
            seed=int(seed),
        )
        seed_histories.append(history)
        metrics = compute_aligned_metrics(
            history,
            protocol=loaded["protocol"],
            strategy_name=loaded["strategy_name"],
            strategy_type="TD3",
            base_candidate=loaded["base_candidate"],
            cap_label=loaded["cap_label"],
            stability=None,
        )
        metrics["seed"] = int(seed)
        metrics["aggregation_stage"] = "metric_computed_before_cross_seed_aggregation"
        seed_metric_rows.append(metrics)
        return_columns[int(seed)] = pd.Series(
            history[RETURN_COLUMN].to_numpy(dtype=float),
            index=common_index,
        )
    seed_history_frame = pd.concat(seed_histories, ignore_index=True, sort=False)
    seed_metrics = pd.DataFrame(seed_metric_rows)
    if sorted(seed_metrics["seed"].unique()) != EXPECTED_SEEDS:
        raise ValueError(f"{loaded['strategy_name']}: unexpected seed set.")

    average_path, _, stability = align_td3_candidate(loaded, common_index)
    average_path_metrics = compute_aligned_metrics(
        average_path,
        protocol=loaded["protocol"],
        strategy_name=loaded["strategy_name"],
        strategy_type="TD3",
        base_candidate=loaded["base_candidate"],
        cap_label=loaded["cap_label"],
        stability=stability,
    )
    average_path_metrics["aggregation_method"] = AVERAGE_PATH_METHOD

    metric_summary = summarize_seed_metrics(seed_metrics)
    metric_summary.insert(0, "candidate", loaded["strategy_name"])
    metric_summary.insert(0, "protocol", loaded["protocol"])
    primary = aggregate_seed_metrics(seed_metrics, average_path_metrics)
    primary["aggregation_method"] = PRIMARY_AGGREGATION_METHOD
    primary["expected_seed_estimand"] = True

    returns = pd.DataFrame(return_columns)
    median_return_path = returns.median(axis=1)
    median_path_metrics = {
        "protocol": loaded["protocol"],
        "strategy_name": loaded["strategy_name"],
        "aggregation_method": MEDIAN_PATH_METHOD,
        **compute_return_only_metrics(median_return_path),
    }
    comparison = build_aggregation_comparison(
        metric_summary,
        average_path_metrics,
        median_path_metrics,
    )
    diversification = compute_diversification_diagnostics(
        loaded["protocol"],
        loaded["strategy_name"],
        returns,
        seed_metrics,
        average_path_metrics,
    )
    cost_diagnostics = compute_cost_diagnostics(
        loaded["protocol"],
        loaded["strategy_name"],
        stacked,
        cash_bps,
    )
    average_path = average_path.copy()
    average_path["path_role"] = "synthetic_average_return_diagnostic_not_primary"
    return {
        "seed_histories": seed_history_frame,
        "seed_metrics": seed_metrics,
        "metric_summary": metric_summary,
        "primary_metrics": primary,
        "average_path": average_path,
        "average_path_metrics": average_path_metrics,
        "median_path_metrics": median_path_metrics,
        "aggregation_comparison": comparison,
        "diversification": diversification,
        "cost_diagnostics": cost_diagnostics,
    }


def build_full_seed_history(
    group: pd.DataFrame,
    common_index: pd.DatetimeIndex,
    *,
    protocol: str,
    strategy_name: str,
    seed: int,
) -> pd.DataFrame:
    """Return one complete 228-date OOS seed history after exact index validation."""
    ordered = group.sort_values(DATE_COLUMN).copy()
    index = pd.DatetimeIndex(ordered[DATE_COLUMN], name=DATE_COLUMN)
    if not index.equals(common_index):
        raise ValueError(f"{protocol} {strategy_name} seed {seed}: common-index mismatch.")
    numeric = [column for column in HISTORY_NUMERIC_COLUMNS if column in ordered]
    history = reset_aligned_equity(ordered[[DATE_COLUMN, *numeric]].copy())
    history.insert(0, "seed", seed)
    history.insert(0, "strategy_type", "TD3")
    history.insert(0, "strategy_name", strategy_name)
    history.insert(0, "protocol", protocol)
    history["transaction_cost_mode"] = "asset_specific"
    history["gross_return"] = history.get("portfolio_return", np.nan)
    history["net_return"] = history[RETURN_COLUMN]
    return history


def summarize_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize metrics after each was computed on a full seed history."""
    rows: list[dict[str, Any]] = []
    for metric in SUMMARY_METRICS:
        values = pd.to_numeric(seed_metrics[metric], errors="raise")
        rows.append(
            {
                "metric": metric,
                "mean_of_seed_metrics": float(values.mean()),
                "median_of_seed_metrics": float(values.median()),
                "std_of_seed_metrics": float(values.std(ddof=1)),
                "minimum_of_seed_metrics": float(values.min()),
                "maximum_of_seed_metrics": float(values.max()),
                "p10_of_seed_metrics": float(values.quantile(0.10)),
                "p90_of_seed_metrics": float(values.quantile(0.90)),
                "n_seeds": int(values.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_seed_metrics(
    seed_metrics: pd.DataFrame,
    average_path_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build the primary expected-seed row; never recompute nonlinear metrics on a mean path."""
    result = dict(average_path_metrics)
    for metric in BASE_METRIC_COLUMNS:
        result[metric] = float(pd.to_numeric(seed_metrics[metric], errors="raise").mean())
    result["std_sharpe"] = float(
        pd.to_numeric(seed_metrics["sharpe"], errors="raise").std(ddof=0)
    )
    result["worst_max_drawdown"] = float(
        pd.to_numeric(seed_metrics["max_drawdown"], errors="raise").min()
    )
    result["dsr_score"] = float(
        pd.to_numeric(seed_metrics["dsr_score"], errors="raise").median()
    )
    result["median_aligned_seed_dsr_n25"] = result["dsr_score"]
    result["dsr_method"] = "median_dsr_across_full_oos_seed_histories_n25"
    result["n_aligned_seeds"] = int(seed_metrics["seed"].nunique())
    return result


def compute_return_only_metrics(returns: pd.Series) -> dict[str, float]:
    """Compute return-path metrics for the diagnostic median path."""
    return {
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(returns, PERIODS_PER_YEAR),
        "annualized_volatility": annualized_volatility(returns, PERIODS_PER_YEAR),
        "sharpe": sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR),
        "sortino": sortino_ratio(returns, periods_per_year=PERIODS_PER_YEAR),
        "calmar": calmar_ratio(returns, periods_per_year=PERIODS_PER_YEAR),
        "max_drawdown": max_drawdown(returns),
    }


def build_aggregation_comparison(
    summary: pd.DataFrame,
    average_path_metrics: dict[str, Any],
    median_path_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Compare metrics-first summaries with mean/median synthetic paths."""
    result = summary.copy()
    result["metric_of_mean_path"] = result["metric"].map(average_path_metrics)
    result["metric_of_median_return_path"] = result["metric"].map(median_path_metrics)
    result["difference"] = (
        result["metric_of_mean_path"] - result["mean_of_seed_metrics"]
    )
    denominator = result["mean_of_seed_metrics"].abs().replace(0.0, np.nan)
    result["relative_difference"] = result["difference"] / denominator
    result["primary_value"] = result["mean_of_seed_metrics"]
    result["primary_method"] = PRIMARY_AGGREGATION_METHOD
    return result


def compute_diversification_diagnostics(
    protocol: str,
    strategy_name: str,
    returns: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    average_path_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Measure mechanical cross-seed smoothing in the average-return path."""
    correlation = returns.corr().to_numpy(dtype=float)
    mean_pairwise = float(correlation[np.triu_indices_from(correlation, k=1)].mean())
    mean_vol = float(seed_metrics["annualized_volatility"].mean())
    mean_drawdown = float(seed_metrics["max_drawdown"].mean())
    mean_sharpe = float(seed_metrics["sharpe"].mean())
    path_vol = float(average_path_metrics["annualized_volatility"])
    path_drawdown = float(average_path_metrics["max_drawdown"])
    path_sharpe = float(average_path_metrics["sharpe"])
    return {
        "protocol": protocol,
        "candidate": strategy_name,
        "mean_pairwise_seed_return_correlation": mean_pairwise,
        "mean_weekly_cross_seed_return_dispersion": float(
            returns.std(axis=1, ddof=1).mean()
        ),
        "mean_individual_annualized_volatility": mean_vol,
        "average_path_annualized_volatility": path_vol,
        "volatility_reduction_pct": 1.0 - path_vol / mean_vol,
        "mean_individual_max_drawdown": mean_drawdown,
        "average_path_max_drawdown": path_drawdown,
        "drawdown_severity_reduction_pct": 1.0 - abs(path_drawdown) / abs(mean_drawdown),
        "mean_individual_sharpe": mean_sharpe,
        "average_path_sharpe": path_sharpe,
        "sharpe_increase": path_sharpe - mean_sharpe,
        "interpretation": "mechanical_cross_policy_diversification_not_algorithm_stability",
    }


def compute_cost_diagnostics(
    protocol: str,
    strategy_name: str,
    stacked: pd.DataFrame,
    cash_bps: float,
) -> dict[str, Any]:
    """Contrast averaged separately executed costs with netted average-weight costs."""
    weight_frames = [
        group.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)[WEIGHT_COLUMNS]
        for _, group in stacked.groupby("seed", sort=True)
    ]
    average_weights = sum(weight_frames) / len(weight_frames)
    weight_changes = average_weights.diff().abs().iloc[1:]
    cost_rates = pd.Series(
        {
            "weight_SPY": 0.0002,
            "weight_TLT": 0.0002,
            "weight_GLD": 0.0002,
            "weight_BTC-USD": 0.0010,
            "weight_CASH": cash_bps / 10_000.0,
        }
    )
    netted_turnover = weight_changes.sum(axis=1)
    netted_cost = weight_changes.mul(cost_rates, axis=1).sum(axis=1)
    stored_turnover = stacked.groupby(DATE_COLUMN)["turnover"].mean().iloc[1:]
    stored_cost = stacked.groupby(DATE_COLUMN)["transaction_cost"].mean().iloc[1:]
    return {
        "protocol": protocol,
        "candidate": strategy_name,
        "mean_separately_computed_turnover": float(stored_turnover.mean()),
        "turnover_of_average_weights": float(netted_turnover.mean()),
        "netting_turnover_reduction_pct": float(
            1.0 - netted_turnover.mean() / stored_turnover.mean()
        ),
        "mean_separately_computed_transaction_cost": float(stored_cost.mean()),
        "transaction_cost_of_average_weight_changes": float(netted_cost.mean()),
        "netting_cost_reduction_pct": float(
            1.0 - netted_cost.mean() / stored_cost.mean()
        ),
        "first_row_excluded_from_netted_comparison": True,
        "current_average_path_cost_interpretation": (
            "mean of net returns and costs already computed in ten separately executed policies; "
            "not costs recomputed from one netted ensemble weight path"
        ),
    }


def build_seed_constraint_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Report how often individual seed histories satisfy each unchanged hard profile."""
    source = seed_metrics.copy()
    source["candidate"] = source["strategy_name"].astype(str)
    source["strategy_name"] = (
        source["candidate"] + "__seed_" + source["seed"].astype(int).astype(str)
    )
    source["strategy_type"] = "td3"
    matrix = build_constraint_pass_fail_matrix(source)
    matrix = matrix.merge(
        source[["strategy_name", "candidate", "seed"]],
        on="strategy_name",
        how="left",
    )
    return (
        matrix.groupby(["profile", "candidate"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_feasible_seeds=("feasible", "sum"),
            feasible_seed_rate=("feasible", "mean"),
            max_drawdown_pass_rate=("max_drawdown_pass", "mean"),
            volatility_pass_rate=("annualized_volatility_pass", "mean"),
            effective_assets_pass_rate=(
                "average_effective_number_of_assets_pass",
                "mean",
            ),
            turnover_pass_rate=("average_turnover_pass", "mean"),
        )
    )


def build_score_method_comparison(
    existing_average_path_ranking: pd.DataFrame,
    controlled_average_path_ranking: pd.DataFrame,
    seed_aggregated_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Separate aggregation effects from the benchmark-dispersion correction."""
    columns = [
        "protocol",
        "strategy_name",
        "strategy_type",
        "robust_score",
        "mandate_aware_score",
        "rank_robust",
        "rank_mandate_aware",
        "rank_sharpe",
    ]
    existing = existing_average_path_ranking[columns].rename(
        columns={column: f"{column}_existing" for column in columns if column not in {"protocol", "strategy_name"}}
    )
    controlled = controlled_average_path_ranking[columns].rename(
        columns={column: f"{column}_controlled_mean_path" for column in columns if column not in {"protocol", "strategy_name"}}
    )
    seed = seed_aggregated_ranking[columns].rename(
        columns={column: f"{column}_seed_aggregated" for column in columns if column not in {"protocol", "strategy_name"}}
    )
    result = existing.merge(controlled, on=["protocol", "strategy_name"]).merge(
        seed, on=["protocol", "strategy_name"]
    )
    result["delta_robust_seed_vs_controlled_mean_path"] = (
        result["robust_score_seed_aggregated"]
        - result["robust_score_controlled_mean_path"]
    )
    result["delta_mandate_seed_vs_controlled_mean_path"] = (
        result["mandate_aware_score_seed_aggregated"]
        - result["mandate_aware_score_controlled_mean_path"]
    )
    return result


def load_named_statistical_reference(
    sources: list[Any],
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """Load existing pairwise values with unambiguous estimator and WRC names."""
    rows: list[dict[str, Any]] = []
    for source in sources:
        protocol_ranking = rankings[rankings["protocol"] == source.protocol]
        candidate = str(
            protocol_ranking[
                protocol_ranking["strategy_type"].str.lower() == "td3"
            ].sort_values("rank_mandate_aware").iloc[0]["strategy_name"]
        )
        benchmark = "trend_spy_cash_12p"
        pairwise_path = source.statistical_dir / "statistical_validation_pairwise_bootstrap.csv"
        wrc_path = source.wrc_dir / "white_reality_check_summary.csv"
        pairwise = pd.read_csv(pairwise_path)
        selected = pairwise[
            (pairwise["candidate"] == candidate)
            & (pairwise["benchmark"] == benchmark)
            & (pairwise["metric"] == "sharpe")
        ]
        if len(selected) != 1:
            raise ValueError(f"{source.protocol}: missing selected legacy ratio row.")
        wrc = pd.read_csv(wrc_path)
        selected_wrc = wrc[wrc["benchmark"] == benchmark]
        if len(selected_wrc) != 1:
            raise ValueError(f"{source.protocol}: missing Trend WRC row.")
        pair = selected.iloc[0]
        wrc_row = selected_wrc.iloc[0]
        rows.append(
            {
                "protocol": source.protocol,
                "candidate": candidate,
                "benchmark": benchmark,
                "pairwise_statistic_name": LEGACY_RATIO_NAME,
                "pairwise_legacy_source_field": "sharpe",
                "pairwise_formula": "CAGR / (weekly sample std(ddof=1) * sqrt(52))",
                "pairwise_risk_free_rate_subtracted": False,
                "pairwise_input_path_role": "datewise_average_return_synthetic_path",
                "candidate_estimate": pair["candidate_estimate"],
                "benchmark_estimate": pair["benchmark_estimate"],
                "bootstrap_mean_delta": pair["mean_delta"],
                "lower_5pct_delta": pair["lower_5pct_delta"],
                "upper_95pct_delta": pair["upper_95pct_delta"],
                "probability_candidate_beats": pair["probability_candidate_beats"],
                "n_aligned_periods": int(pair["n_aligned_periods"]),
                "n_bootstrap": int(pair["n_bootstrap"]),
                "block_size": int(pair["block_size"]),
                "canonical_ranking_statistic_name": CANONICAL_SHARPE_NAME,
                "canonical_ranking_formula": "weekly arithmetic mean / sample std(ddof=1) * sqrt(52)",
                "numerically_comparable_to_canonical_sharpe": False,
                "wrc_statistic_name": WRC_STATISTIC_NAME,
                "wrc_formula": "sqrt(T) * max_j mean(candidate_j weekly net return - benchmark weekly net return)",
                "wrc_uses_cagr_to_volatility_ratio": False,
                "wrc_p_value": wrc_row["p_value"],
                "wrc_n_candidates": int(wrc_row["n_candidates"]),
                "wrc_block_length": int(wrc_row["block_length"]),
                "wrc_n_bootstrap": int(wrc_row["n_bootstrap"]),
                "pairwise_source_sha256": _sha256(pairwise_path),
                "wrc_source_sha256": _sha256(wrc_path),
            }
        )
    return pd.DataFrame(rows)


def build_current_vs_seed_audit(
    *,
    aligned_ranking: pd.DataFrame,
    seed_ranking: pd.DataFrame,
    aligned_constraints: pd.DataFrame,
    seed_constraints: pd.DataFrame,
    aligned_profiles: pd.DataFrame,
    seed_profiles: pd.DataFrame,
    aligned_pareto: pd.DataFrame,
    seed_pareto: pd.DataFrame,
) -> dict[str, pd.DataFrame | str]:
    """Compare the current average-path package with the seed-aggregated package."""
    metrics = [
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "robust_score",
        "mandate_aware_score",
        "rank_sharpe",
        "rank_robust",
        "rank_mandate_aware",
    ]
    left = aligned_ranking[["protocol", "strategy_name", *metrics]].rename(
        columns={metric: f"{metric}_average_path" for metric in metrics}
    )
    right = seed_ranking[["protocol", "strategy_name", *metrics]].rename(
        columns={metric: f"{metric}_seed_aggregated" for metric in metrics}
    )
    ranking_changes = left.merge(right, on=["protocol", "strategy_name"])
    for metric in metrics:
        ranking_changes[f"delta_{metric}"] = (
            ranking_changes[f"{metric}_seed_aggregated"]
            - ranking_changes[f"{metric}_average_path"]
        )

    constraint_changes = aligned_constraints.merge(
        seed_constraints,
        on=["protocol", "profile", "strategy_name"],
        suffixes=("_average_path", "_seed_aggregated"),
    )
    constraint_changes["feasibility_changed"] = (
        constraint_changes["feasible_average_path"].astype(bool)
        != constraint_changes["feasible_seed_aggregated"].astype(bool)
    )

    profile_changes = aligned_profiles.merge(
        seed_profiles,
        on=["protocol", "profile", "strategy_name"],
        suffixes=("_average_path", "_seed_aggregated"),
    )
    profile_changes["rank_changed"] = (
        profile_changes["profile_rank_average_path"]
        != profile_changes["profile_rank_seed_aggregated"]
    )

    old_set = set(
        map(tuple, aligned_pareto[["protocol", "frontier_type", "strategy_name"]].to_numpy())
    )
    new_set = set(
        map(tuple, seed_pareto[["protocol", "frontier_type", "strategy_name"]].to_numpy())
    )
    pareto_rows = []
    for key in sorted(old_set | new_set):
        pareto_rows.append(
            {
                "protocol": key[0],
                "frontier_type": key[1],
                "strategy_name": key[2],
                "on_average_path_frontier": key in old_set,
                "on_seed_aggregated_frontier": key in new_set,
                "membership_changed": (key in old_set) != (key in new_set),
            }
        )
    summary_lines = [
        "# Average-path versus metrics-then-average audit",
        "",
        "TD3 nonlinear metrics now estimate expected performance across a random training seed. ",
        "The average-return path remains a synthetic diagnostic and is not treated as a deployable policy.",
    ]
    return {
        "ranking_changes": ranking_changes,
        "constraint_changes": constraint_changes,
        "profile_changes": profile_changes,
        "pareto_changes": pd.DataFrame(pareto_rows),
        "summary": "\n".join(summary_lines) + "\n",
    }


def build_methodology_metadata(
    *,
    repo: Path,
    aligned: Path,
    protocol_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Record the estimand, formulas, asymmetric evidence, and source dependency."""
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "scripts/build_paper_seed_aggregated_comparison.py",
        "module": "src.analysis.paper_seed_aggregated_comparison",
        "git_commit": _git_commit(repo),
        "prerequisite_aligned_package": str(aligned.relative_to(repo)),
        "primary_estimand": "expected_seed_performance_of_the_training_algorithm",
        "primary_td3_aggregation": PRIMARY_AGGREGATION_METHOD,
        "operation_order": [
            "concatenate four disjoint OOS folds within seed",
            "compute each metric on each 228-date seed history",
            "take the arithmetic mean of the ten seed-level metric values for ranking",
            "report median, sample standard deviation, min, max, p10 and p90 separately",
        ],
        "prohibited_substitution": "metric(datewise_mean_return_path) is not mean(metric(seed))",
        "average_return_path": {
            "method": AVERAGE_PATH_METHOD,
            "role": "synthetic diagnostic only",
            "not_a_single_policy": True,
            "not_an_ex_ante_selected_policy": True,
            "not_a_fully_implemented_ensemble": True,
            "cost_interpretation": (
                "net returns, turnover and costs were computed per policy before averaging; "
                "they were not recomputed from netted average weights"
            ),
        },
        "median_return_path": {
            "method": MEDIAN_PATH_METHOD,
            "role": "return-only diagnostic; not deployable",
        },
        "benchmark_asymmetry": {
            "td3": "distribution over ten training seeds on one market path",
            "benchmarks": "one deterministic historical path and no training-seed distribution",
            "benchmark_seed_dispersion": "unavailable, represented as neutral/NaN rather than zero",
            "robust_score_status": (
                "retained with unchanged weights as a diagnostic only; not primary evidence of "
                "TD3-versus-benchmark superiority"
            ),
        },
        "canonical_sharpe": {
            "name": CANONICAL_SHARPE_NAME,
            "formula": "weekly arithmetic mean / sample std(ddof=1) * sqrt(52)",
            "risk_free_rate": 0.0,
            "nan_treatment": "drop missing returns before calculation",
            "ranking_input": "mean of ten per-seed canonical Sharpe values for TD3",
        },
        "pairwise_legacy_ratio": {
            "name": LEGACY_RATIO_NAME,
            "source_field": "sharpe",
            "formula": "CAGR / (weekly sample std(ddof=1) * sqrt(52))",
            "numerically_comparable_to_canonical_sharpe": False,
            "input_path_role": "datewise average-return synthetic path",
        },
        "white_reality_check": {
            "statistic_name": WRC_STATISTIC_NAME,
            "formula": "sqrt(T) * max mean weekly candidate-minus-benchmark net return differential",
            "uses_cagr_to_volatility_ratio": False,
        },
        "protocols": protocol_metadata,
    }


def write_outputs(
    *,
    output: Path,
    aligned: Path,
    aligned_dates: pd.DataFrame,
    aligned_histories: pd.DataFrame,
    seed_histories: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    metric_summary: pd.DataFrame,
    primary_metrics: pd.DataFrame,
    average_paths: pd.DataFrame,
    median_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    diversification: pd.DataFrame,
    cost_diagnostics: pd.DataFrame,
    score_comparison: pd.DataFrame,
    seed_constraint_summary: pd.DataFrame,
    rankings: pd.DataFrame,
    robust: pd.DataFrame,
    pairwise: pd.DataFrame,
    constraints: pd.DataFrame,
    feasible: pd.DataFrame,
    profiles: pd.DataFrame,
    profile_winners: pd.DataFrame,
    standard: pd.DataFrame,
    pareto: pd.DataFrame,
    dominated: pd.DataFrame,
    statistical_reference: pd.DataFrame,
    audit: dict[str, pd.DataFrame | str],
    methodology: dict[str, Any],
    protocol_metadata: dict[str, Any],
) -> dict[str, str]:
    """Write the reconstructible seed-aggregated package."""
    benchmark_histories = aligned_histories[
        aligned_histories["strategy_type"].str.lower() == "benchmark"
    ].copy()
    files: dict[str, Path] = {
        "aligned_date_index": output / "alignment/aligned_date_index.csv",
        "alignment_inventory": output / "alignment/alignment_inventory.csv",
        "alignment_metadata": output / "alignment/alignment_metadata.json",
        "td3_seed_histories": output / "histories/aligned_td3_seed_histories.csv",
        "benchmark_histories": output / "histories/aligned_benchmark_histories.csv",
        "average_paths": output / "histories/average_return_synthetic_paths.csv",
        "td3_seed_metrics": output / "metrics/td3_per_seed_metrics.csv",
        "td3_metric_summary": output / "metrics/td3_seed_metric_summary.csv",
        "strategy_metrics": output / "metrics/seed_aggregated_strategy_metrics.csv",
        "benchmark_metrics": output / "metrics/benchmark_metrics.csv",
        "median_path_metrics": output / "diagnostics/median_return_path_metrics.csv",
        "aggregation_comparison": output / "diagnostics/aggregation_method_comparison.csv",
        "diversification": output / "diagnostics/seed_diversification_effect.csv",
        "cost_diagnostics": output / "diagnostics/ensemble_cost_turnover_comparison.csv",
        "score_comparison": output / "diagnostics/score_method_comparison.csv",
        "seed_constraints": output / "diagnostics/td3_seed_constraint_summary.csv",
        "ranking": output / "ranking/seed_aggregated_combined_ranking.csv",
        "robust": output / "ranking/seed_aggregated_robust_scores.csv",
        "pairwise": output / "ranking/seed_aggregated_td3_vs_benchmarks.csv",
        "standard": output / "ranking/seed_aggregated_standard_metric_rankings.csv",
        "statistical_reference": output / "ranking/named_statistical_reference.csv",
        "constraints": output / "mandates/seed_aggregated_constraint_pass_fail_matrix.csv",
        "feasible": output / "mandates/seed_aggregated_feasible_strategy_rankings.csv",
        "profiles": output / "mandates/seed_aggregated_mandate_profile_rankings.csv",
        "profile_winners": output / "mandates/seed_aggregated_mandate_profile_winners.csv",
        "pareto": output / "pareto/seed_aggregated_pareto_frontier.csv",
        "dominated": output / "pareto/seed_aggregated_pareto_dominated.csv",
        "audit_ranking": output / "audit/average_path_vs_seed_aggregated_ranking.csv",
        "audit_constraints": output / "audit/average_path_vs_seed_constraint_changes.csv",
        "audit_profiles": output / "audit/average_path_vs_seed_profile_changes.csv",
        "audit_pareto": output / "audit/average_path_vs_seed_pareto_changes.csv",
        "audit_summary": output / "audit/average_path_vs_seed_summary.md",
        "methodology": output / "metadata/methodology.json",
        "source_lineage": output / "metadata/source_lineage.json",
        "paper_macros": output / "paper/seed_results_macros.tex",
        "paper_combined_rows": output / "paper/seed_combined_table_rows.tex",
        "paper_statistical_rows": output / "paper/named_statistical_table_rows.tex",
    }
    aligned_dates.to_csv(files["aligned_date_index"], index=False, date_format="%Y-%m-%d")
    pd.read_csv(aligned / "alignment/alignment_inventory.csv").to_csv(
        files["alignment_inventory"], index=False
    )
    seed_histories.to_csv(files["td3_seed_histories"], index=False, date_format="%Y-%m-%d")
    benchmark_histories.to_csv(files["benchmark_histories"], index=False, date_format="%Y-%m-%d")
    average_paths.to_csv(files["average_paths"], index=False, date_format="%Y-%m-%d")
    seed_metrics.to_csv(files["td3_seed_metrics"], index=False)
    metric_summary.to_csv(files["td3_metric_summary"], index=False)
    primary_metrics.to_csv(files["strategy_metrics"], index=False)
    primary_metrics[primary_metrics["strategy_type"].str.lower() == "benchmark"].to_csv(
        files["benchmark_metrics"], index=False
    )
    median_metrics.to_csv(files["median_path_metrics"], index=False)
    comparison.to_csv(files["aggregation_comparison"], index=False)
    diversification.to_csv(files["diversification"], index=False)
    cost_diagnostics.to_csv(files["cost_diagnostics"], index=False)
    score_comparison.to_csv(files["score_comparison"], index=False)
    seed_constraint_summary.to_csv(files["seed_constraints"], index=False)
    rankings.to_csv(files["ranking"], index=False)
    robust.to_csv(files["robust"], index=False)
    pairwise.to_csv(files["pairwise"], index=False)
    standard.to_csv(files["standard"], index=False)
    statistical_reference.to_csv(files["statistical_reference"], index=False)
    constraints.to_csv(files["constraints"], index=False)
    feasible.to_csv(files["feasible"], index=False)
    profiles.to_csv(files["profiles"], index=False)
    profile_winners.to_csv(files["profile_winners"], index=False)
    pareto.to_csv(files["pareto"], index=False)
    dominated.to_csv(files["dominated"], index=False)
    assert isinstance(audit["ranking_changes"], pd.DataFrame)
    assert isinstance(audit["constraint_changes"], pd.DataFrame)
    assert isinstance(audit["profile_changes"], pd.DataFrame)
    assert isinstance(audit["pareto_changes"], pd.DataFrame)
    audit["ranking_changes"].to_csv(files["audit_ranking"], index=False)
    audit["constraint_changes"].to_csv(files["audit_constraints"], index=False)
    audit["profile_changes"].to_csv(files["audit_profiles"], index=False)
    audit["pareto_changes"].to_csv(files["audit_pareto"], index=False)
    files["audit_summary"].write_text(str(audit["summary"]), encoding="utf-8")
    files["methodology"].write_text(
        json.dumps(_json_safe(methodology), indent=2), encoding="utf-8"
    )
    source_lineage = json.loads(
        (aligned / "metadata/source_lineage.json").read_text(encoding="utf-8")
    )
    source_lineage["derived_package"] = {
        "method": PRIMARY_AGGREGATION_METHOD,
        "prerequisite_aligned_package": str(aligned),
    }
    files["source_lineage"].write_text(
        json.dumps(_json_safe(source_lineage), indent=2), encoding="utf-8"
    )
    files["alignment_metadata"].write_text(
        json.dumps(
            {
                "method": "reuse validated exact 228-date alignment; no date regeneration",
                "prerequisite_validation": "PASS",
                "no_source_overwrite": True,
                "protocols": protocol_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    macros, combined_rows, statistical_rows = build_paper_artifacts(
        rankings=rankings,
        constraints=constraints,
        pareto=pareto,
        statistical_reference=statistical_reference,
    )
    files["paper_macros"].write_text(macros, encoding="utf-8")
    files["paper_combined_rows"].write_text(combined_rows, encoding="utf-8")
    files["paper_statistical_rows"].write_text(statistical_rows, encoding="utf-8")
    return {name: str(path) for name, path in files.items()}


def build_paper_artifacts(
    *,
    rankings: pd.DataFrame,
    constraints: pd.DataFrame,
    pareto: pd.DataFrame,
    statistical_reference: pd.DataFrame,
) -> tuple[str, str, str]:
    """Generate manuscript numbers from the seed-aggregated package."""
    selected: dict[str, dict[str, pd.Series]] = {}
    for protocol in ["zero_cash", "bil_cash"]:
        rows = rankings[rankings["protocol"] == protocol]
        td3 = rows[rows["strategy_type"].str.lower() == "td3"].sort_values(
            "rank_mandate_aware"
        )
        selected[protocol] = {
            "gld": rows[rows["strategy_name"] == "BuyHold_GLD"].iloc[0],
            "td3": td3.iloc[0],
            "trend": rows[rows["strategy_name"] == "trend_spy_cash_12p"].iloc[0],
        }
    macros: dict[str, Any] = {
        "AlignedWeeks": int(rankings["n_observations"].iloc[0]),
        "AlignedStartDate": str(rankings["start_date"].iloc[0]),
        "AlignedEndDate": str(rankings["end_date"].iloc[0]),
        "ZeroBestTDThreeMandateRank": int(selected["zero_cash"]["td3"]["rank_mandate_aware"]),
        "BilBestTDThreeMandateRank": int(selected["bil_cash"]["td3"]["rank_mandate_aware"]),
        "ZeroBestTDThreeSharpeRank": int(selected["zero_cash"]["td3"]["rank_sharpe"]),
        "BilBestTDThreeSharpeRank": int(selected["bil_cash"]["td3"]["rank_sharpe"]),
        "ZeroTDThreeSharpe": f"{selected['zero_cash']['td3']['sharpe']:.4f}",
        "BilTDThreeSharpe": f"{selected['bil_cash']['td3']['sharpe']:.4f}",
        "ZeroBestTDThreeLabel": _paper_label(selected["zero_cash"]["td3"]["strategy_name"]),
        "BilBestTDThreeLabel": _paper_label(selected["bil_cash"]["td3"]["strategy_name"]),
        "AggressiveFeasibleCount": int(
            constraints[
                (constraints["protocol"] == "zero_cash")
                & (constraints["profile"] == "aggressive")
                & constraints["feasible"]
            ].shape[0]
        ),
        "TDThreeFeasibleCount": int(
            constraints[(constraints["strategy_type"] == "td3") & constraints["feasible"]].shape[0]
        ),
        "ZeroParetoTDThreeCount": int(
            pareto[
                (pareto["protocol"] == "zero_cash")
                & (pareto["frontier_type"] == "reduced")
                & (pareto["strategy_type"] == "td3")
            ]["strategy_name"].nunique()
        ),
        "BilParetoTDThreeCount": int(
            pareto[
                (pareto["protocol"] == "bil_cash")
                & (pareto["frontier_type"] == "reduced")
                & (pareto["strategy_type"] == "td3")
            ]["strategy_name"].nunique()
        ),
    }
    for protocol, prefix in [("zero_cash", "Zero"), ("bil_cash", "Bil")]:
        stat = statistical_reference[statistical_reference["protocol"] == protocol].iloc[0]
        macros.update(
            {
                f"{prefix}BootstrapMeanDelta": f"{stat['bootstrap_mean_delta']:.4f}",
                f"{prefix}BootstrapLower": f"{stat['lower_5pct_delta']:.4f}",
                f"{prefix}BootstrapUpper": f"{stat['upper_95pct_delta']:.4f}",
                f"{prefix}BootstrapProbability": f"{stat['probability_candidate_beats']:.3f}",
                f"{prefix}WrcPValue": f"{stat['wrc_p_value']:.4f}",
            }
        )
    macro_text = "% Generated by scripts/build_paper_seed_aggregated_comparison.py; do not edit.\n"
    macro_text += "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    )
    macro_text += "\n"

    lines = ["% Generated from metrics computed per full-OOS TD3 seed; do not edit."]
    for protocol, cash in [("zero_cash", "\\cashzero{}"), ("bil_cash", "\\cashbil{}")]:
        for key, strategy_type in [("gld", "Benchmark"), ("td3", "TD3"), ("trend", "Benchmark")]:
            row = selected[protocol][key]
            lines.append(
                f"{cash} & {_paper_label(row['strategy_name'])} & {strategy_type} & "
                f"{int(row['rank_mandate_aware'])} & {row['annualized_return']:.4f} & "
                f"{row['annualized_volatility']:.4f} & {row['sharpe']:.4f} & "
                f"{row['max_drawdown']:.4f} & {row['mandate_aware_score']:.4f} & "
                f"{row['robust_score']:.4f} \\\\"
            )
    lines.append("\\bottomrule")

    statistical_lines = [
        "% Existing mean-path bootstrap/WRC diagnostics with corrected names; do not edit."
    ]
    for protocol, cash in [("zero_cash", "\\cashzero{}"), ("bil_cash", "\\cashbil{}")]:
        stat = statistical_reference[statistical_reference["protocol"] == protocol].iloc[0]
        statistical_lines.append(
            f"{cash} & {_paper_label(stat['candidate'])} vs Trend SPY/CASH & "
            f"{stat['bootstrap_mean_delta']:.4f} & "
            f"[{stat['lower_5pct_delta']:.4f}, {stat['upper_95pct_delta']:.4f}] & "
            f"{stat['probability_candidate_beats']:.3f} & {stat['wrc_p_value']:.4f} & "
            "No statistical superiority claim is supported. \\\\"
        )
    statistical_lines.append("\\bottomrule")
    return (
        macro_text,
        "\n".join(lines) + "\n",
        "\n".join(statistical_lines) + "\n",
    )


def validate_seed_aggregated_output(
    output_dir: str | Path,
    *,
    expected_observations: int = 228,
) -> dict[str, Any]:
    """Validate order of operations, formulas, downstream outputs, and metadata."""
    output = Path(output_dir)
    seed_histories = pd.read_csv(
        output / "histories/aligned_td3_seed_histories.csv",
        parse_dates=[DATE_COLUMN],
    )
    seed_metrics = pd.read_csv(output / "metrics/td3_per_seed_metrics.csv")
    metrics = pd.read_csv(output / "metrics/seed_aggregated_strategy_metrics.csv")
    ranking = pd.read_csv(output / "ranking/seed_aggregated_combined_ranking.csv")
    constraints = pd.read_csv(
        output / "mandates/seed_aggregated_constraint_pass_fail_matrix.csv"
    )
    profiles = pd.read_csv(
        output / "mandates/seed_aggregated_mandate_profile_rankings.csv"
    )
    pareto = pd.read_csv(output / "pareto/seed_aggregated_pareto_frontier.csv")
    statistical = pd.read_csv(output / "ranking/named_statistical_reference.csv")
    methodology = json.loads((output / "metadata/methodology.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    duplicate_dates = int(
        seed_histories.duplicated(["protocol", "strategy_name", "seed", DATE_COLUMN]).sum()
    )
    missing_values = int(seed_histories[RETURN_COLUMN].isna().sum())
    groups = seed_histories.groupby(["protocol", "strategy_name", "seed"], sort=True)
    if groups.ngroups != 2 * EXPECTED_TD3_STRATEGIES * len(EXPECTED_SEEDS):
        errors.append("Unexpected TD3 seed-history group count.")
    for key, group in groups:
        if len(group) != expected_observations:
            errors.append(f"{key}: expected {expected_observations} observations.")
        if not (group[DATE_COLUMN].dt.dayofweek == 4).all():
            errors.append(f"{key}: non-Friday date.")

    recomputed_rows: list[dict[str, Any]] = []
    for (protocol, strategy_name, seed), history in groups:
        source = seed_metrics[
            (seed_metrics["protocol"] == protocol)
            & (seed_metrics["strategy_name"] == strategy_name)
            & (seed_metrics["seed"] == seed)
        ]
        if len(source) != 1:
            errors.append(f"{protocol} {strategy_name} seed {seed}: metric row missing.")
            continue
        row = source.iloc[0]
        returns = pd.Series(history.sort_values(DATE_COLUMN)[RETURN_COLUMN].to_numpy(dtype=float))
        expected = compute_return_only_metrics(returns)
        for metric, value in expected.items():
            if not np.isclose(float(row[metric]), value, rtol=1e-10, atol=1e-12):
                errors.append(f"{protocol} {strategy_name} seed {seed}: {metric} mismatch.")
        recomputed_rows.append(row.to_dict())

    aggregation_pass = True
    for (protocol, strategy_name), group in seed_metrics.groupby(
        ["protocol", "strategy_name"], sort=True
    ):
        primary = metrics[
            (metrics["protocol"] == protocol)
            & (metrics["strategy_name"] == strategy_name)
        ]
        if len(primary) != 1:
            errors.append(f"{protocol} {strategy_name}: primary metric row missing.")
            aggregation_pass = False
            continue
        row = primary.iloc[0]
        for metric in BASE_METRIC_COLUMNS:
            expected = float(pd.to_numeric(group[metric], errors="raise").mean())
            if not np.isclose(float(row[metric]), expected, rtol=1e-10, atol=1e-12):
                errors.append(f"{protocol} {strategy_name}: aggregate mismatch {metric}.")
                aggregation_pass = False
        if row["aggregation_method"] != PRIMARY_AGGREGATION_METHOD:
            errors.append(f"{protocol} {strategy_name}: aggregation metadata mismatch.")
            aggregation_pass = False

    ranking_pass = True
    for protocol, group in metrics.groupby("protocol", sort=True):
        expected = score_aligned_universe(group.copy())
        actual = ranking[ranking["protocol"] == protocol]
        merged = expected.merge(actual, on="strategy_name", suffixes=("_expected", "_actual"))
        for metric in [*SCORE_COMPONENT_COLUMNS, "rank_robust", "rank_mandate_aware", "rank_sharpe"]:
            if not np.allclose(
                pd.to_numeric(merged[f"{metric}_expected"], errors="coerce"),
                pd.to_numeric(merged[f"{metric}_actual"], errors="coerce"),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                errors.append(f"{protocol}: ranking mismatch {metric}.")
                ranking_pass = False

        source = actual.copy()
        source["strategy_type"] = source["strategy_type"].str.lower()
        source["drawdown_severity"] = source["max_drawdown"].abs()
        expected_constraints = build_constraint_pass_fail_matrix(source)
        actual_constraints = constraints[constraints["protocol"] == protocol]
        if set(
            map(tuple, expected_constraints[["profile", "strategy_name", "feasible"]].to_numpy())
        ) != set(
            map(tuple, actual_constraints[["profile", "strategy_name", "feasible"]].to_numpy())
        ):
            errors.append(f"{protocol}: constraint mismatch.")
        expected_profiles = build_profile_rankings(score_strategies_for_profiles(source))
        actual_profiles = profiles[profiles["protocol"] == protocol]
        merged_profiles = expected_profiles.merge(
            actual_profiles,
            on=["profile", "strategy_name"],
            suffixes=("_expected", "_actual"),
        )
        if not np.allclose(
            merged_profiles["profile_score_expected"],
            merged_profiles["profile_score_actual"],
            rtol=1e-10,
            atol=1e-12,
        ):
            errors.append(f"{protocol}: profile mismatch.")
        expected_pareto, _ = build_pareto_tables(source)
        actual_pareto = pareto[pareto["protocol"] == protocol]
        if set(map(tuple, expected_pareto[["frontier_type", "strategy_name"]].to_numpy())) != set(
            map(tuple, actual_pareto[["frontier_type", "strategy_name"]].to_numpy())
        ):
            errors.append(f"{protocol}: Pareto mismatch.")

    benchmark_dispersion = metrics[metrics["strategy_type"].str.lower() == "benchmark"][
        "std_sharpe"
    ]
    if benchmark_dispersion.notna().any():
        errors.append("Deterministic benchmark seed dispersion must be unavailable, not zero.")
    if methodology.get("primary_td3_aggregation") != PRIMARY_AGGREGATION_METHOD:
        errors.append("Methodology does not declare the primary seed aggregation.")
    if set(statistical["pairwise_statistic_name"]) != {LEGACY_RATIO_NAME}:
        errors.append("Legacy pairwise ratio name mismatch.")
    if set(statistical["canonical_ranking_statistic_name"]) != {CANONICAL_SHARPE_NAME}:
        errors.append("Canonical Sharpe name mismatch.")
    if statistical["wrc_uses_cagr_to_volatility_ratio"].astype(bool).any():
        errors.append("WRC incorrectly marked as using the CAGR-to-volatility ratio.")
    if duplicate_dates:
        errors.append(f"Duplicate seed dates: {duplicate_dates}.")
    if missing_values:
        errors.append(f"Missing seed returns: {missing_values}.")
    if errors:
        raise ValueError("Seed-aggregated comparison validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "PASS",
        "primary_estimand": "expected_seed_performance_of_the_training_algorithm",
        "primary_td3_aggregation": PRIMARY_AGGREGATION_METHOD,
        "n_strategies_compared_per_protocol": {
            protocol: int(group["strategy_name"].nunique())
            for protocol, group in ranking.groupby("protocol")
        },
        "td3_seed_histories": int(groups.ngroups),
        "common_observations_per_seed": expected_observations,
        "duplicate_dates": duplicate_dates,
        "missing_values": missing_values,
        "per_seed_metrics_recomputed": "PASS",
        "metrics_computed_before_seed_aggregation": "PASS" if aggregation_pass else "FAIL",
        "ranking_derived_from_seed_aggregated_metrics": "PASS" if ranking_pass else "FAIL",
        "mandates_derived_from_seed_aggregated_metrics": "PASS",
        "pareto_derived_from_seed_aggregated_metrics": "PASS",
        "canonical_sharpe_and_legacy_ratio_distinct": "PASS",
        "wrc_return_differential_distinct_from_legacy_ratio": "PASS",
        "benchmark_seed_dispersion_not_imputed_as_zero": "PASS",
    }


def _canonical_index(
    dates: pd.DataFrame,
    protocol: str,
    expected_observations: int,
) -> pd.DatetimeIndex:
    selected = dates[dates["protocol"] == protocol].sort_values(DATE_COLUMN)
    index = pd.DatetimeIndex(selected[DATE_COLUMN], name=DATE_COLUMN)
    if len(index) != expected_observations or index.has_duplicates:
        raise ValueError(f"{protocol}: invalid prerequisite aligned index.")
    return index


def _paper_label(strategy_name: Any) -> str:
    labels = {
        "BuyHold_GLD": "BuyHold GLD",
        "trend_spy_cash_12p": "Trend SPY/CASH",
        "V3_real_macro_vintage_clean_no_dxy_cap_0p70": "V3 clean macro cap 0.70",
        "V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80": "V7 macro+GARCH cap 0.80",
    }
    return labels.get(str(strategy_name), str(strategy_name).replace("_", " "))


def _resolve_under_repo(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


def _ensure_output_directories(output: Path) -> None:
    for name in [
        "alignment",
        "histories",
        "metrics",
        "ranking",
        "mandates",
        "pareto",
        "diagnostics",
        "metadata",
        "audit",
        "paper",
    ]:
        (output / name).mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
