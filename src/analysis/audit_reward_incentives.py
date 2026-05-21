"""Audit reward incentives using saved TD3 protocol outputs.

This module is reporting-only. It does not change reward logic, training,
environment dynamics, or model code. The goal is to diagnose whether saved TD3
policies look consistent with useful concentrated timing or with low-signal
concentration that may be indirectly encouraged by cost/turnover penalties.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PRIMARY_DIR = "outputs/tables/protocol_pure_td3_revalidation_60ep_10seeds"
DEFAULT_BASELINE_DIR = "outputs/tables/protocol_pure_td3_revalidation_30ep_5seeds"
DEFAULT_OUTPUT_DIR = "outputs/tables/reward_incentive_audit"


def compute_reward_incentive_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add concentration, turnover, and lazy/justified concentration flags."""
    result = df.copy()
    for column in [
        "average_turnover",
        "average_effective_number_of_assets",
        "sharpe",
        "robust_score",
        "max_drawdown",
        "abs_validation_test_sharpe_gap",
    ]:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["high_concentration_flag"] = (
        result["average_effective_number_of_assets"] < 1.5
    )
    result["extreme_concentration_flag"] = (
        result["average_effective_number_of_assets"] < 1.2
    )
    result["high_turnover_flag"] = result["average_turnover"] > 0.50
    result["low_turnover_high_concentration_flag"] = (
        (result["average_turnover"] < 0.25)
        & (result["average_effective_number_of_assets"] < 1.5)
    )
    weak_performance = (
        (result["sharpe"] < 0.50)
        | (result["robust_score"] < 0.30)
        | (result["abs_validation_test_sharpe_gap"] > 0.75)
    )
    result["lazy_concentration_candidate"] = (
        result["high_concentration_flag"] & weak_performance
    )
    result["justified_concentration_candidate"] = (
        result["high_concentration_flag"]
        & (result["sharpe"] >= 0.75)
        & (result["robust_score"] >= 0.45)
        & (result["max_drawdown"] >= -0.25)
        & (
            result["abs_validation_test_sharpe_gap"].isna()
            | (result["abs_validation_test_sharpe_gap"] <= 0.75)
        )
    )
    return result


def load_reward_incentive_metrics(
    experiment_dir: str,
    comparison_dir: str | None = None,
    mandate_dir: str | None = None,
    experiment_label: str | None = None,
) -> pd.DataFrame:
    """Load TD3 candidate metrics from a protocol-pure revalidation directory."""
    experiment_path = Path(experiment_dir)
    overall_path = experiment_path / "overall_aggregate_by_strategy_split.csv"
    if not overall_path.exists():
        raise FileNotFoundError(f"Missing aggregate metrics: {overall_path}")

    overall = pd.read_csv(overall_path)
    td3_rows = overall.loc[
        overall["strategy_type"].astype(str).str.lower().isin(["drl", "td3"])
    ].copy()
    if td3_rows.empty:
        raise ValueError(f"No TD3/DRL rows found in {overall_path}")

    test = td3_rows.loc[td3_rows["split"].astype(str) == "test"].copy()
    validation = td3_rows.loc[td3_rows["split"].astype(str) == "validation"].copy()
    validation_sharpe = validation.set_index("strategy")["mean_sharpe"]

    rows = []
    for _, row in test.iterrows():
        strategy = row["strategy"]
        validation_value = validation_sharpe.get(strategy, pd.NA)
        test_sharpe = row.get("mean_sharpe", pd.NA)
        validation_test_gap = _safe_subtract(test_sharpe, validation_value)
        rows.append(
            {
                "experiment_label": experiment_label or experiment_path.name,
                "strategy": strategy,
                "candidate_name": strategy,
                "strategy_type": "td3",
                "n_folds": row.get("n_folds", pd.NA),
                "n_seeds": row.get("n_seeds", pd.NA),
                "n_observations": row.get("n_observations", pd.NA),
                "sharpe": row.get("mean_sharpe", pd.NA),
                "sortino": row.get("mean_sortino", pd.NA),
                "calmar": row.get("mean_calmar", pd.NA),
                "cumulative_return": row.get("mean_cumulative_return", pd.NA),
                "annualized_return": row.get("mean_annualized_return", pd.NA),
                "annualized_volatility": row.get(
                    "mean_annualized_volatility",
                    pd.NA,
                ),
                "max_drawdown": row.get("mean_max_drawdown", pd.NA),
                "worst_max_drawdown": row.get("worst_max_drawdown", pd.NA),
                "average_turnover": row.get("mean_average_turnover", pd.NA),
                "mean_transaction_cost": row.get("mean_transaction_cost", pd.NA),
                "average_effective_number_of_assets": row.get(
                    "mean_average_effective_number_of_assets",
                    pd.NA,
                ),
                "average_max_weight": row.get("mean_average_max_weight", pd.NA),
                "mean_cash_weight": row.get("mean_cash_weight", pd.NA),
                "cash_above_10_rate": row.get("cash_above_10_rate", pd.NA),
                "unjustified_cash_excess": row.get(
                    "unjustified_cash_excess",
                    pd.NA,
                ),
                "mean_cash_penalty": row.get("mean_cash_penalty", pd.NA),
                "mean_cash_breach": row.get("mean_cash_breach", pd.NA),
                "mean_turnover_penalty": row.get("mean_turnover_penalty", pd.NA),
                "validation_sharpe": validation_value,
                "validation_test_sharpe_gap": validation_test_gap,
                "abs_validation_test_sharpe_gap": (
                    abs(validation_test_gap)
                    if pd.notna(validation_test_gap)
                    else pd.NA
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics = _merge_robust_scores(metrics, experiment_path / "robust_score_ranking.csv")
    if comparison_dir is not None:
        metrics = _merge_protocol_comparison(metrics, Path(comparison_dir))
    if mandate_dir is not None:
        metrics = _merge_mandate_scores(metrics, Path(mandate_dir))
    return compute_reward_incentive_flags(metrics)


def compare_experiment_folders(
    baseline_dir: str,
    primary_dir: str,
    baseline_label: str = "30ep_5seeds",
    primary_label: str = "60ep_10seeds",
) -> pd.DataFrame:
    """Compare candidate behavior between two protocol-pure revalidation folders."""
    baseline = load_reward_incentive_metrics(
        baseline_dir,
        experiment_label=baseline_label,
    )
    primary = load_reward_incentive_metrics(primary_dir, experiment_label=primary_label)
    combined = pd.concat([baseline, primary], ignore_index=True)
    wide = combined.pivot_table(
        index="strategy",
        columns="experiment_label",
        values=[
            "sharpe",
            "robust_score",
            "average_turnover",
            "average_effective_number_of_assets",
            "average_max_weight",
            "max_drawdown",
            "worst_max_drawdown",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{label}" for metric, label in wide.columns]
    wide = wide.reset_index()

    for metric in [
        "sharpe",
        "robust_score",
        "average_turnover",
        "average_effective_number_of_assets",
        "average_max_weight",
        "max_drawdown",
        "worst_max_drawdown",
    ]:
        baseline_col = f"{metric}_{baseline_label}"
        primary_col = f"{metric}_{primary_label}"
        if baseline_col in wide.columns and primary_col in wide.columns:
            wide[f"delta_{metric}"] = wide[primary_col] - wide[baseline_col]
    return wide


def write_reward_incentive_audit(
    primary_dir: str = DEFAULT_PRIMARY_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    comparison_dir: str | None = None,
    mandate_dir: str | None = None,
    baseline_dir: str | None = DEFAULT_BASELINE_DIR,
    baseline_comparison_dir: str | None = None,
    baseline_mandate_dir: str | None = None,
) -> dict:
    """Write reward incentive audit CSVs from existing protocol outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    primary = load_reward_incentive_metrics(
        primary_dir,
        comparison_dir=comparison_dir,
        mandate_dir=mandate_dir,
        experiment_label=Path(primary_dir).name,
    )
    audit = _select_audit_columns(primary)
    flags = _select_flag_columns(primary)
    summary = build_candidate_behavior_summary(primary)

    audit_path = output_path / "reward_concentration_turnover_audit.csv"
    flags_path = output_path / "reward_lazy_concentration_flags.csv"
    summary_path = output_path / "reward_candidate_behavior_summary.csv"
    audit.to_csv(audit_path, index=False)
    flags.to_csv(flags_path, index=False)
    summary.to_csv(summary_path, index=False)

    paths = {
        "reward_concentration_turnover_audit": str(audit_path),
        "reward_lazy_concentration_flags": str(flags_path),
        "reward_candidate_behavior_summary": str(summary_path),
    }
    comparison = pd.DataFrame()
    if baseline_dir and Path(baseline_dir).exists() and Path(primary_dir).exists():
        comparison = compare_experiment_folders(
            baseline_dir=baseline_dir,
            primary_dir=primary_dir,
            baseline_label=Path(baseline_dir).name,
            primary_label=Path(primary_dir).name,
        )
        comparison_path = output_path / "reward_30ep_vs_60ep_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        paths["reward_30ep_vs_60ep_comparison"] = str(comparison_path)

    return {
        "output_dir": str(output_path),
        "audit": audit,
        "flags": flags,
        "summary": summary,
        "comparison": comparison,
        "paths": paths,
    }


def build_candidate_behavior_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a compact behavior summary by candidate."""
    columns = [
        "strategy",
        "sharpe",
        "robust_score",
        "mandate_aware_score",
        "max_drawdown",
        "worst_max_drawdown",
        "average_turnover",
        "mean_transaction_cost",
        "average_effective_number_of_assets",
        "average_max_weight",
        "mean_cash_weight",
        "cash_above_10_rate",
        "validation_test_sharpe_gap",
        "abs_validation_test_sharpe_gap",
        "lazy_concentration_candidate",
        "justified_concentration_candidate",
    ]
    summary = metrics.copy()
    for column in columns:
        if column not in summary.columns:
            summary[column] = pd.NA
    return summary.loc[:, columns].sort_values(
        ["robust_score", "sharpe"],
        ascending=[False, False],
        na_position="last",
    )


def _merge_robust_scores(metrics: pd.DataFrame, robust_path: Path) -> pd.DataFrame:
    if not robust_path.exists():
        metrics["robust_score"] = pd.NA
        return metrics
    robust = pd.read_csv(robust_path)
    robust = robust.loc[
        robust["type"].astype(str).str.lower().isin(["drl", "td3"])
    ].copy()
    robust = robust.rename(columns={"strategy": "strategy"})
    keep = [
        column
        for column in [
            "strategy",
            "robust_score",
            "dsr_score",
            "median_run_dsr_n25",
            "date_averaged_dsr_n25",
            "dsr_method",
        ]
        if column in robust.columns
    ]
    return metrics.merge(robust.loc[:, keep], on="strategy", how="left")


def _merge_protocol_comparison(metrics: pd.DataFrame, comparison_path: Path) -> pd.DataFrame:
    summary_path = comparison_path / "protocol_comparison_summary.csv"
    if not summary_path.exists():
        return metrics
    comparison = pd.read_csv(summary_path)
    comparison = comparison.loc[
        comparison["strategy_type"].astype(str).str.lower().eq("td3")
    ].copy()
    comparison = comparison.rename(columns={"strategy_name": "strategy"})
    keep = [
        column
        for column in [
            "strategy",
            "total_transaction_cost",
            "average_transaction_cost",
            "feature_version",
        ]
        if column in comparison.columns
    ]
    if not keep or keep == ["strategy"]:
        return metrics
    return metrics.merge(comparison.loc[:, keep], on="strategy", how="left")


def _merge_mandate_scores(metrics: pd.DataFrame, mandate_path: Path) -> pd.DataFrame:
    ranking_path = mandate_path / "mandate_aware_ranking.csv"
    if not ranking_path.exists():
        return metrics
    ranking = pd.read_csv(ranking_path)
    ranking = ranking.loc[
        ranking["strategy_type"].astype(str).str.lower().eq("td3")
    ].copy()
    ranking = ranking.rename(columns={"strategy_name": "strategy"})
    keep = [
        column
        for column in [
            "strategy",
            "mandate_aware_score",
            "mandate_bucket",
            "drawdown_multiplier",
            "recovery_required",
        ]
        if column in ranking.columns
    ]
    return metrics.merge(ranking.loc[:, keep], on="strategy", how="left")


def _select_audit_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_label",
        "strategy",
        "strategy_type",
        "sharpe",
        "sortino",
        "calmar",
        "robust_score",
        "mandate_aware_score",
        "max_drawdown",
        "worst_max_drawdown",
        "average_turnover",
        "mean_transaction_cost",
        "mean_turnover_penalty",
        "average_effective_number_of_assets",
        "average_max_weight",
        "mean_cash_weight",
        "cash_above_10_rate",
        "unjustified_cash_excess",
        "validation_sharpe",
        "validation_test_sharpe_gap",
        "abs_validation_test_sharpe_gap",
        "high_concentration_flag",
        "extreme_concentration_flag",
        "high_turnover_flag",
        "low_turnover_high_concentration_flag",
        "lazy_concentration_candidate",
        "justified_concentration_candidate",
    ]
    return _with_columns(metrics, columns)


def _select_flag_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_label",
        "strategy",
        "average_turnover",
        "average_effective_number_of_assets",
        "average_max_weight",
        "sharpe",
        "robust_score",
        "max_drawdown",
        "abs_validation_test_sharpe_gap",
        "high_concentration_flag",
        "extreme_concentration_flag",
        "high_turnover_flag",
        "low_turnover_high_concentration_flag",
        "lazy_concentration_candidate",
        "justified_concentration_candidate",
    ]
    return _with_columns(metrics, columns)


def _with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, columns]


def _safe_subtract(left, right):
    left_value = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_value = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    if pd.isna(left_value) or pd.isna(right_value):
        return pd.NA
    return float(left_value - right_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TD3 reward incentive behavior.")
    parser.add_argument("--primary-dir", default=DEFAULT_PRIMARY_DIR)
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--comparison-dir")
    parser.add_argument("--mandate-dir")
    parser.add_argument("--baseline-comparison-dir")
    parser.add_argument("--baseline-mandate-dir")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = write_reward_incentive_audit(
        primary_dir=args.primary_dir,
        output_dir=args.output_dir,
        comparison_dir=args.comparison_dir,
        mandate_dir=args.mandate_dir,
        baseline_dir=args.baseline_dir,
        baseline_comparison_dir=args.baseline_comparison_dir,
        baseline_mandate_dir=args.baseline_mandate_dir,
    )
    print(f"Output folder: {report['output_dir']}")
    print("\nReward concentration-turnover audit:")
    print(report["audit"].to_string(index=False))
    if not report["comparison"].empty:
        print("\n30ep vs 60ep comparison:")
        print(report["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
