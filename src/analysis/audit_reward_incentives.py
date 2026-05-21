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
HIGH_CONCENTRATION_EFFECTIVE_ASSETS = 1.5
EXTREME_CONCENTRATION_EFFECTIVE_ASSETS = 1.2
HIGH_TURNOVER_THRESHOLD = 0.50
LOW_TURNOVER_THRESHOLD = 0.25
JUSTIFIED_SHARPE_THRESHOLD = 0.75
JUSTIFIED_ROBUST_SCORE_THRESHOLD = 0.60
JUSTIFIED_MANDATE_AWARE_SCORE_THRESHOLD = 0.40
JUSTIFIED_MAX_DRAWDOWN_THRESHOLD = -0.25
JUSTIFIED_VALIDATION_TEST_GAP_THRESHOLD = 0.75


def compute_reward_incentive_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add concentration, turnover, and lazy/justified concentration flags."""
    result = df.copy()
    if "strategy" not in result.columns and "strategy_name" in result.columns:
        result["strategy"] = result["strategy_name"]
    for column in [
        "average_turnover",
        "average_effective_number_of_assets",
        "sharpe",
        "robust_score",
        "mandate_aware_score",
        "max_drawdown",
        "abs_validation_test_sharpe_gap",
    ]:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["strategy_role"] = result.apply(_classify_strategy_role, axis=1)
    result["concentration_origin"] = result["strategy_role"].apply(
        _classify_concentration_origin
    )
    result["high_concentration_flag"] = (
        result["average_effective_number_of_assets"]
        < HIGH_CONCENTRATION_EFFECTIVE_ASSETS
    )
    result["extreme_concentration_flag"] = (
        result["average_effective_number_of_assets"]
        < EXTREME_CONCENTRATION_EFFECTIVE_ASSETS
    )
    result["high_turnover_flag"] = result["average_turnover"] > HIGH_TURNOVER_THRESHOLD
    result["low_turnover_high_concentration_flag"] = (
        (result["average_turnover"] < LOW_TURNOVER_THRESHOLD)
        & (
            result["average_effective_number_of_assets"]
            < HIGH_CONCENTRATION_EFFECTIVE_ASSETS
        )
    )
    result["structural_concentration_flag"] = (
        result["concentration_origin"].eq("structural")
        & result["high_concentration_flag"]
    )
    result["learned_concentration_flag"] = (
        result["concentration_origin"].eq("learned")
        & result["high_concentration_flag"]
    )
    mandate_score_ok = (
        result["mandate_aware_score"].isna()
        | (result["mandate_aware_score"] >= JUSTIFIED_MANDATE_AWARE_SCORE_THRESHOLD)
    )
    result["justified_concentration_candidate"] = (
        result["learned_concentration_flag"]
        & (result["sharpe"] >= JUSTIFIED_SHARPE_THRESHOLD)
        & (result["robust_score"] >= JUSTIFIED_ROBUST_SCORE_THRESHOLD)
        & (result["max_drawdown"] >= JUSTIFIED_MAX_DRAWDOWN_THRESHOLD)
        & (result["average_turnover"] <= HIGH_TURNOVER_THRESHOLD)
        & (
            result["abs_validation_test_sharpe_gap"].isna()
            | (
                result["abs_validation_test_sharpe_gap"]
                <= JUSTIFIED_VALIDATION_TEST_GAP_THRESHOLD
            )
        )
        & mandate_score_ok
    )
    result["suspicious_or_lazy_concentration_candidate"] = (
        result["learned_concentration_flag"]
        & result["extreme_concentration_flag"]
        & ~result["justified_concentration_candidate"]
    )
    result["lazy_concentration_candidate"] = result[
        "suspicious_or_lazy_concentration_candidate"
    ]
    result["concentration_classification"] = result.apply(
        _classify_concentration_status,
        axis=1,
    )
    result["concentration_reason"] = result.apply(_build_concentration_reason, axis=1)
    result["lazy_reason"] = result.apply(_build_lazy_reason, axis=1)
    result["justification_reason"] = result.apply(_build_justification_reason, axis=1)
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
    validation_return = validation.set_index("strategy")["mean_cumulative_return"]

    rows = []
    for _, row in test.iterrows():
        strategy = row["strategy"]
        validation_value = validation_sharpe.get(strategy, pd.NA)
        validation_return_value = validation_return.get(strategy, pd.NA)
        test_sharpe = row.get("mean_sharpe", pd.NA)
        test_return = row.get("mean_cumulative_return", pd.NA)
        validation_test_gap = _safe_subtract(test_sharpe, validation_value)
        validation_test_return_gap = _safe_subtract(
            test_return,
            validation_return_value,
        )
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
                "cash_above_10pct": row.get("cash_above_10_rate", pd.NA),
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
                "validation_cumulative_return": validation_return_value,
                "validation_test_return_gap": validation_test_return_gap,
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
    baseline_comparison_dir: str | None = None,
    primary_comparison_dir: str | None = None,
    baseline_mandate_dir: str | None = None,
    primary_mandate_dir: str | None = None,
) -> pd.DataFrame:
    """Compare candidate behavior between two protocol-pure revalidation folders."""
    baseline = load_reward_incentive_metrics(
        baseline_dir,
        comparison_dir=baseline_comparison_dir,
        mandate_dir=baseline_mandate_dir,
        experiment_label=baseline_label,
    )
    primary = load_reward_incentive_metrics(
        primary_dir,
        comparison_dir=primary_comparison_dir,
        mandate_dir=primary_mandate_dir,
        experiment_label=primary_label,
    )
    combined = pd.concat([baseline, primary], ignore_index=True)
    wide = combined.pivot_table(
        index="strategy",
        columns="experiment_label",
        values=[
            "sharpe",
            "cumulative_return",
            "annualized_return",
            "robust_score",
            "mandate_aware_score",
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
        "cumulative_return",
        "annualized_return",
        "robust_score",
        "mandate_aware_score",
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
    comparison_rows = (
        load_protocol_comparison_metrics(
            comparison_dir,
            mandate_dir=mandate_dir,
            experiment_label=Path(comparison_dir).name,
        )
        if comparison_dir is not None
        else pd.DataFrame()
    )
    combined = _combine_primary_and_comparison(primary, comparison_rows)
    audit = _select_audit_columns(combined)
    flags = _select_flag_columns(combined)
    summary = build_candidate_behavior_summary(combined)

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
            baseline_comparison_dir=baseline_comparison_dir,
            primary_comparison_dir=comparison_dir,
            baseline_mandate_dir=baseline_mandate_dir,
            primary_mandate_dir=mandate_dir,
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
    detail_columns = [
        "strategy",
        "strategy_type",
        "strategy_role",
        "concentration_origin",
        "concentration_classification",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "calmar",
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
        "validation_test_return_gap",
        "abs_validation_test_sharpe_gap",
        "structural_concentration_flag",
        "learned_concentration_flag",
        "suspicious_or_lazy_concentration_candidate",
        "justified_concentration_candidate",
        "concentration_reason",
        "lazy_reason",
        "justification_reason",
    ]
    detail = _with_columns(metrics, detail_columns).sort_values(
        ["robust_score", "sharpe"],
        ascending=[False, False],
        na_position="last",
    )
    summary_rows = build_role_summary_rows(metrics)
    return pd.concat([detail, summary_rows], ignore_index=True, sort=False)


def build_role_summary_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build explanatory summary rows for structural versus learned concentration."""
    structural = metrics.loc[metrics["structural_concentration_flag"]].copy()
    learned_extreme = metrics.loc[
        metrics["concentration_classification"].eq("learned_extreme_concentration")
    ].copy()
    learned = metrics.loc[metrics["learned_concentration_flag"]].copy()

    best_structural = _best_strategy_name(structural, "mandate_aware_score")
    best_learned = _best_strategy_name(learned, "mandate_aware_score")
    return pd.DataFrame(
        [
            {
                "strategy": "__summary__",
                "strategy_type": "summary",
                "strategy_role": "summary",
                "concentration_origin": "summary",
                "concentration_classification": "summary",
                "n_structural_concentration_benchmarks": int(len(structural)),
                "n_learned_extreme_concentration_models": int(len(learned_extreme)),
                "best_structural_concentrated_benchmark_by_mandate_score": (
                    best_structural
                ),
                "best_learned_concentrated_model_by_mandate_score": best_learned,
                "concentration_reason": (
                    "Structural benchmark concentration and learned TD3 "
                    "concentration are reported separately."
                ),
            }
        ]
    )


def load_protocol_comparison_metrics(
    comparison_dir: str,
    mandate_dir: str | None = None,
    experiment_label: str | None = None,
) -> pd.DataFrame:
    """Load all strategy rows from a protocol comparison summary."""
    comparison_path = Path(comparison_dir)
    summary_path = comparison_path / "protocol_comparison_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing protocol comparison summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    metrics = summary.rename(
        columns={
            "strategy_name": "strategy",
            "cash_above_10pct": "cash_above_10_rate",
            "average_transaction_cost": "mean_transaction_cost",
        }
    ).copy()
    metrics["experiment_label"] = experiment_label or comparison_path.name
    metrics["worst_max_drawdown"] = pd.NA
    metrics["validation_test_sharpe_gap"] = pd.NA
    metrics["abs_validation_test_sharpe_gap"] = pd.NA
    metrics["validation_test_return_gap"] = pd.NA
    metrics["validation_test_robust_score_gap"] = pd.NA
    metrics = _merge_mandate_scores(metrics, Path(mandate_dir)) if mandate_dir else metrics
    return compute_reward_incentive_flags(metrics)


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


def _combine_primary_and_comparison(
    primary_td3: pd.DataFrame,
    comparison_rows: pd.DataFrame,
) -> pd.DataFrame:
    if comparison_rows.empty:
        return primary_td3
    td3_names = set(primary_td3["strategy"].astype(str))
    non_td3 = comparison_rows.loc[
        ~comparison_rows["strategy"].astype(str).isin(td3_names)
    ].copy()
    columns = sorted(set(non_td3.columns).union(primary_td3.columns))
    records = non_td3.to_dict(orient="records") + primary_td3.to_dict(orient="records")
    return pd.DataFrame.from_records(records, columns=columns)


def _select_audit_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_label",
        "strategy",
        "strategy_type",
        "strategy_role",
        "concentration_origin",
        "concentration_classification",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
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
        "validation_test_return_gap",
        "validation_test_robust_score_gap",
        "abs_validation_test_sharpe_gap",
        "high_concentration_flag",
        "extreme_concentration_flag",
        "high_turnover_flag",
        "low_turnover_high_concentration_flag",
        "structural_concentration_flag",
        "learned_concentration_flag",
        "lazy_concentration_candidate",
        "suspicious_or_lazy_concentration_candidate",
        "justified_concentration_candidate",
        "concentration_reason",
        "lazy_reason",
        "justification_reason",
    ]
    return _with_columns(metrics, columns)


def _select_flag_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_label",
        "strategy",
        "strategy_type",
        "strategy_role",
        "concentration_origin",
        "concentration_classification",
        "average_turnover",
        "average_effective_number_of_assets",
        "average_max_weight",
        "cumulative_return",
        "annualized_return",
        "sharpe",
        "robust_score",
        "mandate_aware_score",
        "max_drawdown",
        "abs_validation_test_sharpe_gap",
        "high_concentration_flag",
        "extreme_concentration_flag",
        "high_turnover_flag",
        "low_turnover_high_concentration_flag",
        "structural_concentration_flag",
        "learned_concentration_flag",
        "lazy_concentration_candidate",
        "suspicious_or_lazy_concentration_candidate",
        "justified_concentration_candidate",
        "concentration_reason",
        "lazy_reason",
        "justification_reason",
    ]
    return _with_columns(metrics, columns)


def _classify_strategy_role(row: pd.Series) -> str:
    name = str(row.get("strategy", row.get("strategy_name", "")))
    strategy_type = str(row.get("strategy_type", "")).lower()
    if name.startswith("BuyHold_"):
        return "single_asset_benchmark"
    if name in {"momentum_winner_12p", "risk_adjusted_momentum_winner_12p_12p"}:
        return "winner_take_all_benchmark"
    if name in {"Equal_Weight", "Equal_Weight_Risky", "60_40_SPY_TLT"}:
        return "static_allocation_benchmark"
    if (
        name in {"trend_spy_cash_12p", "defensive_risk_off_12p"}
        or name.startswith("rolling_risk_parity")
        or name.startswith("rolling_markowitz")
    ):
        return "dynamic_allocation_benchmark"
    if strategy_type in {"td3", "drl"} or name.startswith("V"):
        return "learned_td3_allocator"
    return "unknown"


def _classify_concentration_origin(strategy_role: str) -> str:
    if strategy_role in {"single_asset_benchmark", "winner_take_all_benchmark"}:
        return "structural"
    if strategy_role == "static_allocation_benchmark":
        return "partly_structural"
    if strategy_role == "dynamic_allocation_benchmark":
        return "rule_based_dynamic"
    if strategy_role == "learned_td3_allocator":
        return "learned"
    return "unknown"


def _classify_concentration_status(row: pd.Series) -> str:
    if bool(row.get("structural_concentration_flag", False)):
        return "structural_concentration_benchmark"
    if row.get("concentration_origin") == "learned":
        if bool(row.get("extreme_concentration_flag", False)):
            return "learned_extreme_concentration"
        if bool(row.get("high_concentration_flag", False)):
            return "learned_high_concentration"
    return "not_concentrated"


def _build_concentration_reason(row: pd.Series) -> str:
    classification = row.get("concentration_classification")
    effective_assets = row.get("average_effective_number_of_assets")
    if classification == "structural_concentration_benchmark":
        if row.get("strategy_role") == "single_asset_benchmark":
            return "Structural single-asset benchmark; concentration expected by design."
        return "Structural winner-take-all benchmark; concentration expected by design."
    if classification == "learned_extreme_concentration":
        return (
            "Learned TD3 allocator with effective assets below "
            f"{EXTREME_CONCENTRATION_EFFECTIVE_ASSETS}."
        )
    if classification == "learned_high_concentration":
        return (
            "Learned TD3 allocator with effective assets between "
            f"{EXTREME_CONCENTRATION_EFFECTIVE_ASSETS} and "
            f"{HIGH_CONCENTRATION_EFFECTIVE_ASSETS}."
        )
    if pd.notna(effective_assets):
        return "Concentration is not high under the audit threshold."
    return "Concentration could not be classified because effective assets are missing."


def _build_lazy_reason(row: pd.Series) -> str:
    if row.get("concentration_origin") == "structural":
        return ""
    if not bool(row.get("suspicious_or_lazy_concentration_candidate", False)):
        return ""
    reasons = ["Extreme learned concentration."]
    if bool(row.get("high_turnover_flag", False)):
        reasons.append("Turnover is above threshold.")
    if pd.to_numeric(pd.Series([row.get("sharpe")]), errors="coerce").iloc[0] < JUSTIFIED_SHARPE_THRESHOLD:
        reasons.append("Sharpe is below justified-concentration threshold.")
    if pd.to_numeric(pd.Series([row.get("robust_score")]), errors="coerce").iloc[0] < JUSTIFIED_ROBUST_SCORE_THRESHOLD:
        reasons.append("Robust score is below justified-concentration threshold.")
    mandate = pd.to_numeric(pd.Series([row.get("mandate_aware_score")]), errors="coerce").iloc[0]
    if pd.notna(mandate) and mandate < JUSTIFIED_MANDATE_AWARE_SCORE_THRESHOLD:
        reasons.append("Mandate-aware score is below threshold.")
    drawdown = pd.to_numeric(pd.Series([row.get("max_drawdown")]), errors="coerce").iloc[0]
    if pd.notna(drawdown) and drawdown < JUSTIFIED_MAX_DRAWDOWN_THRESHOLD:
        reasons.append("Drawdown is worse than justified-concentration threshold.")
    return " ".join(reasons)


def _build_justification_reason(row: pd.Series) -> str:
    if row.get("concentration_origin") == "structural":
        return "Not evaluated as learned concentration."
    if bool(row.get("justified_concentration_candidate", False)):
        return "Meets learned concentration justification thresholds."
    if row.get("concentration_origin") == "learned" and bool(
        row.get("high_concentration_flag", False)
    ):
        return "Fails one or more justified concentration thresholds."
    return "Not a high learned concentration case."


def _best_strategy_name(df: pd.DataFrame, score_column: str) -> str | pd._libs.missing.NAType:
    if df.empty or score_column not in df.columns:
        return pd.NA
    ranked = df.sort_values(score_column, ascending=False, na_position="last")
    if ranked.empty:
        return pd.NA
    return ranked.iloc[0]["strategy"]


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
