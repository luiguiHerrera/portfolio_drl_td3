"""Compare capped and uncapped TD3 candidates against protocol benchmarks.

This is a reporting layer. It does not train TD3, alter the reward, or change
the environment. It combines existing max-weight cap experiment summaries with
freshly regenerated benchmark protocol rows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.mandate_aware_score import add_mandate_aware_scores
from src.experiments.run_protocol_benchmark_comparison import (
    run_protocol_benchmark_comparison,
)
from src.experiments.run_protocol_td3_comparison import (
    BENCHMARK_ROBUST_SCORE_METHOD,
    DEFAULT_DSR_POLICY,
    TIMING_CONVENTION,
    TURNOVER_CONVENTION,
    _normalize_benchmark_metrics,
)


DEFAULT_OUTPUT_DIR = "outputs/tables/capped_td3_protocol_comparison_60ep_10seeds_cap060"
DEFAULT_RETURNS_PATH = "data/processed/returns_weekly_latest.csv"
DEFAULT_CAP_VALUE = 0.60

BASE_CANDIDATE_LABELS = {
    "V2_reference_full": "V2",
    "V5_no_volatility_block": "V5",
    "V6_financial_state": "V6",
}

FEATURE_VERSION_BY_BASE = {
    "V2_reference_full": "v2",
    "V5_no_volatility_block": "v5",
    "V6_financial_state": "v6",
}

SUMMARY_COLUMNS = [
    "strategy_name",
    "strategy_type",
    "base_candidate",
    "candidate_name",
    "feature_version",
    "max_weight_cap",
    "constraint_status",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "robust_score",
    "mandate_aware_score",
    "performance_robust_rank",
    "mandate_aware_rank",
    "clean_mandate_rank",
    "eligible_rank",
    "benchmark_relative_rank",
    "mandate_bucket",
    "recovery_required",
    "drawdown_multiplier",
    "max_drawdown",
    "worst_max_drawdown",
    "average_turnover",
    "total_transaction_cost",
    "average_transaction_cost",
    "mean_transaction_cost",
    "average_effective_number_of_assets",
    "average_max_weight",
    "mean_cash_weight",
    "cash_above_10pct",
    "cash_above_10_rate",
    "pooled_dsr_n10",
    "pooled_dsr_n25",
    "pooled_dsr_n50",
    "mean_run_dsr_n25",
    "median_run_dsr_n25",
    "date_averaged_dsr_n25",
    "dsr_method",
    "concentration_classification",
    "suspicious_or_lazy_concentration_candidate",
    "justified_concentration_candidate",
    "source_path",
]

PAIRWISE_DELTA_COLUMNS = [
    "annualized_return",
    "sharpe",
    "robust_score",
    "mandate_aware_score",
    "max_drawdown",
    "average_turnover",
    "average_effective_number_of_assets",
    "average_max_weight",
]


def run_capped_td3_protocol_comparison(
    returns_path: str = DEFAULT_RETURNS_PATH,
    v2_path: str | None = None,
    v5_path: str | None = None,
    v6_path: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cap_value: float = DEFAULT_CAP_VALUE,
    transaction_cost: float = 0.001,
    initial_value: float = 100000.0,
    date_column: str = "date",
) -> dict[str, Any]:
    """Build the capped-vs-uncapped TD3 and benchmark comparison report."""
    output_path = Path(output_dir)
    benchmark_output_dir = output_path / "benchmarks"
    output_path.mkdir(parents=True, exist_ok=True)

    input_paths = _collect_input_paths(v2_path=v2_path, v5_path=v5_path, v6_path=v6_path)
    td3_rows = load_capped_td3_rows(input_paths)
    benchmark_result = run_protocol_benchmark_comparison(
        returns_path=returns_path,
        output_dir=str(benchmark_output_dir),
        transaction_cost=transaction_cost,
        initial_value=initial_value,
        date_column=date_column,
    )
    benchmark_rows, benchmark_info = _normalize_benchmark_metrics(
        benchmark_result["metrics_table"],
        benchmark_result["evaluations"],
    )
    benchmark_rows = normalize_benchmark_rows(benchmark_rows)

    combined = build_combined_comparison_table(td3_rows, benchmark_rows)
    pairwise_deltas = build_pairwise_cap_deltas(combined)
    mandate_ranking = combined.sort_values(
        ["mandate_aware_score", "robust_score"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    performance_ranking = combined.sort_values(
        ["robust_score", "sharpe", "annualized_return"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    paths = write_capped_protocol_outputs(
        output_path=output_path,
        summary=combined,
        pairwise_deltas=pairwise_deltas,
        mandate_ranking=mandate_ranking,
        performance_ranking=performance_ranking,
        metadata=build_metadata(
            returns_path=returns_path,
            output_dir=output_dir,
            input_paths=input_paths,
            cap_value=cap_value,
            transaction_cost=transaction_cost,
            benchmark_output_dir=str(benchmark_output_dir),
            benchmark_info=benchmark_info,
        ),
    )

    return {
        "summary": combined,
        "pairwise_deltas": pairwise_deltas,
        "mandate_ranking": mandate_ranking,
        "performance_ranking": performance_ranking,
        "benchmark_result": benchmark_result,
        "paths": paths,
    }


def _collect_input_paths(
    v2_path: str | None,
    v5_path: str | None,
    v6_path: str | None,
) -> dict[str, str]:
    paths = {
        "V2_reference_full": v2_path,
        "V5_no_volatility_block": v5_path,
        "V6_financial_state": v6_path,
    }
    missing = [name for name, path in paths.items() if not path]
    if missing:
        raise ValueError(f"Missing capped experiment paths for: {missing}")
    return {name: str(path) for name, path in paths.items() if path is not None}


def load_capped_td3_rows(input_paths: dict[str, str]) -> pd.DataFrame:
    """Load and normalize TD3 capped/uncapped rows from cap experiment folders."""
    frames = []
    for expected_base, path_value in input_paths.items():
        path = Path(path_value)
        ranking_path = path / "max_weight_cap_rankings.csv"
        if not ranking_path.exists():
            raise FileNotFoundError(f"Missing max_weight_cap_rankings.csv: {path}")
        frame = pd.read_csv(ranking_path)
        if "split" in frame.columns:
            frame = frame[frame["split"].astype(str) == "test"].copy()
        if frame.empty:
            raise ValueError(f"No test rows found in {ranking_path}")
        frame["source_path"] = str(path)
        frame["base_candidate"] = frame.get("base_candidate", expected_base)
        frame["base_candidate"] = frame["base_candidate"].fillna(expected_base)
        robust_path = path / "robust_score_ranking.csv"
        if robust_path.exists():
            frame = _merge_cap_robust_fields(frame, pd.read_csv(robust_path))
        frames.append(frame)

    rows = pd.concat(frames, ignore_index=True)
    return normalize_capped_td3_rows(rows)


def _merge_cap_robust_fields(
    frame: pd.DataFrame,
    robust: pd.DataFrame,
) -> pd.DataFrame:
    if "strategy" not in robust.columns:
        return frame
    keep = [
        column
        for column in [
            "strategy",
            "pooled_dsr_n10",
            "pooled_dsr_n25",
            "pooled_dsr_n50",
            "mean_run_dsr_n25",
            "median_run_dsr_n25",
            "date_averaged_dsr_n25",
            "dsr_method",
        ]
        if column in robust.columns
    ]
    robust_subset = robust.loc[:, keep].rename(columns={"strategy": "candidate_name"})
    return frame.merge(robust_subset, on="candidate_name", how="left")


def normalize_capped_td3_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize max-weight cap rows into the combined protocol schema."""
    result = rows.copy()
    result["max_weight_cap"] = pd.to_numeric(result["max_weight_cap"], errors="coerce")
    result["constraint_status"] = result["max_weight_cap"].apply(_constraint_status)
    result["strategy_type"] = np.where(
        result["max_weight_cap"].isna(),
        "td3_uncapped",
        "td3_capped",
    )
    result["strategy_name"] = result.apply(_td3_strategy_name, axis=1)
    result["feature_version"] = result["base_candidate"].map(FEATURE_VERSION_BY_BASE).fillna(
        "unknown"
    )
    result["candidate_name"] = result.get("candidate_name", result["strategy_name"])
    result["cash_above_10pct"] = result.get("cash_above_10_rate", pd.NA)
    result["average_transaction_cost"] = result.get("mean_transaction_cost", pd.NA)
    result["total_transaction_cost"] = pd.NA
    for column in SUMMARY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, SUMMARY_COLUMNS]


def normalize_benchmark_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize benchmark rows into the capped comparison schema."""
    result = rows.copy()
    result["strategy_type"] = "benchmark"
    result["base_candidate"] = pd.NA
    result["candidate_name"] = pd.NA
    result["feature_version"] = pd.NA
    result["max_weight_cap"] = pd.NA
    result["constraint_status"] = "benchmark"
    result["source_path"] = "protocol_benchmark_runner"
    result["cash_above_10_rate"] = result.get("cash_above_10pct", pd.NA)
    result["mean_transaction_cost"] = pd.NA
    result["average_transaction_cost"] = pd.NA
    result["worst_max_drawdown"] = result.get("max_drawdown", pd.NA)
    for column in SUMMARY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, SUMMARY_COLUMNS]


def build_combined_comparison_table(
    td3_rows: pd.DataFrame,
    benchmark_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Combine TD3 and benchmark rows, then add mandate and protocol ranks."""
    compact_frames = [
        frame.dropna(axis=1, how="all")
        for frame in [td3_rows, benchmark_rows]
        if not frame.empty
    ]
    combined = pd.concat(compact_frames, ignore_index=True)
    combined = add_mandate_aware_scores(combined)
    combined = _add_protocol_ranks(combined)
    for column in SUMMARY_COLUMNS:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined.loc[:, SUMMARY_COLUMNS]


def _add_protocol_ranks(combined: pd.DataFrame) -> pd.DataFrame:
    result = combined.copy()
    result["benchmark_relative_rank"] = pd.to_numeric(
        result["robust_score"],
        errors="coerce",
    ).rank(ascending=False, method="min")
    result["clean_mandate_rank"] = pd.NA
    clean_mask = result["mandate_bucket"].astype(str) == "clean_mandate"
    if clean_mask.any():
        result.loc[clean_mask, "clean_mandate_rank"] = pd.to_numeric(
            result.loc[clean_mask, "mandate_aware_score"],
            errors="coerce",
        ).rank(ascending=False, method="min")
    result["eligible_rank"] = pd.NA
    eligible_mask = result["mandate_bucket"].astype(str) != "not_eligible"
    if eligible_mask.any():
        result.loc[eligible_mask, "eligible_rank"] = pd.to_numeric(
            result.loc[eligible_mask, "mandate_aware_score"],
            errors="coerce",
        ).rank(ascending=False, method="min")
    return result


def build_pairwise_cap_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    """Build cap-vs-uncapped deltas for each TD3 base candidate."""
    td3 = comparison[comparison["strategy_type"].isin(["td3_uncapped", "td3_capped"])].copy()
    rows = []
    for base_candidate, group in td3.groupby("base_candidate", dropna=True):
        uncapped = group[group["strategy_type"] == "td3_uncapped"]
        capped = group[group["strategy_type"] == "td3_capped"]
        if uncapped.empty or capped.empty:
            continue
        baseline = uncapped.iloc[0]
        for _, cap_row in capped.iterrows():
            row: dict[str, Any] = {
                "base_candidate": base_candidate,
                "uncapped_strategy_name": baseline["strategy_name"],
                "capped_strategy_name": cap_row["strategy_name"],
                "max_weight_cap": cap_row["max_weight_cap"],
            }
            for metric in PAIRWISE_DELTA_COLUMNS:
                row[f"uncapped_{metric}"] = baseline.get(metric)
                row[f"capped_{metric}"] = cap_row.get(metric)
                row[f"delta_{metric}"] = _numeric_value(cap_row.get(metric)) - _numeric_value(
                    baseline.get(metric)
                )
            row["summary_decision"] = label_cap_pairwise_decision(row)
            rows.append(row)
    return pd.DataFrame(rows)


def label_cap_pairwise_decision(row: pd.Series | dict[str, Any]) -> str:
    """Assign a conservative cap-vs-uncapped decision label."""
    get = row.get
    effective_delta = _numeric_value(get("delta_average_effective_number_of_assets"))
    max_weight_delta = _numeric_value(get("delta_average_max_weight"))
    robust_delta = _numeric_value(get("delta_robust_score"))
    mandate_delta = _numeric_value(get("delta_mandate_aware_score"))
    drawdown_delta = _numeric_value(get("delta_max_drawdown"))
    return_delta = _numeric_value(get("delta_annualized_return"))

    if effective_delta <= 0.20:
        return "cap_inconclusive"
    if robust_delta >= 0.0 and mandate_delta >= 0.0 and drawdown_delta >= -0.02:
        if return_delta >= 0.0 and max_weight_delta < 0.0:
            return "cap_dominates_uncapped"
        return "cap_improves_mandate_but_hurts_return"
    if mandate_delta > 0.0 and return_delta < 0.0:
        return "cap_improves_mandate_but_hurts_return"
    if robust_delta < -0.05 and mandate_delta < -0.05:
        return "uncapped_preferred"
    return "cap_inconclusive"


def write_capped_protocol_outputs(
    output_path: Path,
    summary: pd.DataFrame,
    pairwise_deltas: pd.DataFrame,
    mandate_ranking: pd.DataFrame,
    performance_ranking: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write all capped protocol comparison outputs."""
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_path / "capped_td3_vs_benchmarks_summary.csv",
        "pairwise_deltas": output_path / "capped_td3_pairwise_deltas.csv",
        "mandate_ranking": output_path / "capped_td3_mandate_ranking.csv",
        "performance_ranking": output_path / "capped_td3_performance_ranking.csv",
        "metadata": output_path / "capped_td3_protocol_metadata.json",
    }
    summary.to_csv(paths["summary"], index=False)
    pairwise_deltas.to_csv(paths["pairwise_deltas"], index=False)
    mandate_ranking.to_csv(paths["mandate_ranking"], index=False)
    performance_ranking.to_csv(paths["performance_ranking"], index=False)
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def build_metadata(
    returns_path: str,
    output_dir: str,
    input_paths: dict[str, str],
    cap_value: float,
    transaction_cost: float,
    benchmark_output_dir: str,
    benchmark_info: dict[str, Any],
) -> dict[str, Any]:
    """Build reproducibility metadata for the reporting layer."""
    metadata = {
        "runner": "src.experiments.run_capped_td3_protocol_comparison",
        "git_commit_hash": _git_commit_hash(),
        "returns_path": returns_path,
        "output_dir": output_dir,
        "input_experiment_folders": input_paths,
        "cap_value": cap_value,
        "transaction_cost": transaction_cost,
        "benchmark_output_dir": benchmark_output_dir,
        "timing_convention": TIMING_CONVENTION,
        "turnover_convention": TURNOVER_CONVENTION,
        "DSR_method_policy": DEFAULT_DSR_POLICY,
        "benchmark_robust_score_method": BENCHMARK_ROBUST_SCORE_METHOD,
        "reporting_only_note": (
            "This runner combines existing capped TD3 experiment outputs with "
            "fresh protocol benchmarks. It does not train models or change "
            "reward, environment, robust_score, or mandate_aware_score logic."
        ),
    }
    metadata.update(benchmark_info)
    return metadata


def _td3_strategy_name(row: pd.Series) -> str:
    base_label = BASE_CANDIDATE_LABELS.get(
        str(row.get("base_candidate")),
        str(row.get("base_candidate", "TD3")),
    )
    return f"{base_label}_{_constraint_status(row.get('max_weight_cap'))}"


def _constraint_status(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "uncapped"
    return f"cap_{float(numeric):.2f}"


def _numeric_value(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return np.nan
    return float(numeric)


def _git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare capped and uncapped TD3 candidates against benchmarks.",
    )
    parser.add_argument("--returns-path", default=DEFAULT_RETURNS_PATH)
    parser.add_argument("--v2-path", required=True)
    parser.add_argument("--v5-path", required=True)
    parser.add_argument("--v6-path", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cap-value", type=float, default=DEFAULT_CAP_VALUE)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--initial-value", type=float, default=100000.0)
    parser.add_argument("--date-column", default="date")
    args = parser.parse_args()

    result = run_capped_td3_protocol_comparison(
        returns_path=args.returns_path,
        v2_path=args.v2_path,
        v5_path=args.v5_path,
        v6_path=args.v6_path,
        output_dir=args.output_dir,
        cap_value=args.cap_value,
        transaction_cost=args.transaction_cost,
        initial_value=args.initial_value,
        date_column=args.date_column,
    )
    print("Top performance ranking:")
    print(result["performance_ranking"].head(15).to_string(index=False))
    print("\nTop mandate-aware ranking:")
    print(result["mandate_ranking"].head(15).to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
