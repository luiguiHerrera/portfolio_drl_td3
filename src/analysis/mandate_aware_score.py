"""Mandate-aware scoring layer for protocol comparison outputs.

This module is reporting-only. It does not replace production robust_score;
it applies drawdown mandate eligibility buckets on top of existing protocol
comparison scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INPUT_FILE = "protocol_comparison_summary.csv"
DEFAULT_OUTPUT_DIR = "outputs/tables/mandate_aware_score"
BUCKET_ORDER = ["clean_mandate", "eligible_yellow", "eligible_red", "not_eligible"]


def assign_drawdown_bucket(max_drawdown: float) -> str:
    """Assign a drawdown mandate bucket from a negative max drawdown value."""
    value = float(max_drawdown)
    if value >= -0.20:
        return "clean_mandate"
    if value >= -0.25:
        return "eligible_yellow"
    if value >= -0.30:
        return "eligible_red"
    return "not_eligible"


def get_drawdown_multiplier(max_drawdown: float) -> float:
    """Return the recovery-asymmetry drawdown multiplier.

    The multiplier is continuous across eligible buckets. Bucket eligibility is
    still handled separately, and not-eligible strategies receive a zero
    mandate-aware score even when this formula is positive.
    """
    recovery_required = calculate_recovery_required(max_drawdown)
    return max(0.0, 1.0 - recovery_required)


def calculate_recovery_required(max_drawdown: float) -> float:
    """Return the gain required to recover from a negative max drawdown."""
    abs_dd = abs(float(max_drawdown))
    if abs_dd >= 1.0:
        return float("inf")
    return abs_dd / (1.0 - abs_dd)


def add_mandate_aware_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add mandate buckets, multipliers, scores, and ranks to a strategy table."""
    _validate_input_frame(df)
    result = df.copy()
    result["robust_score"] = pd.to_numeric(result["robust_score"], errors="coerce")
    result["max_drawdown"] = pd.to_numeric(result["max_drawdown"], errors="coerce")
    result["mandate_bucket"] = result["max_drawdown"].apply(assign_drawdown_bucket)
    result["recovery_required"] = result["max_drawdown"].apply(
        calculate_recovery_required
    )
    result["drawdown_multiplier"] = result["max_drawdown"].apply(
        get_drawdown_multiplier
    )
    result["mandate_aware_score"] = result["robust_score"] * result[
        "drawdown_multiplier"
    ]
    result.loc[result["mandate_bucket"] == "not_eligible", "mandate_aware_score"] = 0.0
    result["performance_robust_rank"] = (
        result["robust_score"].rank(ascending=False, method="min").astype(int)
    )
    result["mandate_aware_rank"] = (
        result["mandate_aware_score"].rank(ascending=False, method="min").astype(int)
    )
    result["mandate_bucket_rank"] = (
        result.groupby("mandate_bucket", group_keys=False)["robust_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    result["mandate_bucket"] = pd.Categorical(
        result["mandate_bucket"],
        categories=BUCKET_ORDER,
        ordered=True,
    )
    return result.sort_values(
        ["mandate_aware_rank", "performance_robust_rank"],
        ascending=[True, True],
    ).reset_index(drop=True)


def write_mandate_aware_outputs(
    input_dir: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Read protocol comparison summary and write mandate-aware output tables."""
    input_path = Path(input_dir) / INPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Missing protocol comparison summary: {input_path}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(input_path)
    ranking = add_mandate_aware_scores(source)
    ranking = _select_ranking_columns(ranking)
    bucket_summary = build_mandate_bucket_summary(ranking)
    flags = build_mandate_eligibility_flags(ranking)

    ranking_path = output_path / "mandate_aware_ranking.csv"
    summary_path = output_path / "mandate_bucket_summary.csv"
    flags_path = output_path / "mandate_eligibility_flags.csv"
    ranking.to_csv(ranking_path, index=False)
    bucket_summary.to_csv(summary_path, index=False)
    flags.to_csv(flags_path, index=False)

    return {
        "output_dir": str(output_path),
        "ranking": ranking,
        "bucket_summary": bucket_summary,
        "eligibility_flags": flags,
        "paths": {
            "mandate_aware_ranking": str(ranking_path),
            "mandate_bucket_summary": str(summary_path),
            "mandate_eligibility_flags": str(flags_path),
        },
    }


def build_mandate_bucket_summary(ranking: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategies by mandate bucket."""
    rows = []
    for bucket in BUCKET_ORDER:
        group = ranking.loc[ranking["mandate_bucket"].astype(str) == bucket].copy()
        if group.empty:
            rows.append(
                {
                    "mandate_bucket": bucket,
                    "n_strategies": 0,
                    "mean_robust_score": pd.NA,
                    "mean_mandate_aware_score": pd.NA,
                    "mean_recovery_required": pd.NA,
                    "mean_drawdown_multiplier": pd.NA,
                    "best_strategy_by_robust_score": pd.NA,
                    "best_strategy_by_mandate_aware_score": pd.NA,
                    "min_drawdown": pd.NA,
                    "max_drawdown": pd.NA,
                }
            )
            continue

        best_robust = group.sort_values(
            "robust_score",
            ascending=False,
            na_position="last",
        ).iloc[0]
        best_mandate = group.sort_values(
            "mandate_aware_score",
            ascending=False,
            na_position="last",
        ).iloc[0]
        rows.append(
            {
                "mandate_bucket": bucket,
                "n_strategies": int(len(group)),
                "mean_robust_score": float(group["robust_score"].mean()),
                "mean_mandate_aware_score": float(group["mandate_aware_score"].mean()),
                "mean_recovery_required": float(group["recovery_required"].mean()),
                "mean_drawdown_multiplier": float(
                    group["drawdown_multiplier"].mean()
                ),
                "best_strategy_by_robust_score": best_robust["strategy_name"],
                "best_strategy_by_mandate_aware_score": best_mandate["strategy_name"],
                "min_drawdown": float(group["max_drawdown"].min()),
                "max_drawdown": float(group["max_drawdown"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_mandate_eligibility_flags(ranking: pd.DataFrame) -> pd.DataFrame:
    """Build explicit mandate eligibility flags for each strategy."""
    flags = ranking[
        [
            "strategy_name",
            "strategy_type",
            "max_drawdown",
            "mandate_bucket",
            "recovery_required",
            "drawdown_multiplier",
        ]
    ].copy()
    bucket = flags["mandate_bucket"].astype(str)
    flags["is_clean_mandate"] = bucket == "clean_mandate"
    flags["is_eligible"] = bucket.isin(
        ["clean_mandate", "eligible_yellow", "eligible_red"]
    )
    flags["is_not_eligible"] = bucket == "not_eligible"
    return flags


def _select_ranking_columns(ranking: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "strategy_name",
        "strategy_type",
        "robust_score",
        "recovery_required",
        "drawdown_multiplier",
        "mandate_aware_score",
        "performance_robust_rank",
        "mandate_aware_rank",
        "mandate_bucket",
        "mandate_bucket_rank",
        "max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "average_turnover",
        "average_effective_number_of_assets",
        "dsr_method",
    ]
    for column in columns:
        if column not in ranking.columns:
            ranking[column] = pd.NA
    return ranking.loc[:, columns]


def _validate_input_frame(df: pd.DataFrame) -> None:
    required = {"strategy_name", "strategy_type", "robust_score", "max_drawdown"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input table is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Input table must not be empty.")
    max_drawdown = pd.to_numeric(df["max_drawdown"], errors="coerce")
    if max_drawdown.isna().any():
        raise ValueError("max_drawdown must be numeric for all rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mandate-aware score tables.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = write_mandate_aware_outputs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print(f"Output folder: {report['output_dir']}")
    print("\nTop mandate-aware ranking:")
    print(report["ranking"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
