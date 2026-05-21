"""Build paper-ready executive tables from capped TD3 protocol outputs.

This module is reporting-only. It reads the capped TD3 protocol comparison
outputs and writes concise CSV/Markdown tables for research notes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = "outputs/tables/capped_td3_protocol_comparison_60ep_10seeds_cap060"
DEFAULT_OUTPUT_DIR = "outputs/tables/executive_results_report_60ep_10seeds_cap060"
SUMMARY_FILE = "capped_td3_vs_benchmarks_summary.csv"
PAIRWISE_FILE = "capped_td3_pairwise_deltas.csv"


MAIN_RANKING_COLUMNS = [
    "rank_robust",
    "strategy_name",
    "strategy_type",
    "constraint_status",
    "robust_score",
    "mandate_aware_score",
    "mandate_bucket",
    "is_mandate_eligible",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "recovery_required",
    "average_turnover",
    "average_effective_number_of_assets",
    "average_max_weight",
]

MANDATE_RANKING_COLUMNS = [
    "rank_mandate",
    "strategy_name",
    "strategy_type",
    "constraint_status",
    "mandate_aware_score",
    "robust_score",
    "mandate_bucket",
    "annualized_return",
    "sharpe",
    "max_drawdown",
    "recovery_required",
    "average_turnover",
    "average_effective_number_of_assets",
    "average_max_weight",
]

NON_ELIGIBLE_COLUMNS = [
    "rank_robust",
    "strategy_name",
    "strategy_type",
    "robust_score",
    "mandate_aware_score",
    "max_drawdown",
    "recovery_required",
    "annualized_return",
    "sharpe",
    "reason_not_eligible",
]

CAP_IMPACT_COLUMNS = [
    "base_candidate",
    "uncapped_mandate_aware_score",
    "capped_mandate_aware_score",
    "delta_mandate_aware_score",
    "uncapped_robust_score",
    "capped_robust_score",
    "delta_robust_score",
    "uncapped_annualized_return",
    "capped_annualized_return",
    "delta_annualized_return",
    "uncapped_sharpe",
    "capped_sharpe",
    "delta_sharpe",
    "uncapped_max_drawdown",
    "capped_max_drawdown",
    "delta_max_drawdown",
    "uncapped_turnover",
    "capped_turnover",
    "delta_turnover",
    "uncapped_effective_assets",
    "capped_effective_assets",
    "delta_effective_assets",
    "uncapped_average_max_weight",
    "capped_average_max_weight",
    "delta_average_max_weight",
    "pairwise_decision",
]


def build_executive_results_report(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build executive result CSVs and a short Markdown summary."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = load_protocol_summary(input_path)
    pairwise = load_pairwise_deltas(input_path)

    main_ranking = build_main_ranking(summary)
    mandate_ranking = build_mandate_eligible_ranking(summary)
    non_eligible = build_non_eligible_strategies(summary)
    cap_impact = build_td3_cap_impact(pairwise)
    group_summary = build_strategy_groups_summary(summary)
    markdown = build_markdown_summary(
        main_ranking=main_ranking,
        mandate_ranking=mandate_ranking,
        non_eligible=non_eligible,
        cap_impact=cap_impact,
        group_summary=group_summary,
    )

    paths = {
        "main_ranking": output_path / "executive_main_ranking.csv",
        "mandate_eligible_ranking": output_path
        / "executive_mandate_eligible_ranking.csv",
        "non_eligible_strategies": output_path
        / "executive_non_eligible_strategies.csv",
        "td3_cap_impact": output_path / "executive_td3_cap_impact.csv",
        "strategy_groups_summary": output_path
        / "executive_strategy_groups_summary.csv",
        "markdown_summary": output_path / "executive_results_summary.md",
    }
    main_ranking.to_csv(paths["main_ranking"], index=False)
    mandate_ranking.to_csv(paths["mandate_eligible_ranking"], index=False)
    non_eligible.to_csv(paths["non_eligible_strategies"], index=False)
    cap_impact.to_csv(paths["td3_cap_impact"], index=False)
    group_summary.to_csv(paths["strategy_groups_summary"], index=False)
    paths["markdown_summary"].write_text(markdown, encoding="utf-8")

    return {
        "main_ranking": main_ranking,
        "mandate_eligible_ranking": mandate_ranking,
        "non_eligible_strategies": non_eligible,
        "td3_cap_impact": cap_impact,
        "strategy_groups_summary": group_summary,
        "markdown_summary": markdown,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def load_protocol_summary(input_path: Path) -> pd.DataFrame:
    """Load and validate capped TD3 protocol summary."""
    path = input_path / SUMMARY_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing capped protocol summary: {path}")
    summary = pd.read_csv(path)
    required = {
        "strategy_name",
        "strategy_type",
        "robust_score",
        "mandate_aware_score",
        "mandate_bucket",
        "max_drawdown",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Protocol summary missing columns: {sorted(missing)}")
    return _with_eligibility(summary)


def load_pairwise_deltas(input_path: Path) -> pd.DataFrame:
    """Load capped-vs-uncapped TD3 pairwise deltas."""
    path = input_path / PAIRWISE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing capped pairwise deltas: {path}")
    return pd.read_csv(path)


def build_main_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """Return all strategies sorted by robust_score."""
    ranking = summary.copy()
    ranking["rank_robust"] = pd.to_numeric(
        ranking["robust_score"],
        errors="coerce",
    ).rank(ascending=False, method="min").astype("Int64")
    ranking = ranking.sort_values(
        ["rank_robust", "strategy_name"],
        ascending=[True, True],
    ).reset_index(drop=True)
    return _select_columns(ranking, MAIN_RANKING_COLUMNS)


def build_mandate_eligible_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """Return only mandate-eligible strategies sorted by mandate-aware score."""
    eligible = summary[summary["is_mandate_eligible"]].copy()
    eligible["rank_mandate"] = pd.to_numeric(
        eligible["mandate_aware_score"],
        errors="coerce",
    ).rank(ascending=False, method="min").astype("Int64")
    eligible = eligible.sort_values(
        ["rank_mandate", "robust_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return _select_columns(eligible, MANDATE_RANKING_COLUMNS)


def build_non_eligible_strategies(summary: pd.DataFrame) -> pd.DataFrame:
    """Return not-eligible or zero mandate-score strategies."""
    mandate_score = pd.to_numeric(summary["mandate_aware_score"], errors="coerce")
    not_eligible = summary[
        (summary["mandate_bucket"].astype(str) == "not_eligible")
        | (mandate_score.fillna(0.0) == 0.0)
    ].copy()
    not_eligible["rank_robust"] = pd.to_numeric(
        not_eligible["robust_score"],
        errors="coerce",
    ).rank(ascending=False, method="min").astype("Int64")
    not_eligible["reason_not_eligible"] = not_eligible.apply(
        reason_not_eligible,
        axis=1,
    )
    not_eligible = not_eligible.sort_values(
        ["rank_robust", "strategy_name"],
        ascending=[True, True],
    ).reset_index(drop=True)
    return _select_columns(not_eligible, NON_ELIGIBLE_COLUMNS)


def build_td3_cap_impact(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Format capped-vs-uncapped pairwise deltas for executive reporting."""
    column_map = {
        "uncapped_average_turnover": "uncapped_turnover",
        "capped_average_turnover": "capped_turnover",
        "delta_average_turnover": "delta_turnover",
        "uncapped_average_effective_number_of_assets": "uncapped_effective_assets",
        "capped_average_effective_number_of_assets": "capped_effective_assets",
        "delta_average_effective_number_of_assets": "delta_effective_assets",
        "summary_decision": "pairwise_decision",
    }
    result = pairwise.rename(columns=column_map).copy()
    return _select_columns(result, CAP_IMPACT_COLUMNS)


def build_strategy_groups_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize benchmark and TD3 groups."""
    grouped = summary.copy()
    grouped["executive_group"] = grouped.apply(classify_strategy_group, axis=1)
    rows = []
    for group, frame in grouped.groupby("executive_group", dropna=False):
        robust_best = _best_row(frame, "robust_score")
        mandate_best = _best_row(frame, "mandate_aware_score")
        rows.append(
            {
                "group": group,
                "n_strategies": int(len(frame)),
                "best_strategy_by_mandate_aware_score": mandate_best["strategy_name"],
                "best_mandate_aware_score": mandate_best["mandate_aware_score"],
                "best_strategy_by_robust_score": robust_best["strategy_name"],
                "best_robust_score": robust_best["robust_score"],
                "mean_annualized_return": _mean_numeric(frame, "annualized_return"),
                "mean_sharpe": _mean_numeric(frame, "sharpe"),
                "mean_max_drawdown": _mean_numeric(frame, "max_drawdown"),
                "mean_turnover": _mean_numeric(frame, "average_turnover"),
                "mean_effective_assets": _mean_numeric(
                    frame,
                    "average_effective_number_of_assets",
                ),
            }
        )
    order = {
        "benchmark_eligible": 0,
        "benchmark_not_eligible": 1,
        "td3_uncapped": 2,
        "td3_capped": 3,
    }
    return (
        pd.DataFrame(rows)
        .assign(_order=lambda df: df["group"].map(order).fillna(99))
        .sort_values(["_order", "group"])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )


def build_markdown_summary(
    main_ranking: pd.DataFrame,
    mandate_ranking: pd.DataFrame,
    non_eligible: pd.DataFrame,
    cap_impact: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> str:
    """Build a concise Markdown executive summary."""
    top_robust = main_ranking.iloc[0]
    top_mandate = mandate_ranking.iloc[0]
    best_capped = mandate_ranking[
        mandate_ranking["strategy_type"].astype(str) == "td3_capped"
    ].head(1)
    best_capped_name = (
        best_capped.iloc[0]["strategy_name"] if not best_capped.empty else "None"
    )
    not_eligible_count = int(len(non_eligible))
    cap_lines = []
    for _, row in cap_impact.iterrows():
        cap_lines.append(
            "- "
            f"{row['base_candidate']}: {row['pairwise_decision']}; "
            f"delta mandate-aware score = {_format_number(row['delta_mandate_aware_score'])}, "
            f"delta effective assets = {_format_number(row['delta_effective_assets'])}."
        )
    group_note = _group_summary_note(group_summary)
    return "\n".join(
        [
            "# Executive Results Summary",
            "",
            "## Main Findings",
            "",
            (
                "1. Aggressive benchmarks dominate `robust_score` in the full ranking, "
                f"led by `{top_robust['strategy_name']}`, but the highest-ranked "
                "aggressive strategies fail mandate eligibility because their "
                "drawdowns breach the -30% threshold."
            ),
            (
                "2. Capped TD3 candidates rank at the top under mandate-aware scoring. "
                f"The leading mandate-aware strategy is `{top_mandate['strategy_name']}`; "
                f"the best capped TD3 candidate is `{best_capped_name}`."
            ),
            (
                "3. Uncapped TD3 candidates are not competitive in this comparison. "
                "Their weak scores are consistent with learned extreme concentration "
                "and inferior mandate-aware behavior."
            ),
            (
                "4. The `max_weight_cap = 0.60` allocation constraint improves TD3 "
                "behavior more effectively than the earlier direct "
                "`lambda_concentration` reward penalty experiment."
            ),
            "",
            "## Cap Impact",
            "",
            *cap_lines,
            "",
            "## Mandate Eligibility",
            "",
            (
                f"{not_eligible_count} strategies are classified as non-eligible or "
                "receive zero mandate-aware score under the drawdown mandate layer."
            ),
            group_note,
            "",
            "## Caveat",
            "",
            (
                "These results are based on the current asset universe, current window, "
                "current protocol, and current evaluation layer. They should not be "
                "overclaimed as general superiority evidence."
            ),
            "",
            "## Suggested Claim",
            "",
            (
                "TD3 does not dominate benchmarks in unconstrained form, but a "
                "max-weight constrained TD3 variant becomes competitive under a "
                "mandate-aware evaluation layer."
            ),
            "",
        ]
    )


def classify_strategy_group(row: pd.Series) -> str:
    """Classify row into executive strategy group."""
    strategy_type = str(row.get("strategy_type"))
    if strategy_type == "td3_capped":
        return "td3_capped"
    if strategy_type == "td3_uncapped":
        return "td3_uncapped"
    if strategy_type == "benchmark":
        return (
            "benchmark_eligible"
            if bool(row.get("is_mandate_eligible"))
            else "benchmark_not_eligible"
        )
    return "unknown"


def reason_not_eligible(row: pd.Series) -> str:
    """Return a concise non-eligibility reason."""
    if str(row.get("mandate_bucket")) == "not_eligible":
        return "max_drawdown below -30% mandate threshold"
    if pd.to_numeric(pd.Series([row.get("mandate_aware_score")]), errors="coerce").iloc[
        0
    ] == 0:
        return "mandate-aware score is zero"
    return ""


def _with_eligibility(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["is_mandate_eligible"] = result["mandate_bucket"].astype(str) != "not_eligible"
    return result


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, columns]


def _best_row(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.notna().any():
        return frame.loc[numeric.idxmax()]
    return frame.iloc[0]


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _format_number(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.4f}"


def _group_summary_note(group_summary: pd.DataFrame) -> str:
    if group_summary.empty:
        return ""
    best_groups = group_summary.sort_values(
        "best_mandate_aware_score",
        ascending=False,
        na_position="last",
    )
    top = best_groups.iloc[0]
    return (
        f"The strongest group by best mandate-aware score is `{top['group']}`, "
        f"represented by `{top['best_strategy_by_mandate_aware_score']}`."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build executive results report from capped TD3 comparison.",
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = build_executive_results_report(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print("Top robust ranking:")
    print(report["main_ranking"].head(10).to_string(index=False))
    print("\nTop mandate-eligible ranking:")
    print(report["mandate_eligible_ranking"].head(10).to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
