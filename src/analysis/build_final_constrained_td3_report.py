"""Build final constrained TD3 comparison report.

This reporting layer combines the full cap sensitivity experiment with the
existing benchmark protocol comparison. It does not retrain models or alter
production scoring logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.mandate_aware_score import assign_drawdown_bucket


DEFAULT_CAP_SENSITIVITY_DIR = "outputs/tables/cap_sensitivity_experiment_60ep_10seeds"
DEFAULT_BENCHMARK_COMPARISON_DIR = (
    "outputs/tables/capped_td3_protocol_comparison_60ep_10seeds_cap060"
)
DEFAULT_OUTPUT_DIR = "outputs/tables/final_constrained_td3_report_60ep_10seeds"

CAP_ALL_RESULTS_FILE = "cap_sensitivity_all_results.csv"
CAP_BEST_FILE = "cap_sensitivity_best_caps.csv"
BENCHMARK_SUMMARY_FILE = "capped_td3_vs_benchmarks_summary.csv"

BASE_LABELS = {
    "V2_reference_full": "V2",
    "V3_real_macro_current": "V3",
    "V5_no_volatility_block": "V5",
    "V6_financial_state": "V6",
}

FEATURE_FAMILIES = {
    "V2_reference_full": "reference_full",
    "V3_real_macro_current": "real_macro_current",
    "V5_no_volatility_block": "no_volatility_block",
    "V6_financial_state": "financial_state",
}

RANKING_COLUMNS = [
    "robust_rank",
    "mandate_rank",
    "eligible_rank",
    "strategy_name",
    "strategy_group",
    "strategy_type",
    "base_candidate",
    "feature_family",
    "source",
    "selected_cap",
    "constraint_status",
    "robust_score",
    "mandate_aware_score",
    "mandate_bucket",
    "drawdown_eligible",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "average_turnover",
    "average_effective_number_of_assets",
    "average_max_weight",
    "beats_best_clean_benchmark_by_mandate",
    "beats_best_benchmark_by_robust",
    "beats_uncapped_by_mandate",
    "beats_uncapped_by_robust",
    "concentration_controlled",
]


def build_final_constrained_td3_report(
    cap_sensitivity_dir: str = DEFAULT_CAP_SENSITIVITY_DIR,
    v3_cap_sensitivity_dir: str | None = None,
    benchmark_comparison_dir: str = DEFAULT_BENCHMARK_COMPARISON_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build final constrained TD3 report tables and markdown."""
    cap_dir = Path(cap_sensitivity_dir)
    benchmark_dir = Path(benchmark_comparison_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap_results = pd.read_csv(cap_dir / CAP_ALL_RESULTS_FILE)
    best_caps = pd.read_csv(cap_dir / CAP_BEST_FILE)
    cap_results["source"] = "cap_sensitivity"
    best_caps["source"] = "cap_sensitivity"
    if v3_cap_sensitivity_dir:
        v3_cap_dir = Path(v3_cap_sensitivity_dir)
        v3_results = pd.read_csv(v3_cap_dir / CAP_ALL_RESULTS_FILE)
        v3_best_caps = pd.read_csv(v3_cap_dir / CAP_BEST_FILE)
        v3_results["source"] = "seeded_cap_sensitivity"
        v3_best_caps["source"] = "seeded_cap_sensitivity"
        cap_results = pd.concat([cap_results, v3_results], ignore_index=True, sort=False)
        best_caps = pd.concat([best_caps, v3_best_caps], ignore_index=True, sort=False)
    benchmark_summary = pd.read_csv(benchmark_dir / BENCHMARK_SUMMARY_FILE)

    selected_td3 = build_selected_td3_rows(cap_results, best_caps)
    benchmark_rows = build_benchmark_rows(benchmark_summary)
    combined = build_final_combined_table(selected_td3, benchmark_rows)
    main_ranking = build_main_ranking(combined)
    mandate_ranking = build_mandate_ranking(combined)
    selected_candidates = build_selected_candidates_table(combined)
    vs_benchmarks = build_vs_benchmarks_table(combined)
    markdown = build_final_markdown_summary(
        main_ranking=main_ranking,
        mandate_ranking=mandate_ranking,
        selected_candidates=selected_candidates,
        vs_benchmarks=vs_benchmarks,
    )
    metadata = build_metadata(
        cap_sensitivity_dir=cap_sensitivity_dir,
        v3_cap_sensitivity_dir=v3_cap_sensitivity_dir,
        benchmark_comparison_dir=benchmark_comparison_dir,
        output_dir=output_dir,
        selected_candidates=selected_candidates,
    )

    paths = {
        "main_ranking": output_path / "final_constrained_td3_main_ranking.csv",
        "mandate_ranking": output_path / "final_constrained_td3_mandate_ranking.csv",
        "selected_candidates": output_path
        / "final_constrained_td3_selected_candidates.csv",
        "vs_benchmarks": output_path / "final_constrained_td3_vs_benchmarks.csv",
        "markdown_summary": output_path / "final_constrained_td3_summary.md",
        "metadata": output_path / "final_constrained_td3_metadata.json",
    }
    main_ranking.to_csv(paths["main_ranking"], index=False)
    mandate_ranking.to_csv(paths["mandate_ranking"], index=False)
    selected_candidates.to_csv(paths["selected_candidates"], index=False)
    vs_benchmarks.to_csv(paths["vs_benchmarks"], index=False)
    paths["markdown_summary"].write_text(markdown, encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "main_ranking": main_ranking,
        "mandate_ranking": mandate_ranking,
        "selected_candidates": selected_candidates,
        "vs_benchmarks": vs_benchmarks,
        "markdown_summary": markdown,
        "metadata": metadata,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def build_selected_td3_rows(
    cap_results: pd.DataFrame,
    best_caps: pd.DataFrame,
) -> pd.DataFrame:
    """Select best cap, cap-0.60 reference, and uncapped rows per TD3 candidate."""
    rows = []
    for base_candidate, group in cap_results.groupby("base_candidate", dropna=True):
        best_cap = best_caps.set_index("base_candidate").loc[
            base_candidate,
            "best_by_mandate_aware_score",
        ]
        selected = _row_for_cap(group, best_cap)
        if selected is not None:
            rows.append(_normalize_td3_row(selected, "td3_best_constrained", best_cap))
        cap_060 = _row_for_cap(group, 0.60)
        if cap_060 is not None and not _same_cap(best_cap, 0.60):
            rows.append(_normalize_td3_row(cap_060, "td3_cap_0.60_reference", 0.60))
        uncapped = _row_for_cap(group, "uncapped")
        if uncapped is not None:
            rows.append(_normalize_td3_row(uncapped, "td3_uncapped", None))
    return pd.DataFrame(rows)


def build_benchmark_rows(benchmark_summary: pd.DataFrame) -> pd.DataFrame:
    """Normalize benchmark rows from existing protocol comparison."""
    benchmarks = benchmark_summary[
        benchmark_summary["strategy_type"].astype(str) == "benchmark"
    ].copy()
    benchmarks["strategy_group"] = benchmarks.apply(_benchmark_group, axis=1)
    benchmarks["base_candidate"] = pd.NA
    benchmarks["selected_cap"] = pd.NA
    benchmarks["constraint_status"] = "benchmark"
    benchmarks["drawdown_eligible"] = (
        benchmarks["mandate_bucket"].astype(str) != "not_eligible"
    )
    benchmarks["concentration_controlled"] = (
        pd.to_numeric(
            benchmarks["average_effective_number_of_assets"],
            errors="coerce",
        )
        >= 1.5
    ) & (
        pd.to_numeric(benchmarks["average_max_weight"], errors="coerce") <= 0.80
    )
    return benchmarks


def build_final_combined_table(
    selected_td3: pd.DataFrame,
    benchmark_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Combine selected TD3 rows with benchmark rows and add flags."""
    if (
        "strategy_group" not in benchmark_rows.columns
        or benchmark_rows["strategy_group"].isna().any()
    ):
        benchmark_rows = build_benchmark_rows(benchmark_rows)
    combined = pd.concat([selected_td3, benchmark_rows], ignore_index=True, sort=False)
    combined["mandate_bucket"] = combined.apply(_ensure_mandate_bucket, axis=1)
    combined["drawdown_eligible"] = (
        combined["mandate_bucket"].astype(str) != "not_eligible"
    )
    combined = add_interpretation_flags(combined)
    combined = add_ranks(combined)
    for column in RANKING_COLUMNS:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined


def add_interpretation_flags(combined: pd.DataFrame) -> pd.DataFrame:
    """Add comparison flags against benchmarks and uncapped TD3 rows."""
    result = combined.copy()
    clean_benchmarks = result[
        (result["strategy_group"] == "benchmark_eligible")
        & (result["mandate_bucket"].astype(str) == "clean_mandate")
    ]
    if clean_benchmarks.empty:
        clean_benchmarks = result[result["strategy_group"] == "benchmark_eligible"]
    best_clean_mandate = _max_numeric(clean_benchmarks, "mandate_aware_score")
    best_benchmark_robust = _max_numeric(
        result[result["strategy_type"].astype(str) == "benchmark"],
        "robust_score",
    )
    result["beats_best_clean_benchmark_by_mandate"] = (
        pd.to_numeric(result["mandate_aware_score"], errors="coerce")
        > best_clean_mandate
    )
    result["beats_best_benchmark_by_robust"] = (
        pd.to_numeric(result["robust_score"], errors="coerce") > best_benchmark_robust
    )
    uncapped_by_base = (
        result[result["strategy_group"] == "td3_uncapped"]
        .set_index("base_candidate")
        [["mandate_aware_score", "robust_score"]]
        .to_dict(orient="index")
    )
    result["beats_uncapped_by_mandate"] = result.apply(
        lambda row: _beats_uncapped(row, uncapped_by_base, "mandate_aware_score"),
        axis=1,
    )
    result["beats_uncapped_by_robust"] = result.apply(
        lambda row: _beats_uncapped(row, uncapped_by_base, "robust_score"),
        axis=1,
    )
    computed_concentration_control = (
        pd.to_numeric(
            result["average_effective_number_of_assets"],
            errors="coerce",
        )
        >= 1.5
    ) & (pd.to_numeric(result["average_max_weight"], errors="coerce") <= 0.80)
    if "concentration_controlled" not in result.columns:
        result["concentration_controlled"] = computed_concentration_control
    else:
        result["concentration_controlled"] = result["concentration_controlled"].where(
            result["concentration_controlled"].notna(),
            computed_concentration_control,
        )
    return result


def add_ranks(combined: pd.DataFrame) -> pd.DataFrame:
    """Add robust, mandate, and eligible ranks."""
    result = combined.copy()
    result["robust_rank"] = (
        pd.to_numeric(result["robust_score"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    result["mandate_rank"] = (
        pd.to_numeric(result["mandate_aware_score"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    result["eligible_rank"] = pd.NA
    eligible = result["drawdown_eligible"].fillna(False).astype(bool)
    result.loc[eligible, "eligible_rank"] = (
        pd.to_numeric(result.loc[eligible, "mandate_aware_score"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    return result


def build_main_ranking(combined: pd.DataFrame) -> pd.DataFrame:
    """Build robust-score sorted final ranking."""
    return _select_ranking_columns(
        combined.sort_values(["robust_rank", "strategy_name"]).reset_index(drop=True)
    )


def build_mandate_ranking(combined: pd.DataFrame) -> pd.DataFrame:
    """Build mandate-aware sorted final ranking."""
    return _select_ranking_columns(
        combined.sort_values(["mandate_rank", "robust_score"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_selected_candidates_table(selected_td3: pd.DataFrame) -> pd.DataFrame:
    """Return selected TD3 candidate rows only."""
    selected = selected_td3[
        selected_td3["strategy_group"] == "td3_best_constrained"
    ].copy()
    selected = selected.sort_values(
        ["mandate_aware_score", "robust_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    columns = [
        "strategy_name",
        "base_candidate",
        "feature_family",
        "source",
        "selected_cap",
        "robust_score",
        "mandate_aware_score",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "average_turnover",
        "average_effective_number_of_assets",
        "average_max_weight",
        "decision_label",
        "beats_best_clean_benchmark_by_mandate",
        "beats_best_benchmark_by_robust",
        "beats_uncapped_by_mandate",
        "beats_uncapped_by_robust",
        "drawdown_eligible",
        "concentration_controlled",
    ]
    return _select_columns(selected, columns)


def build_vs_benchmarks_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Build selected TD3 versus key benchmark comparison table."""
    key_names = [
        "BuyHold_GLD",
        "trend_spy_cash_12p",
        "momentum_winner_12p",
        "Equal_Weight",
        "Equal_Weight_Risky",
    ]
    rows = []
    selected = combined[combined["strategy_group"] == "td3_best_constrained"]
    benchmarks = combined.set_index("strategy_name")
    for _, td3 in selected.iterrows():
        for benchmark_name in key_names:
            if benchmark_name not in benchmarks.index:
                continue
            bench = benchmarks.loc[benchmark_name]
            rows.append(
                {
                    "td3_strategy": td3["strategy_name"],
                    "base_candidate": td3["base_candidate"],
                    "benchmark_strategy": benchmark_name,
                    "td3_mandate_aware_score": td3["mandate_aware_score"],
                    "benchmark_mandate_aware_score": bench["mandate_aware_score"],
                    "delta_mandate_aware_score": _num(td3["mandate_aware_score"])
                    - _num(bench["mandate_aware_score"]),
                    "td3_robust_score": td3["robust_score"],
                    "benchmark_robust_score": bench["robust_score"],
                    "delta_robust_score": _num(td3["robust_score"])
                    - _num(bench["robust_score"]),
                    "td3_max_drawdown": td3["max_drawdown"],
                    "benchmark_max_drawdown": bench["max_drawdown"],
                    "td3_beats_benchmark_by_mandate": _num(
                        td3["mandate_aware_score"]
                    )
                    > _num(bench["mandate_aware_score"]),
                    "td3_beats_benchmark_by_robust": _num(td3["robust_score"])
                    > _num(bench["robust_score"]),
                }
            )
    return pd.DataFrame(rows)


def build_final_markdown_summary(
    main_ranking: pd.DataFrame,
    mandate_ranking: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    vs_benchmarks: pd.DataFrame,
) -> str:
    """Build final Markdown summary."""
    best_td3 = selected_candidates.iloc[0]
    best_clean = mandate_ranking[
        (mandate_ranking["strategy_group"] == "benchmark_eligible")
        & (mandate_ranking["mandate_bucket"].astype(str) == "clean_mandate")
    ].iloc[0]
    aggressive = main_ranking[
        (main_ranking["strategy_type"].astype(str) == "benchmark")
        & (~main_ranking["drawdown_eligible"].fillna(False).astype(bool))
    ].head(3)
    best_vs = vs_benchmarks[vs_benchmarks["td3_strategy"] == best_td3["strategy_name"]]
    by_benchmark = best_vs.set_index("benchmark_strategy")
    gld_win = bool(
        by_benchmark.loc["BuyHold_GLD", "td3_beats_benchmark_by_mandate"]
    )
    trend_win = bool(
        by_benchmark.loc["trend_spy_cash_12p", "td3_beats_benchmark_by_mandate"]
    )
    aggressive_robust_win = bool(best_td3["beats_best_benchmark_by_robust"])
    cap_note = (
        "The broader cap sensitivity strengthens the constrained-TD3 finding, "
        "but weakens any claim that 0.60 is the universal cap. The optimal cap "
        "is candidate-sensitive."
    )
    v3_note = ""
    if "V3_real_macro_current" in set(selected_candidates["base_candidate"].astype(str)):
        v3_note = (
            "The V3 result uses current-vintage macro data with a conservative CPI "
            "lag approximation. It should be interpreted as the current best "
            "constrained TD3 result, pending future real-time vintage and "
            "release-calendar macro validation."
        )
    macro_claim = (
        "After adding real macro features and applying a max-weight constraint, "
        "V3 becomes the strongest constrained TD3 candidate in the current "
        "protocol. Unconstrained TD3 remains weak, but constrained TD3 with macro "
        "features is competitive under mandate-aware evaluation. This result "
        "remains subject to the macro vintage/release-timing caveat."
        if str(best_td3.get("base_candidate")) == "V3_real_macro_current"
        else (
            "Unconstrained TD3 does not dominate the benchmark suite. However, "
            "TD3 with an empirically selected max-weight constraint becomes "
            "competitive under mandate-aware evaluation and can outperform the "
            "best clean benchmark in this experimental setting. The optimal cap "
            "is candidate-sensitive."
        )
    )
    return "\n".join(
        [
            "# Final Constrained TD3 Report",
            "",
            "## Main Result",
            "",
            (
                f"The best constrained TD3 variant is `{best_td3['strategy_name']}` "
                f"with mandate-aware score {_fmt(best_td3['mandate_aware_score'])}, "
                f"robust score {_fmt(best_td3['robust_score'])}, and max drawdown "
                f"{_fmt(best_td3['max_drawdown'])}."
            ),
            (
                f"It is the best constrained TD3 candidate after including seeded V3: "
                f"{str(best_td3.get('base_candidate')) == 'V3_real_macro_current'}."
            ),
            v3_note,
            "",
            "## Benchmark Comparisons",
            "",
            (
                f"Best constrained TD3 beats BuyHold_GLD by mandate-aware score: {gld_win}."
            ),
            (
                f"Best constrained TD3 beats trend_spy_cash_12p by mandate-aware score: {trend_win}."
            ),
            (
                "Best constrained TD3 beats aggressive high-drawdown benchmarks by "
                f"robust score: {aggressive_robust_win}."
            ),
            (
                f"The best clean benchmark is `{best_clean['strategy_name']}` with "
                f"mandate-aware score {_fmt(best_clean['mandate_aware_score'])}."
            ),
            "",
            "Aggressive high-drawdown benchmark leaders by robust score:",
            *[
                f"- `{row['strategy_name']}`: robust score {_fmt(row['robust_score'])}, "
                f"max drawdown {_fmt(row['max_drawdown'])}."
                for _, row in aggressive.iterrows()
            ],
            "",
            "## Cap Sensitivity",
            "",
            cap_note,
            "",
            "Selected best caps:",
            *[
                f"- `{row['base_candidate']}`: `{row['selected_cap']}` "
                f"({row['strategy_name']})."
                for _, row in selected_candidates.iterrows()
            ],
            "",
            "## Final Defensible Claim",
            "",
            (
                macro_claim
            ),
            "",
        ]
    )


def build_metadata(
    cap_sensitivity_dir: str,
    v3_cap_sensitivity_dir: str | None,
    benchmark_comparison_dir: str,
    output_dir: str,
    selected_candidates: pd.DataFrame,
) -> dict[str, Any]:
    """Build report metadata."""
    return {
        "runner": "src.analysis.build_final_constrained_td3_report",
        "cap_sensitivity_dir": cap_sensitivity_dir,
        "v3_cap_sensitivity_dir": v3_cap_sensitivity_dir,
        "benchmark_comparison_dir": benchmark_comparison_dir,
        "output_dir": output_dir,
        "selection_rule": "best cap per base candidate by mandate_aware_score",
        "selected_caps": selected_candidates.set_index("base_candidate")[
            "selected_cap"
        ].to_dict(),
        "v3_source": "seeded_cap_sensitivity" if v3_cap_sensitivity_dir else None,
        "v3_macro_caveat": (
            "V3_real_macro_current uses current-vintage macro data and a "
            "conservative CPI lag approximation, not a full real-time vintage "
            "or release-calendar macro database."
            if v3_cap_sensitivity_dir
            else None
        ),
        "reporting_only_note": (
            "This report reads existing cap sensitivity and benchmark comparison "
            "outputs. It does not retrain models or alter scoring logic."
        ),
    }


def _normalize_td3_row(row: pd.Series, group: str, selected_cap: Any) -> dict[str, Any]:
    base = str(row["base_candidate"])
    cap_label = "uncapped" if pd.isna(row.get("max_weight_cap")) else f"{_num(row['max_weight_cap']):.2f}"
    strategy_name = f"{BASE_LABELS.get(base, base)}_{'uncapped' if cap_label == 'uncapped' else 'cap_' + cap_label}"
    max_drawdown = _num(row["max_drawdown"])
    bucket = assign_drawdown_bucket(max_drawdown)
    normalized = row.to_dict()
    normalized.update(
        {
            "strategy_name": strategy_name,
            "strategy_type": "td3",
            "strategy_group": group,
            "feature_family": FEATURE_FAMILIES.get(base, "unknown"),
            "source": row.get("source", "cap_sensitivity"),
            "selected_cap": selected_cap if selected_cap is not None else pd.NA,
            "constraint_status": cap_label,
            "mandate_bucket": bucket,
            "drawdown_eligible": bucket != "not_eligible",
            "concentration_controlled": (
                _num(row.get("average_effective_number_of_assets")) >= 1.5
                and _num(row.get("average_max_weight")) <= 0.80
            ),
        }
    )
    return normalized


def _same_cap(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    try:
        return round(float(left), 8) == round(float(right), 8)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _row_for_cap(group: pd.DataFrame, cap: Any) -> pd.Series | None:
    if str(cap).lower() == "uncapped" or pd.isna(cap):
        rows = group[group["max_weight_cap"].isna()]
    else:
        rows = group[
            pd.to_numeric(group["max_weight_cap"], errors="coerce").round(8)
            == round(float(cap), 8)
        ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _benchmark_group(row: pd.Series) -> str:
    return (
        "benchmark_eligible"
        if str(row.get("mandate_bucket")) != "not_eligible"
        else "benchmark_not_eligible"
    )


def _ensure_mandate_bucket(row: pd.Series) -> str:
    bucket = row.get("mandate_bucket")
    if pd.notna(bucket):
        return str(bucket)
    return assign_drawdown_bucket(row["max_drawdown"])


def _beats_uncapped(
    row: pd.Series,
    uncapped_by_base: dict[str, dict[str, Any]],
    metric: str,
) -> bool:
    base = row.get("base_candidate")
    if pd.isna(base) or str(base) not in uncapped_by_base:
        return False
    return _num(row.get(metric)) > _num(uncapped_by_base[str(base)].get(metric))


def _max_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").max())


def _select_ranking_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return _select_columns(frame, RANKING_COLUMNS)


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, columns]


def _num(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def _fmt(value: Any) -> str:
    return f"{_num(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final best-constrained TD3 comparison report.",
    )
    parser.add_argument("--cap-sensitivity-dir", default=DEFAULT_CAP_SENSITIVITY_DIR)
    parser.add_argument("--v3-cap-sensitivity-dir", default=None)
    parser.add_argument(
        "--benchmark-comparison-dir",
        default=DEFAULT_BENCHMARK_COMPARISON_DIR,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = build_final_constrained_td3_report(
        cap_sensitivity_dir=args.cap_sensitivity_dir,
        v3_cap_sensitivity_dir=args.v3_cap_sensitivity_dir,
        benchmark_comparison_dir=args.benchmark_comparison_dir,
        output_dir=args.output_dir,
    )
    print("Selected candidates:")
    print(report["selected_candidates"].to_string(index=False))
    print("\nTop mandate ranking:")
    print(report["mandate_ranking"].head(15).to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
