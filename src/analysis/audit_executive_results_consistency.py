"""Audit executive capped TD3 results for methodological consistency.

This module is reporting-only. It reads the capped protocol comparison and the
executive report outputs, then writes pass/warning/fail consistency checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.mandate_aware_score import (
    assign_drawdown_bucket,
    get_drawdown_multiplier,
)


DEFAULT_COMPARISON_DIR = "outputs/tables/capped_td3_protocol_comparison_60ep_10seeds_cap060"
DEFAULT_EXECUTIVE_DIR = "outputs/tables/executive_results_report_60ep_10seeds_cap060"
DEFAULT_OUTPUT_DIR = "outputs/tables/executive_results_consistency_audit_60ep_10seeds_cap060"

SUMMARY_FILE = "capped_td3_vs_benchmarks_summary.csv"
PAIRWISE_FILE = "capped_td3_pairwise_deltas.csv"
METADATA_FILE = "capped_td3_protocol_metadata.json"
EXECUTIVE_MAIN_FILE = "executive_main_ranking.csv"
EXECUTIVE_MANDATE_FILE = "executive_mandate_eligible_ranking.csv"
EXECUTIVE_NON_ELIGIBLE_FILE = "executive_non_eligible_strategies.csv"

EXPECTED_SOURCE_FOLDERS = {
    "V2_reference_full": "outputs/tables/max_weight_cap_experiment_v2_60ep_10seeds_cap060",
    "V5_no_volatility_block": "outputs/tables/max_weight_cap_experiment_v5_60ep_10seeds_cap060",
    "V6_financial_state": "outputs/tables/max_weight_cap_experiment_v6_60ep_10seeds_cap060",
}

NUMERIC_CONSISTENCY_COLUMNS = [
    "annualized_return",
    "max_drawdown",
    "sharpe",
    "average_turnover",
    "average_effective_number_of_assets",
    "average_max_weight",
    "robust_score",
    "mandate_aware_score",
]


def audit_executive_results_consistency(
    comparison_dir: str = DEFAULT_COMPARISON_DIR,
    executive_dir: str = DEFAULT_EXECUTIVE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run all executive consistency checks and write audit outputs."""
    comparison_path = Path(comparison_dir)
    executive_path = Path(executive_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inputs = load_audit_inputs(comparison_path, executive_path)
    checks = build_consistency_checks(inputs)
    issues = checks[checks["status"].isin(["warning", "fail"])].copy()
    summary_md = build_consistency_summary(checks, issues)

    checks_path = output_path / "executive_consistency_checks.csv"
    issues_path = output_path / "executive_consistency_issues.csv"
    summary_path = output_path / "executive_consistency_summary.md"
    checks.to_csv(checks_path, index=False)
    issues.to_csv(issues_path, index=False)
    summary_path.write_text(summary_md, encoding="utf-8")

    return {
        "checks": checks,
        "issues": issues,
        "summary_markdown": summary_md,
        "paths": {
            "checks": str(checks_path),
            "issues": str(issues_path),
            "summary": str(summary_path),
        },
    }


def load_audit_inputs(comparison_path: Path, executive_path: Path) -> dict[str, Any]:
    """Load required comparison and executive report files."""
    summary = _read_csv(comparison_path / SUMMARY_FILE)
    pairwise = _read_csv(comparison_path / PAIRWISE_FILE)
    executive_main = _read_csv(executive_path / EXECUTIVE_MAIN_FILE)
    executive_mandate = _read_csv(executive_path / EXECUTIVE_MANDATE_FILE)
    executive_non_eligible = _read_csv(executive_path / EXECUTIVE_NON_ELIGIBLE_FILE)
    metadata_path = comparison_path / METADATA_FILE
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "comparison_path": comparison_path,
        "executive_path": executive_path,
        "summary": summary,
        "pairwise": pairwise,
        "executive_main": executive_main,
        "executive_mandate": executive_mandate,
        "executive_non_eligible": executive_non_eligible,
        "metadata": metadata,
    }


def build_consistency_checks(inputs: dict[str, Any]) -> pd.DataFrame:
    """Build the pass/warning/fail audit table."""
    checks = [
        check_td3_rows_are_test_only(inputs),
        check_benchmarks_use_same_protocol(inputs),
        check_score_comparability(inputs),
        check_mandate_score_formula(inputs),
        check_drawdown_buckets(inputs),
        check_duplicate_strategy_rows(inputs),
        check_metric_column_consistency(inputs),
        check_td3_source_folders(inputs),
        check_executive_has_no_train_validation_rows(inputs),
        check_not_eligible_scores(inputs),
        check_eligible_scores(inputs),
        check_capped_td3_above_buyhold_gld(inputs),
    ]
    return pd.DataFrame(checks)


def check_td3_rows_are_test_only(inputs: dict[str, Any]) -> dict[str, str]:
    """Verify capped TD3 source ranking rows are test split only."""
    summary = inputs["summary"]
    td3 = summary[summary["strategy_type"].astype(str).str.startswith("td3")]
    problems = []
    for source_path, group in td3.groupby("source_path", dropna=True):
        ranking_path = Path(str(source_path)) / "max_weight_cap_rankings.csv"
        if not ranking_path.exists():
            problems.append(f"missing source ranking: {ranking_path}")
            continue
        source = pd.read_csv(ranking_path)
        if "split" not in source.columns:
            problems.append(f"missing split column: {ranking_path}")
            continue
        candidates = set(group["candidate_name"].dropna().astype(str))
        matched = source[source["candidate_name"].astype(str).isin(candidates)]
        bad_splits = sorted(set(matched.loc[matched["split"] != "test", "split"]))
        if bad_splits:
            problems.append(f"{ranking_path} has non-test splits {bad_splits}")
        if len(matched) != len(candidates):
            problems.append(f"{ranking_path} missing matched test candidate rows")
    return _check(
        "td3_test_split_only",
        "TD3 capped/uncapped rows use split == test only.",
        "fail" if problems else "pass",
        "; ".join(problems) if problems else f"Checked {len(td3)} TD3 rows.",
    )


def check_benchmarks_use_same_protocol(inputs: dict[str, Any]) -> dict[str, str]:
    """Verify benchmark rows were regenerated by the protocol runner."""
    metadata = inputs["metadata"]
    comparison_path = inputs["comparison_path"]
    benchmark_dir = Path(str(metadata.get("benchmark_output_dir", "")))
    required_files = [
        benchmark_dir / "benchmark_metrics_table.csv",
        benchmark_dir / "benchmark_comparison_summary.csv",
        benchmark_dir / "benchmark_diagnostics.csv",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    protocol_ok = (
        metadata.get("returns_path") == "data/processed/returns_weekly_latest.csv"
        and "information through t-1" in str(metadata.get("timing_convention", ""))
        and "sum(abs" in str(metadata.get("turnover_convention", ""))
        and str(benchmark_dir).startswith(str(comparison_path))
    )
    if missing:
        return _check(
            "benchmark_same_returns_protocol",
            "Benchmark rows are produced on the same returns file and protocol.",
            "fail",
            f"Missing benchmark outputs: {missing}",
        )
    if not protocol_ok:
        return _check(
            "benchmark_same_returns_protocol",
            "Benchmark rows are produced on the same returns file and protocol.",
            "warning",
            "Metadata does not fully match expected returns/protocol conventions.",
        )
    return _check(
        "benchmark_same_returns_protocol",
        "Benchmark rows are produced on the same returns file and protocol.",
        "pass",
        f"Benchmarks regenerated under {benchmark_dir}.",
    )


def check_score_comparability(inputs: dict[str, Any]) -> dict[str, str]:
    """Check robust and mandate-aware score comparability caveat."""
    summary = inputs["summary"]
    metadata = inputs["metadata"]
    benchmark_methods = set(
        summary.loc[summary["strategy_type"] == "benchmark", "dsr_method"]
        .dropna()
        .astype(str)
    )
    td3_methods = set(
        summary.loc[
            summary["strategy_type"].astype(str).str.startswith("td3"),
            "dsr_method",
        ]
        .dropna()
        .astype(str)
    )
    robust_complete = summary["robust_score"].notna().all()
    mandate_complete = summary["mandate_aware_score"].notna().all()
    if not robust_complete or not mandate_complete:
        return _check(
            "score_comparability",
            "robust_score and mandate_aware_score are present for TD3 and benchmarks.",
            "fail",
            "One or more strategies have missing score fields.",
        )
    note = metadata.get("benchmark_robust_score_note", "")
    if benchmark_methods != td3_methods:
        return _check(
            "score_comparability",
            "robust_score and mandate_aware_score are comparable with DSR caveat.",
            "warning",
            (
                f"Benchmark DSR methods {sorted(benchmark_methods)} differ from "
                f"TD3 methods {sorted(td3_methods)}; metadata documents this: {note}"
            ),
        )
    return _check(
        "score_comparability",
        "robust_score and mandate_aware_score are comparable with DSR caveat.",
        "pass",
        "All strategy scores are present and DSR methods match.",
    )


def check_mandate_score_formula(inputs: dict[str, Any]) -> dict[str, str]:
    """Verify mandate-aware score uses the recovery-based formula."""
    summary = inputs["summary"]
    mismatches = _mandate_formula_mismatches(summary)
    status = "fail" if mismatches else "pass"
    return _check(
        "mandate_score_formula",
        "Benchmark and TD3 mandate-aware fields use the same recovery formula.",
        status,
        "; ".join(mismatches[:5]) if mismatches else "All rows match expected formula.",
    )


def check_drawdown_buckets(inputs: dict[str, Any]) -> dict[str, str]:
    """Verify drawdown buckets are consistent across strategy types."""
    summary = inputs["summary"]
    mismatches = []
    for _, row in summary.iterrows():
        expected = assign_drawdown_bucket(row["max_drawdown"])
        actual = str(row["mandate_bucket"])
        if actual != expected:
            mismatches.append(f"{row['strategy_name']}: expected {expected}, got {actual}")
    return _check(
        "drawdown_bucket_consistency",
        "Drawdown buckets are consistent across all strategy types.",
        "fail" if mismatches else "pass",
        "; ".join(mismatches[:5]) if mismatches else "All buckets match max_drawdown.",
    )


def check_duplicate_strategy_rows(inputs: dict[str, Any]) -> dict[str, str]:
    """Detect duplicate strategy names in comparison or executive outputs."""
    duplicates = []
    for label in ["summary", "executive_main", "executive_mandate"]:
        frame = inputs[label]
        if "strategy_name" not in frame.columns:
            continue
        dupes = sorted(frame.loc[frame["strategy_name"].duplicated(), "strategy_name"])
        if dupes:
            duplicates.append(f"{label}: {dupes}")
    return _check(
        "duplicate_strategy_rows",
        "There are no duplicate strategy rows.",
        "fail" if duplicates else "pass",
        "; ".join(duplicates) if duplicates else "No duplicate strategy names detected.",
    )


def check_metric_column_consistency(inputs: dict[str, Any]) -> dict[str, str]:
    """Verify executive metric values match the comparison summary."""
    summary = inputs["summary"].set_index("strategy_name")
    executive = inputs["executive_main"].set_index("strategy_name")
    missing = sorted(set(summary.index).difference(executive.index))
    mismatches = []
    for strategy in sorted(set(summary.index).intersection(executive.index)):
        for column in NUMERIC_CONSISTENCY_COLUMNS:
            if column not in summary.columns or column not in executive.columns:
                continue
            left = _num(summary.loc[strategy, column])
            right = _num(executive.loc[strategy, column])
            if pd.isna(left) and pd.isna(right):
                continue
            if not np.isclose(left, right, equal_nan=True, atol=1e-10):
                mismatches.append(f"{strategy}.{column}: {left} != {right}")
    if missing or mismatches:
        return _check(
            "metric_column_consistency",
            "Core metric column meanings are consistent between comparison and executive outputs.",
            "fail",
            f"Missing={missing}; mismatches={mismatches[:5]}",
        )
    return _check(
        "metric_column_consistency",
        "Core metric column meanings are consistent between comparison and executive outputs.",
        "pass",
        "Executive metrics match comparison summary values.",
    )


def check_td3_source_folders(inputs: dict[str, Any]) -> dict[str, str]:
    """Verify TD3 capped rows come from the requested experiment folders."""
    metadata = inputs["metadata"]
    summary = inputs["summary"]
    actual = metadata.get("input_experiment_folders", {})
    problems = []
    for key, expected in EXPECTED_SOURCE_FOLDERS.items():
        if actual.get(key) != expected:
            problems.append(f"{key}: expected {expected}, got {actual.get(key)}")
    td3_sources = set(
        summary.loc[
            summary["strategy_type"].astype(str).str.startswith("td3"),
            "source_path",
        ]
        .dropna()
        .astype(str)
    )
    unexpected_sources = sorted(td3_sources.difference(EXPECTED_SOURCE_FOLDERS.values()))
    if unexpected_sources:
        problems.append(f"unexpected TD3 source_path values: {unexpected_sources}")
    return _check(
        "td3_source_folders",
        "Capped TD3 rows come from the requested source folders.",
        "fail" if problems else "pass",
        "; ".join(problems) if problems else "All TD3 source folders match request.",
    )


def check_executive_has_no_train_validation_rows(inputs: dict[str, Any]) -> dict[str, str]:
    """Verify executive report did not include train or validation rows."""
    executive_frames = [inputs["executive_main"], inputs["executive_mandate"]]
    bad = []
    for index, frame in enumerate(executive_frames):
        if "split" in frame.columns:
            values = sorted(set(frame["split"].dropna().astype(str)))
            if values and values != ["test"]:
                bad.append(f"executive_frame_{index} has split values {values}")
    source_check = check_td3_rows_are_test_only(inputs)
    if source_check["status"] == "fail":
        bad.append(source_check["details"])
    return _check(
        "executive_no_train_validation_rows",
        "Executive report does not include train or validation rows.",
        "fail" if bad else "pass",
        "; ".join(bad) if bad else "No train/validation split rows detected.",
    )


def check_not_eligible_scores(inputs: dict[str, Any]) -> dict[str, str]:
    """Verify not-eligible or >30% drawdown rows have zero mandate score."""
    summary = inputs["summary"]
    score = pd.to_numeric(summary["mandate_aware_score"], errors="coerce")
    max_dd = pd.to_numeric(summary["max_drawdown"], errors="coerce")
    bad = summary[
        ((summary["mandate_bucket"].astype(str) == "not_eligible") | (max_dd < -0.30))
        & (score > 1e-12)
    ]
    return _check(
        "not_eligible_zero_score",
        "No strategy worse than -30% drawdown receives positive mandate-aware score.",
        "fail" if not bad.empty else "pass",
        (
            f"Positive scores for {bad['strategy_name'].tolist()}"
            if not bad.empty
            else "All non-eligible rows have zero mandate-aware score."
        ),
    )


def check_eligible_scores(inputs: dict[str, Any]) -> dict[str, str]:
    """Verify eligible strategies do not unexpectedly receive zero mandate score."""
    summary = inputs["summary"]
    score = pd.to_numeric(summary["mandate_aware_score"], errors="coerce")
    robust = pd.to_numeric(summary["robust_score"], errors="coerce")
    max_dd = pd.to_numeric(summary["max_drawdown"], errors="coerce")
    bad = summary[(max_dd >= -0.30) & (robust > 0.0) & (score <= 0.0)]
    return _check(
        "eligible_nonzero_score",
        "Eligible strategies with positive robust_score have nonzero mandate-aware score.",
        "fail" if not bad.empty else "pass",
        (
            f"Unexpected zero scores for {bad['strategy_name'].tolist()}"
            if not bad.empty
            else "All eligible positive-robust rows have nonzero mandate-aware score."
        ),
    )


def check_capped_td3_above_buyhold_gld(inputs: dict[str, Any]) -> dict[str, str]:
    """Verify V5/V2 capped strategies rank above BuyHold_GLD by mandate score."""
    summary = inputs["summary"].set_index("strategy_name")
    required = ["V5_cap_0.60", "V2_cap_0.60", "BuyHold_GLD"]
    missing = [name for name in required if name not in summary.index]
    if missing:
        return _check(
            "v5_v2_above_buyhold_gld",
            "V5_cap_0.60 and V2_cap_0.60 are above BuyHold_GLD by mandate-aware score.",
            "fail",
            f"Missing rows: {missing}",
        )
    gld = _num(summary.loc["BuyHold_GLD", "mandate_aware_score"])
    v5 = _num(summary.loc["V5_cap_0.60", "mandate_aware_score"])
    v2 = _num(summary.loc["V2_cap_0.60", "mandate_aware_score"])
    passed = v5 > gld and v2 > gld
    return _check(
        "v5_v2_above_buyhold_gld",
        "V5_cap_0.60 and V2_cap_0.60 are above BuyHold_GLD by mandate-aware score.",
        "pass" if passed else "fail",
        f"V5={v5:.6f}, V2={v2:.6f}, BuyHold_GLD={gld:.6f}",
    )


def build_consistency_summary(checks: pd.DataFrame, issues: pd.DataFrame) -> str:
    """Build Markdown consistency verdict."""
    fail_count = int((checks["status"] == "fail").sum())
    warning_count = int((checks["status"] == "warning").sum())
    verdict = (
        "The executive comparison is not yet reliable."
        if fail_count
        else "The executive comparison is usable with listed caveats."
    )
    lines = [
        "# Executive Results Consistency Audit",
        "",
        f"Final verdict: {verdict}",
        "",
        f"Checks: {len(checks)} total, {fail_count} fail, {warning_count} warning.",
        "",
        "## Issues",
        "",
    ]
    if issues.empty:
        lines.append("No warnings or failures were detected.")
    else:
        for _, row in issues.iterrows():
            lines.append(
                f"- {row['status'].upper()} `{row['check_id']}`: {row['details']}"
            )
    lines.extend(["", "## Checks", ""])
    for _, row in checks.iterrows():
        lines.append(f"- {row['status'].upper()} `{row['check_id']}`")
    lines.append("")
    return "\n".join(lines)


def _mandate_formula_mismatches(summary: pd.DataFrame) -> list[str]:
    mismatches = []
    for _, row in summary.iterrows():
        robust = _num(row["robust_score"])
        max_drawdown = _num(row["max_drawdown"])
        expected = robust * get_drawdown_multiplier(max_drawdown)
        if assign_drawdown_bucket(max_drawdown) == "not_eligible":
            expected = 0.0
        actual = _num(row["mandate_aware_score"])
        if not np.isclose(expected, actual, atol=1e-6, equal_nan=True):
            mismatches.append(
                f"{row['strategy_name']}: expected {expected:.6f}, got {actual:.6f}"
            )
    return mismatches


def _check(check_id: str, description: str, status: str, details: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "description": description,
        "status": status,
        "details": details,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing audit input file: {path}")
    return pd.read_csv(path)


def _num(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return np.nan
    return float(numeric)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit executive capped TD3 comparison consistency.",
    )
    parser.add_argument("--comparison-dir", default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--executive-dir", default=DEFAULT_EXECUTIVE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = audit_executive_results_consistency(
        comparison_dir=args.comparison_dir,
        executive_dir=args.executive_dir,
        output_dir=args.output_dir,
    )
    print(result["checks"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
