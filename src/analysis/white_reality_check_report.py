"""White Reality Check style report for final constrained TD3 candidates.

This module is reporting-only. It uses already-generated out-of-sample return
histories, date-averages TD3 fold/seed histories through the statistical
validation helpers, and applies a block-bootstrap White Reality Check style
multiple-model correction against chosen clean benchmarks.

It is not an SPA test and does not retrain models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.statistical_validation_report import (
    DEFAULT_BENCHMARKS,
    align_return_pair,
    compute_return_metrics,
    load_selected_candidates_and_metadata,
    locate_strategy_histories,
    _with_cap_sensitivity_overrides,
)


DEFAULT_FINAL_REPORT_DIR = (
    "outputs/tables/"
    "final_constrained_td3_report_with_v3_clean_no_dxy_v7_clean_garch_v4_v7_v8_60ep_10seeds"
)
DEFAULT_OUTPUT_DIR = "outputs/tables/white_reality_check_final"
MIN_OVERLAP = 30
WEEKLY_PERIODS_PER_YEAR = 52
RETURN_COLUMN_POLICY = "financial_net_return preferred, portfolio_return fallback"


def build_white_reality_check_report(
    final_report_dir: str = DEFAULT_FINAL_REPORT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    benchmarks: list[str] | None = None,
    n_bootstrap: int = 2000,
    block_length: int = 8,
    seed: int = 123,
    v3_cap_sensitivity_dir: str | None = None,
    v3_vintage_cap_sensitivity_dir: str | None = None,
    v3_clean_no_dxy_cap_sensitivity_dir: str | None = None,
    v4_cap_sensitivity_dir: str | None = None,
    v7_cap_sensitivity_dir: str | None = None,
    v7_clean_no_dxy_garch_cap_sensitivity_dir: str | None = None,
    v8_cap_sensitivity_dir: str | None = None,
    benchmark_dir: str | None = None,
    td3_history_dir: str | None = None,
    asset_specific_only: bool | None = None,
) -> dict[str, Any]:
    """Build White Reality Check CSVs and markdown from existing histories."""
    final_dir = Path(final_report_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected, metadata, report_mode = load_selected_candidates_and_metadata(final_dir)
    metadata = _with_cap_sensitivity_overrides(
        metadata,
        v3_cap_sensitivity_dir=v3_cap_sensitivity_dir,
        v3_vintage_cap_sensitivity_dir=v3_vintage_cap_sensitivity_dir,
        v3_clean_no_dxy_cap_sensitivity_dir=v3_clean_no_dxy_cap_sensitivity_dir,
        v4_cap_sensitivity_dir=v4_cap_sensitivity_dir,
        v7_cap_sensitivity_dir=v7_cap_sensitivity_dir,
        v7_clean_no_dxy_garch_cap_sensitivity_dir=(
            v7_clean_no_dxy_garch_cap_sensitivity_dir
        ),
        v8_cap_sensitivity_dir=v8_cap_sensitivity_dir,
    )
    if benchmark_dir is not None:
        metadata["benchmark_comparison_dir"] = str(Path(benchmark_dir))
    if td3_history_dir is not None:
        metadata["td3_history_dir"] = str(Path(td3_history_dir))
    require_asset_specific = (
        report_mode == "asset_specific" if asset_specific_only is None else asset_specific_only
    )
    histories, history_records, warnings = locate_strategy_histories(
        final_report_dir=final_dir,
        selected_candidates=selected,
        metadata=metadata,
        require_asset_specific=require_asset_specific,
    )
    benchmark_names = DEFAULT_BENCHMARKS if benchmarks is None else benchmarks
    candidate_names = [
        str(name)
        for name in selected["strategy_name"].dropna().astype(str)
        if str(name) in histories
    ]

    summary_rows: list[dict[str, Any]] = []
    differential_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    for offset, benchmark_name in enumerate(benchmark_names):
        result = run_white_reality_check_for_benchmark(
            histories=histories,
            candidate_names=candidate_names,
            benchmark_name=benchmark_name,
            n_bootstrap=n_bootstrap,
            block_length=block_length,
            seed=seed + offset,
            min_overlap=MIN_OVERLAP,
        )
        summary_rows.append(result["summary"])
        differential_frames.append(result["differentials"])
        bootstrap_frames.append(result["bootstrap_distribution"])
        warnings.extend(result["warnings"])

    summary = pd.DataFrame(summary_rows)
    differentials = (
        pd.concat(differential_frames, ignore_index=True)
        if differential_frames
        else pd.DataFrame(columns=_differential_columns())
    )
    bootstrap_distribution = (
        pd.concat(bootstrap_frames, ignore_index=True)
        if bootstrap_frames
        else pd.DataFrame(columns=["benchmark", "bootstrap_id", "boot_statistic"])
    )
    markdown = build_summary_markdown(summary, warnings)

    paths = {
        "summary": out_dir / "white_reality_check_summary.csv",
        "candidate_differentials": out_dir / "white_reality_check_candidate_differentials.csv",
        "bootstrap_distribution": out_dir / "white_reality_check_bootstrap_distribution.csv",
        "metadata": out_dir / "white_reality_check_metadata.json",
        "markdown": out_dir / "white_reality_check_summary.md",
    }
    summary.to_csv(paths["summary"], index=False)
    differentials.to_csv(paths["candidate_differentials"], index=False)
    bootstrap_distribution.to_csv(paths["bootstrap_distribution"], index=False)
    paths["markdown"].write_text(markdown, encoding="utf-8")

    metadata_out = {
        "final_report_dir": str(final_dir),
        "report_mode": report_mode,
        "asset_specific_only": require_asset_specific,
        "candidates_tested": candidate_names,
        "benchmarks_tested": benchmark_names,
        "history_dirs_used": history_records.to_dict(orient="records"),
        "return_column_used": RETURN_COLUMN_POLICY,
        "bootstrap_method": "centered block bootstrap over aligned differential matrix",
        "block_length": block_length,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "minimum_overlap_periods": MIN_OVERLAP,
        "caveats": [
            "White Reality Check style validation corrects for model search over the tested candidate set.",
            "This is not an SPA test.",
            "The test is based on mean weekly return differentials only.",
            "TD3 fold/seed histories are date-averaged before constructing differentials.",
            "High p-values mean no evidence that the best searched candidate is superior to the benchmark.",
        ],
        "missing_history_warnings": sorted(set(warnings)),
    }
    paths["metadata"].write_text(json.dumps(_json_safe(metadata_out), indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "candidate_differentials": differentials,
        "bootstrap_distribution": bootstrap_distribution,
        "history_records": history_records,
        "warnings": sorted(set(warnings)),
        "markdown": markdown,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def run_white_reality_check_for_benchmark(
    histories: dict[str, pd.Series],
    candidate_names: list[str],
    benchmark_name: str,
    n_bootstrap: int = 2000,
    block_length: int = 8,
    seed: int = 123,
    min_overlap: int = MIN_OVERLAP,
) -> dict[str, Any]:
    """Run WRC style max-mean-differential test for one benchmark."""
    warnings: list[str] = []
    if benchmark_name not in histories:
        warnings.append(f"Missing benchmark history for {benchmark_name}.")
        return {
            "summary": _empty_summary(benchmark_name, n_bootstrap, block_length, "missing_benchmark_history"),
            "differentials": pd.DataFrame(columns=_differential_columns()),
            "bootstrap_distribution": pd.DataFrame(columns=["benchmark", "bootstrap_id", "boot_statistic"]),
            "warnings": warnings,
        }

    benchmark = histories[benchmark_name]
    differential_rows: list[dict[str, Any]] = []
    diff_series: dict[str, pd.Series] = {}
    for candidate_name in candidate_names:
        candidate = histories.get(candidate_name)
        if candidate is None:
            warnings.append(f"Missing candidate history for {candidate_name}.")
            continue
        aligned = align_return_pair(candidate, benchmark)
        if len(aligned) < min_overlap:
            warnings.append(
                f"Insufficient overlap for {candidate_name} vs {benchmark_name}: {len(aligned)} periods."
            )
            continue
        diff = aligned["candidate"] - aligned["benchmark"]
        diff_series[candidate_name] = diff
        differential_rows.append(
            build_candidate_differential_row(
                benchmark_name=benchmark_name,
                candidate_name=candidate_name,
                aligned=aligned,
            )
        )

    differentials = pd.DataFrame(differential_rows, columns=_differential_columns())
    if not diff_series:
        return {
            "summary": _empty_summary(benchmark_name, n_bootstrap, block_length, "no_valid_candidates"),
            "differentials": differentials,
            "bootstrap_distribution": pd.DataFrame(columns=["benchmark", "bootstrap_id", "boot_statistic"]),
            "warnings": warnings or [f"No valid candidate histories for {benchmark_name}."],
        }

    diff_matrix = align_differential_matrix(diff_series)
    if len(diff_matrix) < min_overlap:
        warnings.append(
            f"Common differential matrix for {benchmark_name} has only {len(diff_matrix)} periods."
        )
        return {
            "summary": _empty_summary(benchmark_name, n_bootstrap, block_length, "insufficient_common_overlap"),
            "differentials": differentials,
            "bootstrap_distribution": pd.DataFrame(columns=["benchmark", "bootstrap_id", "boot_statistic"]),
            "warnings": warnings,
        }

    wrc = white_reality_check_statistic(
        diff_matrix,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        seed=seed,
    )
    mean_diffs = diff_matrix.mean(axis=0)
    best_candidate = str(mean_diffs.idxmax())
    best_mean_diff = float(mean_diffs.max())
    interpretation = interpret_p_value(wrc["p_value"])
    summary = {
        "benchmark": benchmark_name,
        "n_candidates": diff_matrix.shape[1],
        "n_periods": diff_matrix.shape[0],
        "best_candidate_by_mean_diff": best_candidate,
        "best_mean_diff": best_mean_diff,
        "observed_statistic": wrc["observed_statistic"],
        "p_value": wrc["p_value"],
        "block_length": block_length,
        "n_bootstrap": n_bootstrap,
        "interpretation": interpretation,
    }
    bootstrap_distribution = pd.DataFrame(
        {
            "benchmark": benchmark_name,
            "bootstrap_id": np.arange(n_bootstrap, dtype=int),
            "boot_statistic": wrc["bootstrap_statistics"],
        }
    )
    return {
        "summary": summary,
        "differentials": differentials,
        "bootstrap_distribution": bootstrap_distribution,
        "warnings": warnings,
    }


def white_reality_check_statistic(
    differential_matrix: pd.DataFrame,
    n_bootstrap: int = 2000,
    block_length: int = 8,
    seed: int = 123,
) -> dict[str, Any]:
    """Compute centered block-bootstrap WRC statistic."""
    if differential_matrix.empty:
        raise ValueError("differential_matrix must not be empty.")
    values = differential_matrix.to_numpy(dtype=float)
    n_periods = values.shape[0]
    means = values.mean(axis=0)
    observed = float(np.sqrt(n_periods) * np.max(means))
    centered = values - means
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap, dtype=float)
    for bootstrap_id in range(n_bootstrap):
        boot = sample_blocks(centered, block_length=block_length, rng=rng)
        boot_stats[bootstrap_id] = float(np.sqrt(n_periods) * np.max(boot.mean(axis=0)))
    p_value = float((1 + np.sum(boot_stats >= observed)) / (n_bootstrap + 1))
    return {
        "observed_statistic": observed,
        "bootstrap_statistics": boot_stats,
        "p_value": p_value,
    }


def build_candidate_differential_row(
    benchmark_name: str,
    candidate_name: str,
    aligned: pd.DataFrame,
) -> dict[str, Any]:
    """Build one candidate-vs-benchmark differential row."""
    diff = aligned["candidate"] - aligned["benchmark"]
    candidate_metrics = compute_return_metrics(aligned["candidate"])
    benchmark_metrics = compute_return_metrics(aligned["benchmark"])
    return {
        "benchmark": benchmark_name,
        "candidate": candidate_name,
        "n_periods": len(aligned),
        "mean_return_diff": float(diff.mean()),
        "annualized_mean_diff": float(diff.mean() * WEEKLY_PERIODS_PER_YEAR),
        "win_rate": float((diff > 0.0).mean()),
        "candidate_mean_return": float(aligned["candidate"].mean()),
        "benchmark_mean_return": float(aligned["benchmark"].mean()),
        "candidate_sharpe": candidate_metrics["sharpe"],
        "benchmark_sharpe": benchmark_metrics["sharpe"],
        "sharpe_diff": candidate_metrics["sharpe"] - benchmark_metrics["sharpe"],
    }


def align_differential_matrix(diff_series: dict[str, pd.Series]) -> pd.DataFrame:
    """Align candidate differential series on common dates."""
    frame = pd.concat(
        [series.rename(candidate) for candidate, series in diff_series.items()],
        axis=1,
        join="inner",
    )
    return frame.replace([np.inf, -np.inf], np.nan).dropna().sort_index()


def sample_blocks(
    values: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample rows using non-overlapping length-preserving moving blocks."""
    if len(values) == 0:
        return values
    block = max(1, min(int(block_length), len(values)))
    pieces = []
    total = 0
    while total < len(values):
        start = int(rng.integers(0, len(values) - block + 1))
        piece = values[start : start + block]
        pieces.append(piece)
        total += len(piece)
    return np.concatenate(pieces, axis=0)[: len(values)]


def interpret_p_value(p_value: float) -> str:
    """Conservative publication-safe p-value interpretation."""
    if pd.isna(p_value):
        return "not_available"
    if p_value <= 0.05:
        return "evidence_against_null_at_5pct"
    if p_value <= 0.10:
        return "weak_evidence_against_null_at_10pct"
    return "no_evidence_best_candidate_superior_after_wrc"


def build_summary_markdown(summary: pd.DataFrame, warnings: list[str]) -> str:
    """Create Markdown summary."""
    lines = [
        "# White Reality Check Report",
        "",
        "This is a White Reality Check style multiple-model data-snooping correction. It is reporting-only and does not retrain models.",
        "",
        "The test asks whether the best candidate among the evaluated TD3/constrained strategy set outperforms a benchmark after accounting for model search.",
        "",
        "It uses aligned realized out-of-sample weekly net returns. TD3 fold/seed histories are date-averaged before the test, consistent with the existing statistical validation layer.",
        "",
        "A high p-value means there is no evidence that the best searched candidate is superior to the benchmark. This does not invalidate mandate-aware stabilization results.",
        "",
        "The test addresses mean return differentials only. It is not an SPA test and does not imply market dominance.",
        "",
        "## Results",
        "",
    ]
    if summary.empty:
        lines.append("No valid White Reality Check results were produced.")
    else:
        for _, row in summary.iterrows():
            lines.append(
                f"- `{row['benchmark']}`: best candidate `{row['best_candidate_by_mean_diff']}`, "
                f"mean diff {_fmt(row['best_mean_diff'])}, p-value {_fmt(row['p_value'])}, "
                f"interpretation `{row['interpretation']}`."
            )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in sorted(set(warnings)):
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _empty_summary(
    benchmark_name: str,
    n_bootstrap: int,
    block_length: int,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark_name,
        "n_candidates": 0,
        "n_periods": 0,
        "best_candidate_by_mean_diff": "",
        "best_mean_diff": np.nan,
        "observed_statistic": np.nan,
        "p_value": np.nan,
        "block_length": block_length,
        "n_bootstrap": n_bootstrap,
        "interpretation": interpretation,
    }


def _differential_columns() -> list[str]:
    return [
        "benchmark",
        "candidate",
        "n_periods",
        "mean_return_diff",
        "annualized_mean_diff",
        "win_rate",
        "candidate_mean_return",
        "benchmark_mean_return",
        "candidate_sharpe",
        "benchmark_sharpe",
        "sharpe_diff",
    ]


def _parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _fmt(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.4f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if not isinstance(value, (str, bool, dict, list)) and pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build White Reality Check style report from final constrained TD3 histories.",
    )
    parser.add_argument("--final-report-dir", default=DEFAULT_FINAL_REPORT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--v3-cap-sensitivity-dir", default=None)
    parser.add_argument("--v3-vintage-cap-sensitivity-dir", default=None)
    parser.add_argument("--v3-clean-no-dxy-cap-sensitivity-dir", default=None)
    parser.add_argument("--v4-cap-sensitivity-dir", default=None)
    parser.add_argument("--v7-cap-sensitivity-dir", default=None)
    parser.add_argument("--v7-clean-no-dxy-garch-cap-sensitivity-dir", default=None)
    parser.add_argument("--v8-cap-sensitivity-dir", default=None)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--td3-history-dir", default=None)
    parser.add_argument("--asset-specific-only", action="store_true")
    args = parser.parse_args()

    result = build_white_reality_check_report(
        final_report_dir=args.final_report_dir,
        output_dir=args.output_dir,
        benchmarks=_parse_csv_list(args.benchmarks, DEFAULT_BENCHMARKS),
        n_bootstrap=args.n_bootstrap,
        block_length=args.block_length,
        seed=args.seed,
        v3_cap_sensitivity_dir=args.v3_cap_sensitivity_dir,
        v3_vintage_cap_sensitivity_dir=args.v3_vintage_cap_sensitivity_dir,
        v3_clean_no_dxy_cap_sensitivity_dir=args.v3_clean_no_dxy_cap_sensitivity_dir,
        v4_cap_sensitivity_dir=args.v4_cap_sensitivity_dir,
        v7_cap_sensitivity_dir=args.v7_cap_sensitivity_dir,
        v7_clean_no_dxy_garch_cap_sensitivity_dir=(
            args.v7_clean_no_dxy_garch_cap_sensitivity_dir
        ),
        v8_cap_sensitivity_dir=args.v8_cap_sensitivity_dir,
        benchmark_dir=args.benchmark_dir,
        td3_history_dir=args.td3_history_dir,
        asset_specific_only=args.asset_specific_only or None,
    )

    print("White Reality Check summary:")
    print(result["summary"].to_string(index=False))
    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
