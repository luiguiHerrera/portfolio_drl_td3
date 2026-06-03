"""Recover final corrected limited TD3 reports from completed histories.

This script is reporting-only. It reads completed per-run histories and
per-candidate aggregates, then rebuilds missing robust/cap-sensitivity summary
files without invoking TD3 training.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.robust_score import (
    POOLED_DSR_WARNING,
    _rename_report_columns,
    compute_composite_robust_score,
)


DEFAULT_OUTPUT_DIR = "outputs/tables/final_corrected_limited_td3_60ep_10seeds"
EXPECTED_HISTORIES = 800
EXPECTED_EPISODES = 60
CAP_RE = re.compile(r"_cap_(?P<cap>uncapped|[0-9]+p[0-9]+)_seed_")
ORIGINAL_PANDAS_READ_CSV = pd.read_csv
DELTA_METRICS = [
    "annualized_return",
    "sharpe",
    "robust_score",
    "mandate_aware_score",
    "max_drawdown",
    "average_turnover",
    "average_effective_number_of_assets",
    "average_max_weight",
]


def read_csv_with_retry(
    path: str | Path,
    retries: int = 3,
    sleep_seconds: float = 1.0,
    timeout_seconds: int = 15,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV with retries for transient Desktop/iCloud I/O timeouts."""
    csv_path = Path(path)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _read_csv_with_timeout(csv_path, timeout_seconds, **kwargs)
        except (TimeoutError, OSError, pd.errors.ParserError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(sleep_seconds)
    raise RuntimeError(
        f"Failed to read required CSV after {retries + 1} attempts: {csv_path}"
    ) from last_error


def _read_csv_with_timeout(
    path: Path,
    timeout_seconds: int,
    **kwargs: Any,
) -> pd.DataFrame:
    if timeout_seconds <= 0:
        return ORIGINAL_PANDAS_READ_CSV(path, **kwargs)

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _timeout_handler(_signum, _frame):
        raise TimeoutError(f"Timed out reading CSV: {path}")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return ORIGINAL_PANDAS_READ_CSV(path, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def recover_reports(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    expected_histories: int = EXPECTED_HISTORIES,
    episodes: int = EXPECTED_EPISODES,
    retries: int = 3,
    sleep_seconds: float = 1.0,
) -> dict[str, Any]:
    root = Path(output_dir)
    per_candidate_dir = root / "per_candidate"
    if not per_candidate_dir.exists():
        raise FileNotFoundError(f"Missing per_candidate directory: {per_candidate_dir}")

    global_expected = [
        "cap_sensitivity_all_results.csv",
        "cap_sensitivity_pairwise_deltas.csv",
        "cap_sensitivity_best_caps.csv",
        "cap_sensitivity_summary.csv",
        "cap_sensitivity_summary.md",
        "cap_sensitivity_metadata.json",
    ]
    missing_global_before = [
        name for name in global_expected if not (root / name).exists()
    ]

    candidates = [
        candidate_dir
        for candidate_dir in sorted(per_candidate_dir.iterdir())
        if candidate_dir.is_dir()
    ]
    if not candidates:
        raise ValueError(f"No candidate directories found under {per_candidate_dir}")

    history_paths = _history_paths_for_candidates(candidates)
    print(f"Found {len(history_paths)} test histories.", flush=True)
    if len(history_paths) != expected_histories:
        raise ValueError(
            f"Expected {expected_histories} test histories, found {len(history_paths)}."
        )

    read_failures: list[str] = []
    regenerated_files: list[str] = []
    all_ranking_frames: list[pd.DataFrame] = []
    reports: dict[str, dict[str, Any]] = {}

    for candidate_dir in candidates:
        candidate = candidate_dir.name
        print(f"Processing candidate: {candidate}", flush=True)
        candidate_metadata = _candidate_metadata(candidate_dir)
        missing_candidate_files = _missing_candidate_report_files(candidate_dir)
        missing_robust_files = [
            name for name in missing_candidate_files if name.startswith("robust_score_")
        ]
        needs_cap_rebuild = (
            any(name.startswith("max_weight_cap_") for name in missing_candidate_files)
            or _is_recovered_cap_metadata(candidate_dir / "max_weight_cap_metadata.json")
        )
        if missing_robust_files or needs_cap_rebuild:
            print(
                "  Rebuilding report files: "
                + ", ".join(missing_candidate_files or ["recovered cap summaries"]),
                flush=True,
            )
            if missing_robust_files:
                robust_report = build_fast_recovered_robust_score_report(
                    comparison_dir=candidate_dir,
                    split="test",
                    retries=retries,
                    sleep_seconds=sleep_seconds,
                )
                regenerated_files.extend(
                    [
                        robust_report["ranking_path"],
                        robust_report["component_details_path"],
                        robust_report["warnings_path"],
                    ]
                )

            seed_results = read_csv_with_retry(
                candidate_dir / "seed_fold_strategy_results.csv",
                retries=retries,
                sleep_seconds=sleep_seconds,
            )
            overall = read_csv_with_retry(
                candidate_dir / "overall_aggregate_by_strategy_split.csv",
                retries=retries,
                sleep_seconds=sleep_seconds,
            )
            robust_ranking = read_csv_with_retry(
                candidate_dir / "robust_score_ranking.csv",
                retries=retries,
                sleep_seconds=sleep_seconds,
            )
            cap_metrics = build_max_weight_cap_metrics(
                seed_results=seed_results,
                candidates=candidate_metadata,
            )
            cap_summary = build_max_weight_cap_summary(
                overall=overall,
                robust_ranking=robust_ranking,
                candidates=candidate_metadata,
                episodes=episodes,
            )
            cap_rankings = build_max_weight_cap_rankings(cap_summary)
            cap_metrics_path = candidate_dir / "max_weight_cap_metrics.csv"
            cap_summary_path = candidate_dir / "max_weight_cap_summary.csv"
            cap_rankings_path = candidate_dir / "max_weight_cap_rankings.csv"
            cap_metadata_path = candidate_dir / "max_weight_cap_metadata.json"
            cap_metrics.to_csv(cap_metrics_path, index=False)
            cap_summary.to_csv(cap_summary_path, index=False)
            cap_rankings.to_csv(cap_rankings_path, index=False)
            cap_metadata_path.write_text(
                json.dumps(
                    _recovery_candidate_metadata(
                        candidate=candidate,
                        candidate_dir=candidate_dir,
                        histories_found=len(_history_paths_for_candidate(candidate_dir)),
                        episodes=episodes,
                        missing_files_before=missing_candidate_files,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            regenerated_files.extend(
                [
                    str(cap_metrics_path),
                    str(cap_summary_path),
                    str(cap_rankings_path),
                    str(cap_metadata_path),
                ]
            )

        rankings_path = candidate_dir / "max_weight_cap_rankings.csv"
        print(f"  Loading rankings: {rankings_path}", flush=True)
        if not rankings_path.exists():
            raise FileNotFoundError(f"Missing candidate rankings: {rankings_path}")
        rankings = read_csv_with_retry(
            rankings_path,
            retries=retries,
            sleep_seconds=sleep_seconds,
        )
        rankings["candidate_output_dir"] = str(candidate_dir)
        all_ranking_frames.append(rankings)
        reports[candidate] = {
            "paths": {
                "output_dir": str(candidate_dir),
                "rankings": str(rankings_path),
            }
        }

    print("Building global cap sensitivity tables.", flush=True)
    all_results = build_cap_sensitivity_all_results(all_ranking_frames)
    pairwise_deltas = build_cap_sensitivity_pairwise_deltas(all_results)
    best_caps = build_cap_sensitivity_best_caps(all_results)
    summary = build_cap_sensitivity_summary(all_results, best_caps)
    markdown = build_cap_sensitivity_markdown(summary, best_caps)
    metadata = build_cap_sensitivity_metadata(
        returns_path="recovered_from_existing_histories",
        output_dir=str(root),
        candidates=[path.name for path in candidates],
        max_weight_grid=_cap_grid_from_results(all_results),
        episodes=episodes,
        seeds=_seeds_from_history_paths(history_paths),
        max_folds=None,
        transaction_cost=float("nan"),
        base_config_path=str(root / "final_corrected_config.yaml"),
        reports=reports,
    )
    metadata.update(
        {
            "recovery_script": "scripts/recover_final_corrected_limited_reports.py",
            "reporting_only": True,
            "td3_training_called": False,
            "expected_histories": expected_histories,
            "found_histories": len(history_paths),
            "bad_reads": len(read_failures),
            "missing_global_files_before": missing_global_before,
            "score_scope": "recovered_final_corrected_limited_universe",
            "csv_read_retries": retries,
            "csv_read_sleep_seconds": sleep_seconds,
        }
    )
    paths = write_cap_sensitivity_outputs(
        output_dir=root,
        all_results=all_results,
        pairwise_deltas=pairwise_deltas,
        best_caps=best_caps,
        summary=summary,
        markdown=markdown,
        metadata=metadata,
    )
    regenerated_files.extend(str(path) for path in paths.values())

    return {
        "missing_global_before": missing_global_before,
        "regenerated_files": sorted(set(regenerated_files)),
        "histories_found": len(history_paths),
        "bad_reads": len(read_failures),
        "all_results": all_results,
        "best_caps": best_caps,
        "summary": summary,
        "paths": paths,
    }


def _patch_robust_score_csv_reader(retries: int, sleep_seconds: float) -> None:
    import src.analysis.robust_score as robust_score_module

    def retrying_read_csv(path, *args, **kwargs):
        return read_csv_with_retry(
            path,
            retries=retries,
            sleep_seconds=sleep_seconds,
            **kwargs,
        )

    pd.read_csv = retrying_read_csv
    robust_score_module.pd.read_csv = retrying_read_csv


def build_fast_recovered_robust_score_report(
    comparison_dir: Path,
    split: str,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    """Build robust score files from aggregate metrics without history DSR reads."""
    metrics_path = comparison_dir / "overall_aggregate_by_strategy_split.csv"
    metrics = read_csv_with_retry(
        metrics_path,
        retries=retries,
        sleep_seconds=sleep_seconds,
    )
    metrics = metrics.loc[metrics["split"] == split].copy()
    if metrics.empty:
        raise ValueError(f"No aggregate metrics rows found for split: {split}")

    metrics = _rename_report_columns(metrics)
    metrics["unjustified_cash_excess"] = metrics.get("unjustified_cash_excess", pd.NA)
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
    for column in ranking_columns:
        if column not in scored:
            scored[column] = pd.NA
    ranking = scored.sort_values("robust_score", ascending=False).reset_index(drop=True)
    ranking_path = comparison_dir / "robust_score_ranking.csv"
    details_path = comparison_dir / "robust_score_component_details.csv"
    warnings_path = comparison_dir / "robust_score_warnings.txt"
    ranking.loc[:, ranking_columns].to_csv(ranking_path, index=False)
    scored.to_csv(details_path, index=False)
    warnings = "\n".join(
        [
            POOLED_DSR_WARNING,
            "Recovery note: DSR observations were not reloaded from histories; "
            "compute_composite_robust_score used its documented fallback-from-Sharpe "
            "path for missing DSR columns.",
        ]
    )
    warnings_path.write_text(warnings + "\n", encoding="utf-8")
    return {
        "ranking": ranking.loc[:, ranking_columns],
        "component_details": scored,
        "warnings": warnings,
        "ranking_path": str(ranking_path),
        "component_details_path": str(details_path),
        "warnings_path": str(warnings_path),
    }


def build_max_weight_cap_metrics(
    seed_results: pd.DataFrame,
    candidates: list[dict],
) -> pd.DataFrame:
    metadata = _candidate_metadata_frame(candidates)
    metrics = seed_results.rename(columns={"strategy": "candidate_name"}).copy()
    return metrics.merge(metadata, on="candidate_name", how="left")


def build_max_weight_cap_summary(
    overall: pd.DataFrame,
    robust_ranking: pd.DataFrame,
    candidates: list[dict],
    episodes: int,
) -> pd.DataFrame:
    metadata = _candidate_metadata_frame(candidates)
    summary = overall.rename(columns={"strategy": "candidate_name"}).copy()
    summary = summary.merge(metadata, on="candidate_name", how="left")
    summary["episodes"] = episodes
    summary = summary.rename(
        columns={
            "mean_sharpe": "sharpe",
            "mean_sortino": "sortino",
            "mean_calmar": "calmar",
            "mean_cumulative_return": "cumulative_return",
            "mean_annualized_return": "annualized_return",
            "mean_annualized_volatility": "annualized_volatility",
            "mean_max_drawdown": "max_drawdown",
            "mean_average_turnover": "average_turnover",
            "mean_average_effective_number_of_assets": (
                "average_effective_number_of_assets"
            ),
            "mean_average_max_weight": "average_max_weight",
        }
    )
    robust = robust_ranking.rename(columns={"strategy": "candidate_name"}).copy()
    robust_keep = [
        column
        for column in [
            "candidate_name",
            "robust_score",
            "median_run_dsr_n25",
            "date_averaged_dsr_n25",
            "dsr_method",
        ]
        if column in robust.columns
    ]
    summary = summary.merge(robust.loc[:, robust_keep], on="candidate_name", how="left")
    summary["mandate_bucket"] = summary["max_drawdown"].apply(_drawdown_bucket)
    summary["drawdown_multiplier"] = summary["max_drawdown"].apply(_drawdown_multiplier)
    summary["mandate_aware_score"] = (
        pd.to_numeric(summary["robust_score"], errors="coerce")
        * pd.to_numeric(summary["drawdown_multiplier"], errors="coerce")
    )
    summary.loc[summary["mandate_bucket"] == "not_eligible", "mandate_aware_score"] = 0.0
    summary["concentration_classification"] = summary.apply(
        _concentration_classification,
        axis=1,
    )
    summary["suspicious_or_lazy_concentration_candidate"] = (
        summary["concentration_classification"] == "learned_extreme_concentration"
    )
    summary["justified_concentration_candidate"] = False
    return _select_summary_columns(summary)


def build_max_weight_cap_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    test = summary.loc[summary["split"].astype(str) == "test"].copy()
    if test.empty:
        return pd.DataFrame()
    baseline = test.loc[test["max_weight_cap"].isna()]
    if baseline.empty:
        raise ValueError("Uncapped baseline is required.")
    baseline_row = baseline.iloc[0]
    for metric in [
        "average_effective_number_of_assets",
        "average_max_weight",
        "average_turnover",
        "sharpe",
        "robust_score",
        "mandate_aware_score",
        "max_drawdown",
        "annualized_return",
    ]:
        baseline_value = _numeric(baseline_row[metric])
        test[f"delta_{metric}_vs_baseline"] = (
            pd.to_numeric(test[metric], errors="coerce") - baseline_value
        )
    test["decision_label"] = test.apply(label_cap_sensitivity_decision, axis=1)
    return test.sort_values(
        ["mandate_aware_score", "robust_score"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_cap_sensitivity_all_results(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["cap_label"] = combined["max_weight_cap"].apply(format_cap_label)
    combined["decision_label"] = combined.apply(label_cap_sensitivity_decision, axis=1)
    return combined.sort_values(
        ["base_candidate", "max_weight_cap"],
        na_position="first",
    ).reset_index(drop=True)


def build_cap_sensitivity_pairwise_deltas(all_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for base_candidate, group in all_results.groupby("base_candidate", dropna=True):
        baseline_rows = group[group["max_weight_cap"].isna()]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        for _, row in group.iterrows():
            result = {
                "base_candidate": base_candidate,
                "candidate_name": row["candidate_name"],
                "max_weight_cap": row["max_weight_cap"],
                "cap_label": row["cap_label"],
                "decision_label": row["decision_label"],
            }
            for metric in DELTA_METRICS:
                result[f"uncapped_{metric}"] = baseline.get(metric)
                result[f"capped_{metric}"] = row.get(metric)
                result[f"delta_{metric}"] = _numeric(row.get(metric)) - _numeric(
                    baseline.get(metric)
                )
            rows.append(result)
    return pd.DataFrame(rows)


def build_cap_sensitivity_best_caps(all_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for base_candidate, group in all_results.groupby("base_candidate", dropna=True):
        rows.append(
            {
                "base_candidate": base_candidate,
                "best_by_mandate_aware_score": _best_cap_label(
                    group,
                    "mandate_aware_score",
                    ascending=False,
                ),
                "best_mandate_aware_score": _best_metric_value(
                    group,
                    "mandate_aware_score",
                    ascending=False,
                ),
                "best_by_robust_score": _best_cap_label(
                    group,
                    "robust_score",
                    ascending=False,
                ),
                "best_robust_score": _best_metric_value(
                    group,
                    "robust_score",
                    ascending=False,
                ),
                "best_by_max_drawdown": _best_cap_label(
                    group,
                    "max_drawdown",
                    ascending=False,
                ),
                "best_max_drawdown": _best_metric_value(
                    group,
                    "max_drawdown",
                    ascending=False,
                ),
                "best_by_turnover": _best_cap_label(
                    group,
                    "average_turnover",
                    ascending=True,
                ),
                "best_turnover": _best_metric_value(
                    group,
                    "average_turnover",
                    ascending=True,
                ),
                "best_by_effective_assets": _best_cap_label(
                    group,
                    "average_effective_number_of_assets",
                    ascending=False,
                ),
                "best_effective_assets": _best_metric_value(
                    group,
                    "average_effective_number_of_assets",
                    ascending=False,
                ),
                "overall_interpretation": _interpret_candidate(group),
            }
        )
    return pd.DataFrame(rows)


def build_cap_sensitivity_summary(
    all_results: pd.DataFrame,
    best_caps: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    best_lookup = best_caps.set_index("base_candidate")
    for base_candidate, group in all_results.groupby("base_candidate", dropna=True):
        baseline_rows = group[group["max_weight_cap"].isna()]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        best = best_lookup.loc[base_candidate]
        capped = group[group["max_weight_cap"].notna()]
        rows.append(
            {
                "base_candidate": base_candidate,
                "uncapped_mandate_aware_score": baseline["mandate_aware_score"],
                "best_cap_by_mandate_aware_score": best[
                    "best_by_mandate_aware_score"
                ],
                "best_cap_mandate_aware_score": best["best_mandate_aware_score"],
                "best_cap_by_robust_score": best["best_by_robust_score"],
                "best_cap_robust_score": best["best_robust_score"],
                "n_caps_improve_mandate": int(
                    (
                        pd.to_numeric(capped["mandate_aware_score"], errors="coerce")
                        > _numeric(baseline["mandate_aware_score"])
                    ).sum()
                ),
                "interpretation": _interpret_candidate(group),
            }
        )
    return pd.DataFrame(rows)


def build_cap_sensitivity_markdown(
    summary: pd.DataFrame,
    best_caps: pd.DataFrame,
) -> str:
    lines = [
        "# Recovered Final Corrected Limited TD3 Cap Sensitivity",
        "",
        "This report was rebuilt from completed per-run histories without TD3 retraining.",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Best Caps",
        "",
        _markdown_table(best_caps),
        "",
    ]
    return "\n".join(lines)


def build_cap_sensitivity_metadata(**kwargs: Any) -> dict[str, Any]:
    return {
        "runner": "scripts/recover_final_corrected_limited_reports.py",
        **kwargs,
    }


def write_cap_sensitivity_outputs(
    output_dir: Path,
    all_results: pd.DataFrame,
    pairwise_deltas: pd.DataFrame,
    best_caps: pd.DataFrame,
    summary: pd.DataFrame,
    markdown: str,
    metadata: dict[str, Any],
) -> dict[str, str]:
    paths = {
        "all_results": output_dir / "cap_sensitivity_all_results.csv",
        "pairwise_deltas": output_dir / "cap_sensitivity_pairwise_deltas.csv",
        "best_caps": output_dir / "cap_sensitivity_best_caps.csv",
        "summary": output_dir / "cap_sensitivity_summary.csv",
        "markdown_summary": output_dir / "cap_sensitivity_summary.md",
        "metadata": output_dir / "cap_sensitivity_metadata.json",
    }
    all_results.to_csv(paths["all_results"], index=False)
    pairwise_deltas.to_csv(paths["pairwise_deltas"], index=False)
    best_caps.to_csv(paths["best_caps"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["markdown_summary"].write_text(markdown, encoding="utf-8")
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


def _missing_candidate_report_files(candidate_dir: Path) -> list[str]:
    expected = [
        "robust_score_ranking.csv",
        "robust_score_component_details.csv",
        "robust_score_warnings.txt",
        "max_weight_cap_metrics.csv",
        "max_weight_cap_summary.csv",
        "max_weight_cap_rankings.csv",
        "max_weight_cap_metadata.json",
    ]
    return [name for name in expected if not (candidate_dir / name).exists()]


def _is_recovered_cap_metadata(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("runner") == "scripts/recover_final_corrected_limited_reports.py"


def _candidate_metadata_frame(candidates: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_name": candidate["name"],
                "base_candidate": candidate["base_candidate"],
                "max_weight_cap": candidate["max_weight_cap"],
            }
            for candidate in candidates
        ]
    )


def _select_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_name",
        "base_candidate",
        "max_weight_cap",
        "split",
        "n_folds",
        "n_seeds",
        "episodes",
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
        "concentration_classification",
        "suspicious_or_lazy_concentration_candidate",
        "justified_concentration_candidate",
    ]
    result = summary.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, columns]


def _drawdown_bucket(max_drawdown: Any) -> str:
    drawdown = _numeric(max_drawdown)
    if pd.isna(drawdown):
        return "unknown"
    if drawdown >= -0.15:
        return "mandate_eligible"
    if drawdown >= -0.25:
        return "borderline"
    return "not_eligible"


def _drawdown_multiplier(max_drawdown: Any) -> float:
    bucket = _drawdown_bucket(max_drawdown)
    if bucket == "mandate_eligible":
        return 1.0
    if bucket == "borderline":
        return 0.5
    if bucket == "unknown":
        return 0.0
    return 0.0


def _concentration_classification(row: pd.Series) -> str:
    max_weight = _numeric(row.get("average_max_weight"))
    effective_assets = _numeric(row.get("average_effective_number_of_assets"))
    if max_weight >= 0.85 or effective_assets < 1.5:
        return "learned_extreme_concentration"
    return "not_concentrated"


def label_cap_sensitivity_decision(row: pd.Series) -> str:
    cap = row.get("max_weight_cap")
    if pd.isna(cap):
        return "baseline"
    mandate_delta = _numeric(row.get("delta_mandate_aware_score_vs_baseline"))
    robust_delta = _numeric(row.get("delta_robust_score_vs_baseline"))
    effective_delta = _numeric(
        row.get("delta_average_effective_number_of_assets_vs_baseline")
    )
    drawdown_delta = _numeric(row.get("delta_max_drawdown_vs_baseline"))
    if (
        mandate_delta > 0.0
        and robust_delta >= -0.05
        and effective_delta > 0.20
        and drawdown_delta >= -0.02
    ):
        return "dominates_baseline"
    if mandate_delta > 0.0:
        return "improves_mandate_score"
    if effective_delta > 0.20 and robust_delta >= -0.05:
        return "improves_diversification"
    return "not_selected"


def format_cap_label(value: Any) -> str:
    if pd.isna(value):
        return "uncapped"
    return f"{float(value):.2f}"


def _best_cap_label(group: pd.DataFrame, metric: str, ascending: bool) -> str:
    return str(_best_row(group, metric, ascending)["cap_label"])


def _best_metric_value(group: pd.DataFrame, metric: str, ascending: bool) -> float:
    return _numeric(_best_row(group, metric, ascending)[metric])


def _best_row(group: pd.DataFrame, metric: str, ascending: bool) -> pd.Series:
    numeric = pd.to_numeric(group[metric], errors="coerce")
    ordered = group.assign(_metric=numeric).sort_values(
        "_metric",
        ascending=ascending,
        na_position="last",
    )
    return ordered.iloc[0]


def _interpret_candidate(group: pd.DataFrame) -> str:
    capped = group[group["max_weight_cap"].notna()]
    if capped.empty:
        return "No capped variants available."
    improved = (
        pd.to_numeric(capped["delta_mandate_aware_score_vs_baseline"], errors="coerce")
        > 0.0
    ).sum()
    if improved:
        return "At least one cap improves mandate-aware score versus uncapped."
    return "Caps do not improve mandate-aware score versus uncapped."


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    text_frame = frame.fillna("").astype(str)
    columns = list(text_frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in text_frame.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _numeric(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def _history_paths_for_candidates(candidate_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for candidate_dir in candidate_dirs:
        paths.extend(_history_paths_for_candidate(candidate_dir))
    return sorted(paths)


def _history_paths_for_candidate(candidate_dir: Path) -> list[Path]:
    paths = []
    for run_dir in sorted(candidate_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        history_path = run_dir / "test_policy_history.csv"
        if history_path.exists():
            paths.append(history_path)
    return paths


def _candidate_metadata(candidate_dir: Path) -> list[dict[str, Any]]:
    strategies = read_csv_with_retry(
        candidate_dir / "overall_aggregate_by_strategy_split.csv",
    )["strategy"].dropna().astype(str)
    rows = []
    for candidate_name in sorted(strategies.unique()):
        rows.append(
            {
                "name": candidate_name,
                "base_candidate": candidate_dir.name,
                "max_weight_cap": _parse_cap_from_candidate_name(candidate_name),
            }
        )
    return rows


def _parse_cap_from_candidate_name(candidate_name: str) -> float | None:
    match = re.search(r"_cap_(?P<cap>uncapped|[0-9]+p[0-9]+)$", candidate_name)
    if not match:
        raise ValueError(f"Could not parse cap from candidate name: {candidate_name}")
    label = match.group("cap")
    if label == "uncapped":
        return None
    return float(label.replace("p", "."))


def _cap_grid_from_results(all_results: pd.DataFrame) -> list[float | None]:
    caps = []
    for value in all_results["max_weight_cap"].drop_duplicates().tolist():
        if pd.isna(value):
            caps.append(None)
        else:
            caps.append(float(value))
    return caps


def _seeds_from_history_paths(paths: list[Path]) -> list[int]:
    seeds = set()
    for path in paths:
        match = re.search(r"_seed_(?P<seed>[0-9]+)$", path.parent.name)
        if match:
            seeds.add(int(match.group("seed")))
    return sorted(seeds)


def _recovery_candidate_metadata(
    candidate: str,
    candidate_dir: Path,
    histories_found: int,
    episodes: int,
    missing_files_before: list[str],
) -> dict[str, Any]:
    return {
        "runner": "scripts/recover_final_corrected_limited_reports.py",
        "candidate": candidate,
        "output_dir": str(candidate_dir),
        "histories_found": histories_found,
        "episodes": episodes,
        "missing_files_before_recovery": missing_files_before,
        "reporting_only": True,
        "td3_training_called": False,
        "note": (
            "Recovered from completed per-run histories after the original run "
            "failed during aggregation/reporting."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover final corrected limited TD3 reports from histories."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-histories", type=int, default=EXPECTED_HISTORIES)
    parser.add_argument("--episodes", type=int, default=EXPECTED_EPISODES)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()

    report = recover_reports(
        output_dir=args.output_dir,
        expected_histories=args.expected_histories,
        episodes=args.episodes,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
    )
    all_results = report["all_results"]
    top_mandate = all_results.sort_values(
        ["mandate_aware_score", "robust_score"],
        ascending=[False, False],
    ).iloc[0]
    top_robust = all_results.sort_values(
        ["robust_score", "mandate_aware_score"],
        ascending=[False, False],
    ).iloc[0]
    print("Recovery complete.")
    print("histories_found:", report["histories_found"])
    print("bad_reads:", report["bad_reads"])
    print("missing_global_before:", report["missing_global_before"])
    print("regenerated_files:")
    for path in report["regenerated_files"]:
        print(path)
    print(
        "top_mandate:",
        top_mandate["candidate_name"],
        top_mandate["cap_label"],
        top_mandate["mandate_aware_score"],
    )
    print(
        "top_robust:",
        top_robust["candidate_name"],
        top_robust["cap_label"],
        top_robust["robust_score"],
    )


if __name__ == "__main__":
    main()
