"""Full max-weight cap sensitivity experiment wrapper.

This wrapper reuses the experiment-only max-weight cap runner for each TD3
candidate, then builds combined reporting tables across candidates and cap
levels. It does not change TD3 architecture, reward, environment defaults, or
training logic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.run_feature_block_ablation import BASE_CONFIG_PATH, RETURNS_PATH
from src.experiments.run_max_weight_cap_experiment import (
    parse_max_weight_grid,
    parse_seed_list,
    run_max_weight_cap_experiment,
)
from src.experiments.run_protocol_pure_td3_revalidation import (
    DSR_METHOD,
    TIMING_CONVENTION,
)


DEFAULT_OUTPUT_DIR = "outputs/tables/cap_sensitivity_experiment_60ep_10seeds"
DEFAULT_CANDIDATES = [
    "V2_reference_full",
    "V5_no_volatility_block",
    "V6_financial_state",
]
DEFAULT_CAP_GRID = [None, 0.50, 0.60, 0.70, 0.80]
DEFAULT_SEEDS = [7, 21, 42, 84, 101, 123, 202, 303, 404, 505]
DEFAULT_EPISODES = 60

MEANINGFUL_DIVERSIFICATION_IMPROVEMENT = 0.20
MATERIAL_SCORE_DETERIORATION = 0.05
DRAWDOWN_WORSENING_TOLERANCE = -0.02
TURNOVER_IMPROVEMENT = -0.05

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


def run_cap_sensitivity_experiment(
    returns_path: str = RETURNS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    candidates: list[str] | None = None,
    max_weight_grid: list[float | None] | None = None,
    episodes: int = DEFAULT_EPISODES,
    seeds: list[int] | None = None,
    max_folds: int | None = None,
    transaction_cost: float = 0.001,
    base_config_path: str = BASE_CONFIG_PATH,
) -> dict[str, Any]:
    """Run per-candidate cap grids and write combined cap sensitivity reports."""
    selected_candidates = list(DEFAULT_CANDIDATES if candidates is None else candidates)
    selected_caps = list(DEFAULT_CAP_GRID if max_weight_grid is None else max_weight_grid)
    selected_seeds = list(DEFAULT_SEEDS if seeds is None else seeds)

    destination = Path(output_dir)
    per_candidate_dir = destination / "per_candidate"
    per_candidate_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    all_rows = []
    for candidate in selected_candidates:
        candidate_output = per_candidate_dir / candidate
        report = run_max_weight_cap_experiment(
            returns_path=returns_path,
            output_dir=str(candidate_output),
            candidate=candidate,
            max_weight_grid=selected_caps,
            episodes=episodes,
            seeds=selected_seeds,
            max_folds=max_folds,
            transaction_cost=transaction_cost,
            base_config_path=base_config_path,
        )
        reports[candidate] = report
        rankings = report["rankings"].copy()
        rankings["candidate_output_dir"] = str(candidate_output)
        all_rows.append(rankings)

    all_results = build_cap_sensitivity_all_results(all_rows)
    pairwise_deltas = build_cap_sensitivity_pairwise_deltas(all_results)
    best_caps = build_cap_sensitivity_best_caps(all_results)
    summary = build_cap_sensitivity_summary(all_results, best_caps)
    markdown = build_cap_sensitivity_markdown(summary, best_caps)
    metadata = build_cap_sensitivity_metadata(
        returns_path=returns_path,
        output_dir=str(destination),
        candidates=selected_candidates,
        max_weight_grid=selected_caps,
        episodes=episodes,
        seeds=selected_seeds,
        max_folds=max_folds,
        transaction_cost=transaction_cost,
        base_config_path=base_config_path,
        reports=reports,
    )

    paths = write_cap_sensitivity_outputs(
        output_dir=destination,
        all_results=all_results,
        pairwise_deltas=pairwise_deltas,
        best_caps=best_caps,
        summary=summary,
        markdown=markdown,
        metadata=metadata,
    )

    return {
        "all_results": all_results,
        "pairwise_deltas": pairwise_deltas,
        "best_caps": best_caps,
        "summary": summary,
        "markdown_summary": markdown,
        "metadata": metadata,
        "paths": paths,
        "reports": reports,
    }


def build_cap_sensitivity_all_results(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-candidate cap ranking frames."""
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
    """Compute cap-vs-uncapped deltas per candidate and cap."""
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
    """Identify the best cap per candidate by several metrics."""
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
                "overall_interpretation": interpret_candidate_cap_sensitivity(group),
            }
        )
    return pd.DataFrame(rows)


def build_cap_sensitivity_summary(
    all_results: pd.DataFrame,
    best_caps: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact candidate-level cap sensitivity summary."""
    rows = []
    for base_candidate, group in all_results.groupby("base_candidate", dropna=True):
        baseline = group[group["max_weight_cap"].isna()].iloc[0]
        capped = group[group["max_weight_cap"].notna()].copy()
        best_row = capped.sort_values(
            ["mandate_aware_score", "robust_score"],
            ascending=[False, False],
        ).iloc[0]
        best_cap_info = best_caps.set_index("base_candidate").loc[base_candidate]
        rows.append(
            {
                "base_candidate": base_candidate,
                "uncapped_mandate_aware_score": baseline["mandate_aware_score"],
                "best_cap_by_mandate_aware_score": best_cap_info[
                    "best_by_mandate_aware_score"
                ],
                "best_cap_mandate_aware_score": best_cap_info[
                    "best_mandate_aware_score"
                ],
                "best_cap_by_robust_score": best_cap_info["best_by_robust_score"],
                "best_cap_robust_score": best_cap_info["best_robust_score"],
                "best_tested_cap_candidate": best_row["candidate_name"],
                "best_tested_cap_label": best_row["cap_label"],
                "best_tested_cap_decision": best_row["decision_label"],
                "n_caps_better_than_uncapped_mandate": int(
                    (
                        pd.to_numeric(capped["mandate_aware_score"], errors="coerce")
                        > _numeric(baseline["mandate_aware_score"])
                    ).sum()
                ),
                "n_caps_better_than_uncapped_robust": int(
                    (
                        pd.to_numeric(capped["robust_score"], errors="coerce")
                        > _numeric(baseline["robust_score"])
                    ).sum()
                ),
                "n_caps_lower_turnover_than_uncapped": int(
                    (
                        pd.to_numeric(capped["average_turnover"], errors="coerce")
                        < _numeric(baseline["average_turnover"])
                    ).sum()
                ),
                "n_caps_higher_effective_assets_than_uncapped": int(
                    (
                        pd.to_numeric(
                            capped["average_effective_number_of_assets"],
                            errors="coerce",
                        )
                        > _numeric(baseline["average_effective_number_of_assets"])
                    ).sum()
                ),
                "overall_interpretation": best_cap_info["overall_interpretation"],
            }
        )
    return pd.DataFrame(rows)


def label_cap_sensitivity_decision(row: pd.Series) -> str:
    """Assign requested cap sensitivity decision labels."""
    if pd.isna(row.get("max_weight_cap")):
        return "uncapped_baseline"
    effective_delta = _numeric(row.get("delta_average_effective_number_of_assets_vs_baseline"))
    robust_delta = _numeric(row.get("delta_robust_score_vs_baseline"))
    mandate_delta = _numeric(row.get("delta_mandate_aware_score_vs_baseline"))
    drawdown_delta = _numeric(row.get("delta_max_drawdown_vs_baseline"))
    turnover_delta = _numeric(row.get("delta_average_turnover_vs_baseline"))
    return_delta = _numeric(row.get("delta_annualized_return_vs_baseline"))

    if effective_delta <= MEANINGFUL_DIVERSIFICATION_IMPROVEMENT:
        if robust_delta < -MATERIAL_SCORE_DETERIORATION and mandate_delta < -MATERIAL_SCORE_DETERIORATION:
            return "uncapped_preferred"
        return "cap_inconclusive"
    if robust_delta >= -MATERIAL_SCORE_DETERIORATION and mandate_delta >= -MATERIAL_SCORE_DETERIORATION:
        if drawdown_delta >= DRAWDOWN_WORSENING_TOLERANCE:
            if return_delta < -MATERIAL_SCORE_DETERIORATION:
                return "cap_improves_mandate_but_hurts_return"
            return "cap_dominates_uncapped"
        return "cap_inconclusive"
    if mandate_delta > 0.0 and return_delta < 0.0:
        return "cap_improves_mandate_but_hurts_return"
    if effective_delta > MEANINGFUL_DIVERSIFICATION_IMPROVEMENT and turnover_delta >= 0.0:
        return "cap_reduces_concentration_only"
    if robust_delta < -MATERIAL_SCORE_DETERIORATION and mandate_delta < -MATERIAL_SCORE_DETERIORATION:
        return "uncapped_preferred"
    return "cap_inconclusive"


def interpret_candidate_cap_sensitivity(group: pd.DataFrame) -> str:
    """Classify overall cap sensitivity for one candidate."""
    capped = group[group["max_weight_cap"].notna()].copy()
    if capped.empty:
        return "no_clear_cap_benefit"
    mandate_labels = set(capped["decision_label"].astype(str))
    mandate_improvements = (
        pd.to_numeric(capped["delta_mandate_aware_score_vs_baseline"], errors="coerce")
        > MATERIAL_SCORE_DETERIORATION
    )
    robust_improvements = (
        pd.to_numeric(capped["delta_robust_score_vs_baseline"], errors="coerce")
        > MATERIAL_SCORE_DETERIORATION
    )
    if mandate_improvements.all() and robust_improvements.all():
        return "stable_cap_benefit"
    if mandate_improvements.any() or robust_improvements.any():
        return "threshold_sensitive"
    if mandate_labels == {"uncapped_preferred"}:
        return "no_clear_cap_benefit"
    return "threshold_sensitive"


def build_cap_sensitivity_markdown(
    summary: pd.DataFrame,
    best_caps: pd.DataFrame,
) -> str:
    """Build Markdown summary for cap sensitivity experiment."""
    lines = [
        "# Cap Sensitivity Experiment Summary",
        "",
        "This experiment tests whether the max-weight cap result is robust across "
        "reasonable cap levels rather than cherry-picked at 0.60.",
        "",
        "## Candidate Summary",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "- "
            f"{row['base_candidate']}: best mandate cap = "
            f"`{row['best_cap_by_mandate_aware_score']}`; "
            f"best robust cap = `{row['best_cap_by_robust_score']}`; "
            f"interpretation = `{row['overall_interpretation']}`."
        )
    lines.extend(["", "## Best Cap Table", "", _markdown_table(best_caps), ""])
    return "\n".join(lines)


def build_cap_sensitivity_metadata(
    returns_path: str,
    output_dir: str,
    candidates: list[str],
    max_weight_grid: list[float | None],
    episodes: int,
    seeds: list[int],
    max_folds: int | None,
    transaction_cost: float,
    base_config_path: str,
    reports: dict[str, Any],
) -> dict[str, Any]:
    """Build metadata for the combined cap sensitivity run."""
    return {
        "runner": "src.experiments.run_cap_sensitivity_experiment",
        "git_commit_hash": _git_commit_hash(),
        "returns_path": returns_path,
        "output_dir": output_dir,
        "candidates": candidates,
        "max_weight_grid": [cap if cap is not None else "uncapped" for cap in max_weight_grid],
        "episodes": episodes,
        "seeds": seeds,
        "max_folds": max_folds,
        "transaction_cost": transaction_cost,
        "base_config_path": base_config_path,
        "timing_convention": TIMING_CONVENTION,
        "DSR_method_policy": DSR_METHOD,
        "experiment_only_warning": (
            "Max-weight caps are experiment-only allocation projections and are "
            "not default behavior."
        ),
        "candidate_output_dirs": {
            candidate: report["output_dir"] for candidate, report in reports.items()
        },
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
    """Write combined cap sensitivity outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
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
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def parse_candidate_list(value: str | None) -> list[str]:
    """Parse comma-separated candidate names."""
    if value is None or not value.strip():
        return list(DEFAULT_CANDIDATES)
    return [item.strip() for item in value.split(",") if item.strip()]


def format_cap_label(value: Any) -> str:
    """Format cap value for reports."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "uncapped"
    return f"{float(numeric):.2f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a simple Markdown table without optional dependencies."""
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


def _best_cap_label(group: pd.DataFrame, metric: str, ascending: bool) -> str:
    row = _best_row(group, metric, ascending)
    return str(row["cap_label"])


def _best_metric_value(group: pd.DataFrame, metric: str, ascending: bool) -> float:
    row = _best_row(group, metric, ascending)
    return _numeric(row[metric])


def _best_row(group: pd.DataFrame, metric: str, ascending: bool) -> pd.Series:
    numeric = pd.to_numeric(group[metric], errors="coerce")
    ordered = group.assign(_metric=numeric).sort_values(
        "_metric",
        ascending=ascending,
        na_position="last",
    )
    return ordered.iloc[0]


def _numeric(value: Any) -> float:
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
        description="Run full TD3 max-weight cap sensitivity experiment.",
    )
    parser.add_argument("--returns-path", default=RETURNS_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--max-weight-grid", default="uncapped,0.50,0.60,0.70,0.80")
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--base-config-path", default=BASE_CONFIG_PATH)
    args = parser.parse_args()

    report = run_cap_sensitivity_experiment(
        returns_path=args.returns_path,
        output_dir=args.output_dir,
        candidates=parse_candidate_list(args.candidates),
        max_weight_grid=parse_max_weight_grid(args.max_weight_grid),
        episodes=args.episodes,
        seeds=parse_seed_list(args.seeds),
        max_folds=args.max_folds,
        transaction_cost=args.transaction_cost,
        base_config_path=args.base_config_path,
    )
    print("Cap sensitivity summary:")
    print(report["summary"].to_string(index=False))
    print("\nBest caps:")
    print(report["best_caps"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
