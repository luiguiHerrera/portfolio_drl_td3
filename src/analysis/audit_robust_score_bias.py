"""Audit robust_score component sensitivity and possible ranking bias.

This module is reporting-only. It recomputes sensitivity variants from an
existing protocol comparison output and does not modify the production
robust_score implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.robust_score import (
    DEFAULT_COMPOSITE_WEIGHTS,
    compute_composite_robust_score,
)


DEFAULT_OUTPUT_DIR = "outputs/tables/robust_score_bias_audit"
METRICS_FILE = "protocol_comparison_metrics.csv"

AUDIT_COLUMNS = [
    "strategy_name",
    "strategy_type",
    "robust_score",
    "current_recomputed_score",
    "dsr_score",
    "sortino_score",
    "calmar_score",
    "drawdown_score",
    "stability_score",
    "discipline_score",
    "sharpe",
    "max_drawdown",
    "average_turnover",
    "average_max_weight",
    "average_effective_number_of_assets",
    "cash_above_10pct",
    "dsr_method",
]


def build_robust_score_bias_audit(
    comparison_dir: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build robust_score component, sensitivity, flag, and method audit tables."""
    comparison_path = Path(comparison_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = _load_protocol_metrics(comparison_path)
    scoring_input = _to_scoring_input(metrics)
    scored = compute_composite_robust_score(scoring_input)
    scored = _attach_identity_columns(scored, metrics)

    component_audit = _build_component_audit(metrics, scored)
    flags = _build_bias_flags(component_audit)
    sensitivity = _build_rank_sensitivity(scored, flags)
    method_comparison = _build_method_comparison(component_audit)

    component_path = output_path / "robust_score_component_audit.csv"
    sensitivity_path = output_path / "robust_score_rank_sensitivity.csv"
    flags_path = output_path / "robust_score_drawdown_turnover_flags.csv"
    method_path = output_path / "robust_score_method_comparison.csv"

    component_audit.to_csv(component_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    flags.to_csv(flags_path, index=False)
    method_comparison.to_csv(method_path, index=False)

    return {
        "output_dir": str(output_path),
        "component_audit": component_audit,
        "rank_sensitivity": sensitivity,
        "flags": flags,
        "method_comparison": method_comparison,
        "paths": {
            "component_audit": str(component_path),
            "rank_sensitivity": str(sensitivity_path),
            "flags": str(flags_path),
            "method_comparison": str(method_path),
        },
    }


def _load_protocol_metrics(comparison_path: Path) -> pd.DataFrame:
    metrics_path = comparison_path / METRICS_FILE
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing protocol comparison metrics: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    required = {"strategy_name", "strategy_type"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"{METRICS_FILE} is missing required columns: {sorted(missing)}")
    if metrics.empty:
        raise ValueError(f"{METRICS_FILE} must not be empty.")
    return metrics


def _to_scoring_input(metrics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": metrics["strategy_name"],
            "type": metrics["strategy_type"],
            "sharpe": _numeric(metrics, "sharpe"),
            "sortino": _numeric(metrics, "sortino"),
            "calmar": _numeric(metrics, "calmar"),
            "max_drawdown": _numeric(metrics, "max_drawdown"),
            "worst_drawdown": _numeric(metrics, "max_drawdown"),
            "turnover": _numeric(metrics, "average_turnover"),
            "effective_assets": _numeric(metrics, "average_effective_number_of_assets"),
            "cash_above_10_rate": _numeric(metrics, "cash_above_10pct"),
            "pooled_dsr_n10": _numeric(metrics, "pooled_dsr_n10"),
            "pooled_dsr_n25": _numeric(metrics, "pooled_dsr_n25"),
            "pooled_dsr_n50": _numeric(metrics, "pooled_dsr_n50"),
            "mean_run_dsr_n25": _numeric(metrics, "mean_run_dsr_n25"),
            "median_run_dsr_n25": _numeric(metrics, "median_run_dsr_n25"),
            "date_averaged_dsr_n25": _numeric(metrics, "date_averaged_dsr_n25"),
            "dsr_method": metrics.get("dsr_method", pd.Series(pd.NA, index=metrics.index)),
        }
    )


def _attach_identity_columns(scored: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    result = scored.copy()
    result["strategy_name"] = metrics["strategy_name"].to_numpy()
    result["strategy_type"] = metrics["strategy_type"].to_numpy()
    result["production_robust_score"] = _numeric(metrics, "robust_score").to_numpy()
    return result


def _build_component_audit(metrics: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    audit = pd.DataFrame(
        {
            "strategy_name": scored["strategy_name"],
            "strategy_type": scored["strategy_type"],
            "robust_score": scored["production_robust_score"],
            "current_recomputed_score": scored["robust_score"],
            "dsr_score": scored["dsr_score"],
            "sortino_score": scored["sortino_score"],
            "calmar_score": scored["calmar_score"],
            "drawdown_score": scored["drawdown_score"],
            "stability_score": scored["stability_score"],
            "discipline_score": scored["discipline_score"],
            "sharpe": _numeric(metrics, "sharpe"),
            "max_drawdown": _numeric(metrics, "max_drawdown"),
            "average_turnover": _numeric(metrics, "average_turnover"),
            "average_max_weight": _numeric(metrics, "average_max_weight"),
            "average_effective_number_of_assets": _numeric(
                metrics,
                "average_effective_number_of_assets",
            ),
            "cash_above_10pct": _numeric(metrics, "cash_above_10pct"),
            "dsr_method": scored["dsr_method"],
        }
    )
    return audit.loc[:, AUDIT_COLUMNS].sort_values(
        "current_recomputed_score",
        ascending=False,
        na_position="last",
    )


def _build_bias_flags(component_audit: pd.DataFrame) -> pd.DataFrame:
    flags = component_audit[
        [
            "strategy_name",
            "strategy_type",
            "robust_score",
            "current_recomputed_score",
            "max_drawdown",
            "average_turnover",
            "average_effective_number_of_assets",
            "dsr_method",
        ]
    ].copy()
    flags["high_drawdown_flag"] = flags["max_drawdown"] < -0.35
    flags["high_turnover_flag"] = flags["average_turnover"] > 0.50
    flags["high_concentration_flag"] = (
        flags["average_effective_number_of_assets"] < 1.50
    )
    flags["single_asset_strategy_flag"] = (
        flags["average_effective_number_of_assets"] <= 1.05
    )
    flags["dsr_method_date_averaged_flag"] = flags["dsr_method"] == "date_averaged"
    flags["dsr_method_median_run_flag"] = flags["dsr_method"] == "median_run"
    return flags


def _build_rank_sensitivity(
    scored: pd.DataFrame,
    flags: pd.DataFrame,
) -> pd.DataFrame:
    base = scored.merge(
        flags[
            [
                "strategy_name",
                "high_drawdown_flag",
                "high_turnover_flag",
                "high_concentration_flag",
                "single_asset_strategy_flag",
            ]
        ],
        on="strategy_name",
        how="left",
    )
    variants = {
        "current": _weighted_score(base, DEFAULT_COMPOSITE_WEIGHTS),
        "no_dsr": _weighted_score(base, _weights(dsr_score=0.0)),
        "half_dsr": _weighted_score(base, _weights(dsr_score=0.15)),
        "double_drawdown": _weighted_score(base, _weights(drawdown_score=0.30)),
        "double_discipline": _weighted_score(base, _weights(discipline_score=0.10)),
        "drawdown_hard_cap": _apply_hard_penalty(
            _weighted_score(base, DEFAULT_COMPOSITE_WEIGHTS),
            base["high_drawdown_flag"],
        ),
        "turnover_hard_cap": _apply_hard_penalty(
            _weighted_score(base, DEFAULT_COMPOSITE_WEIGHTS),
            base["high_turnover_flag"],
        ),
        "mandate_style": _apply_mandate_penalty(
            _weighted_score(
                base,
                {
                    "dsr_score": 0.15,
                    "sortino_score": 0.15,
                    "calmar_score": 0.15,
                    "drawdown_score": 0.25,
                    "stability_score": 0.10,
                    "discipline_score": 0.20,
                },
            ),
            base,
        ),
    }

    current_rank = _rank_series(variants["current"])
    rows = []
    for variant_name, scores in variants.items():
        ranks = _rank_series(scores)
        for idx, score in scores.items():
            rows.append(
                {
                    "variant": variant_name,
                    "rank": int(ranks.loc[idx]),
                    "strategy_name": base.loc[idx, "strategy_name"],
                    "strategy_type": base.loc[idx, "strategy_type"],
                    "robust_score_variant": float(score),
                    "rank_change_vs_current": int(ranks.loc[idx] - current_rank.loc[idx]),
                }
            )
    return pd.DataFrame(rows).sort_values(["variant", "rank"]).reset_index(drop=True)


def _build_method_comparison(component_audit: pd.DataFrame) -> pd.DataFrame:
    if component_audit.empty:
        return pd.DataFrame()
    return (
        component_audit.groupby("dsr_method", dropna=False)
        .agg(
            n_strategies=("strategy_name", "count"),
            mean_robust_score=("robust_score", "mean"),
            mean_current_recomputed_score=("current_recomputed_score", "mean"),
            median_current_recomputed_score=("current_recomputed_score", "median"),
            mean_dsr_score=("dsr_score", "mean"),
            median_dsr_score=("dsr_score", "median"),
            mean_sharpe=("sharpe", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_turnover=("average_turnover", "mean"),
            mean_effective_assets=("average_effective_number_of_assets", "mean"),
        )
        .reset_index()
        .sort_values("mean_dsr_score", ascending=False, na_position="last")
    )


def _weights(**overrides: float) -> dict[str, float]:
    weights = dict(DEFAULT_COMPOSITE_WEIGHTS)
    weights.update(overrides)
    return weights


def _weighted_score(scored: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        raise ValueError("variant weights must sum to a positive value.")
    score = pd.Series(0.0, index=scored.index, dtype=float)
    for component, weight in weights.items():
        score += _numeric(scored, component).fillna(0.5) * float(weight)
    return (score / weight_sum).clip(0.0, 1.0)


def _apply_hard_penalty(score: pd.Series, condition: pd.Series) -> pd.Series:
    adjusted = score.copy()
    adjusted.loc[condition.fillna(False)] = adjusted.loc[condition.fillna(False)] - 0.25
    return adjusted.clip(0.0, 1.0)


def _apply_mandate_penalty(score: pd.Series, scored: pd.DataFrame) -> pd.Series:
    penalty = (
        0.20 * scored["high_drawdown_flag"].fillna(False).astype(float)
        + 0.15 * scored["high_turnover_flag"].fillna(False).astype(float)
        + 0.10 * scored["high_concentration_flag"].fillna(False).astype(float)
        + 0.10 * scored["single_asset_strategy_flag"].fillna(False).astype(float)
    )
    return (score - penalty).clip(0.0, 1.0)


def _rank_series(score: pd.Series) -> pd.Series:
    return score.rank(ascending=False, method="min").astype(int)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit robust_score bias and sensitivity.")
    parser.add_argument("comparison_dir")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = build_robust_score_bias_audit(
        comparison_dir=args.comparison_dir,
        output_dir=args.output_dir,
    )
    print(f"Output folder: {report['output_dir']}")
    print("\nMethod comparison:")
    print(report["method_comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
