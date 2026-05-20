"""Protocol comparison runner for TD3 candidates and benchmarks.

This module intentionally does not train TD3 models. It combines the official
benchmark-only protocol output with ingested TD3 candidate result rows so future
training runs can plug into one common reporting layer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.run_protocol_benchmark_comparison import (
    run_protocol_benchmark_comparison,
)


DEFAULT_OUTPUT_DIR = "outputs/tables/protocol_td3_comparison"
DEFAULT_DSR_POLICY = "median_run -> date_averaged -> pooled -> fallback_from_sharpe"
TIMING_CONVENTION = "information through t-1, weights for t, realized return at t"
TURNOVER_CONVENTION = "sum(abs(w_t - w_{t-1}))"

DEFAULT_TD3_CANDIDATES = (
    {
        "name": "V2_reference_full",
        "feature_version": "v2",
        "config_path": "configs/config.yaml",
    },
    {
        "name": "V5_no_volatility_block",
        "feature_version": "v5",
        "config_path": "configs/config.yaml",
    },
    {
        "name": "V6_financial_state",
        "feature_version": "v6",
        "config_path": "configs/config.yaml",
    },
)

PROTOCOL_COLUMNS = [
    "strategy_name",
    "strategy_type",
    "candidate_name",
    "feature_version",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "average_turnover",
    "total_transaction_cost",
    "average_transaction_cost",
    "average_max_weight",
    "average_effective_number_of_assets",
    "mean_cash_weight",
    "cash_above_10pct",
    "robust_score",
    "pooled_dsr_n10",
    "pooled_dsr_n25",
    "pooled_dsr_n50",
    "mean_run_dsr_n25",
    "median_run_dsr_n25",
    "date_averaged_dsr_n25",
    "dsr_method",
]

MODEL_SELECTION_FLAGS = [
    "beats_equal_weight",
    "beats_equal_weight_risky",
    "beats_risk_parity",
    "beats_markowitz_long_only",
    "drawdown_not_worse_than_equal_weight",
    "turnover_acceptable",
    "robust_score_rank",
    "final_protocol_rank",
]


@dataclass(frozen=True)
class ProtocolTD3Candidate:
    """Configuration metadata for an ingested TD3 candidate."""

    name: str
    config_path: str | None = None
    feature_version: str | None = None
    output_dir: str | None = None
    metrics_path: str | None = None
    history_path: str | None = None
    seeds: list[int] = field(default_factory=list)
    episodes: int | None = None
    folds: list[str] = field(default_factory=list)


def run_protocol_td3_comparison(
    returns_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    candidates: list[ProtocolTD3Candidate | dict[str, Any]] | None = None,
    td3_results: pd.DataFrame | list[dict[str, Any]] | None = None,
    td3_results_path: str | None = None,
    smoke: bool = False,
    transaction_cost: float = 0.001,
    initial_value: float = 100000.0,
    date_column: str = "date",
) -> dict[str, Any]:
    """Combine protocol benchmarks with ingested TD3 candidate results."""
    output_path = Path(output_dir)
    benchmark_output_dir = output_path / "benchmarks"
    histories_dir = output_path / "histories"
    output_path.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)

    candidate_configs = _normalize_candidates(candidates)
    benchmark_result = run_protocol_benchmark_comparison(
        returns_path=returns_path,
        output_dir=str(benchmark_output_dir),
        transaction_cost=transaction_cost,
        initial_value=initial_value,
        date_column=date_column,
    )

    benchmark_metrics = _normalize_benchmark_metrics(benchmark_result["metrics_table"])
    td3_metrics, td3_ingestion_info = _load_td3_candidate_metrics(
        td3_results=td3_results,
        td3_results_path=td3_results_path,
        candidates=candidate_configs,
    )
    combined_metrics = _concat_protocol_frames([benchmark_metrics, td3_metrics])
    comparison_summary = _build_protocol_comparison_summary(combined_metrics)
    diagnostics = _build_protocol_diagnostics(combined_metrics)
    model_selection = _build_model_selection_table(combined_metrics)

    paths = _write_protocol_outputs(
        output_path=output_path,
        histories_dir=histories_dir,
        benchmark_result=benchmark_result,
        benchmark_metrics=benchmark_metrics,
        td3_metrics=td3_metrics,
        combined_metrics=combined_metrics,
        comparison_summary=comparison_summary,
        diagnostics=diagnostics,
        model_selection=model_selection,
    )
    _copy_td3_histories(candidate_configs, histories_dir, paths["histories"])

    metadata = _build_protocol_metadata(
        returns_path=returns_path,
        candidates=candidate_configs,
        smoke=smoke,
        transaction_cost=transaction_cost,
        td3_ingestion_info=td3_ingestion_info,
    )
    metadata_path = output_path / "protocol_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["metadata"] = str(metadata_path)

    return {
        "benchmark_result": benchmark_result,
        "benchmark_metrics": benchmark_metrics,
        "td3_candidate_metrics": td3_metrics,
        "combined_metrics": combined_metrics,
        "comparison_summary": comparison_summary,
        "diagnostics": diagnostics,
        "model_selection": model_selection,
        "metadata": metadata,
        "paths": paths,
    }


def _normalize_candidates(
    candidates: list[ProtocolTD3Candidate | dict[str, Any]] | None,
) -> list[ProtocolTD3Candidate]:
    raw_candidates = candidates if candidates is not None else list(DEFAULT_TD3_CANDIDATES)
    normalized = []
    for candidate in raw_candidates:
        if isinstance(candidate, ProtocolTD3Candidate):
            normalized.append(candidate)
        else:
            normalized.append(
                ProtocolTD3Candidate(
                    name=str(candidate["name"]),
                    config_path=candidate.get("config_path"),
                    feature_version=candidate.get("feature_version"),
                    output_dir=candidate.get("output_dir"),
                    metrics_path=candidate.get("metrics_path"),
                    history_path=candidate.get("history_path"),
                    seeds=list(candidate.get("seeds", [])),
                    episodes=candidate.get("episodes"),
                    folds=list(candidate.get("folds", [])),
                )
            )
    return normalized


def _normalize_benchmark_metrics(metrics_table: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics_table.copy()
    metrics["strategy_name"] = metrics["benchmark_name"]
    metrics["strategy_type"] = "benchmark"
    metrics["candidate_name"] = pd.NA
    metrics["feature_version"] = pd.NA
    return _ensure_protocol_columns(metrics)


def _load_td3_candidate_metrics(
    td3_results: pd.DataFrame | list[dict[str, Any]] | None,
    td3_results_path: str | None,
    candidates: list[ProtocolTD3Candidate],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    ingestion_info: dict[str, Any] = {
        "td3_results_path": td3_results_path,
        "td3_results_path_type": None,
        "td3_ingestion_source_files": [],
        "td3_transaction_cost_note": None,
    }
    if td3_results is not None:
        frames.append(_coerce_td3_results(td3_results))

    if td3_results_path:
        path_frame, path_info = _ingest_td3_results_path(td3_results_path)
        frames.append(path_frame)
        ingestion_info.update(path_info)

    for candidate in candidates:
        if candidate.metrics_path:
            metrics_path = Path(candidate.metrics_path)
            if metrics_path.exists():
                frame = pd.read_csv(metrics_path)
                frame["candidate_name"] = frame.get("candidate_name", candidate.name)
                frames.append(frame)
                ingestion_info["td3_ingestion_source_files"].append(str(metrics_path))

    if not frames:
        return (
            _ensure_protocol_columns(pd.DataFrame(columns=PROTOCOL_COLUMNS)),
            ingestion_info,
        )

    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    td3_metrics = pd.concat(frames, ignore_index=True)
    td3_metrics = _normalize_generic_td3_columns(td3_metrics)
    td3_metrics["candidate_name"] = td3_metrics.get(
        "candidate_name",
        td3_metrics.get("strategy_name"),
    )
    td3_metrics["strategy_name"] = td3_metrics.get(
        "strategy_name",
        td3_metrics["candidate_name"],
    )
    td3_metrics["strategy_type"] = "td3"

    for index, row in td3_metrics.iterrows():
        candidate_name = row.get("candidate_name")
        candidate = candidate_by_name.get(str(candidate_name))
        if candidate and pd.isna(row.get("feature_version")):
            td3_metrics.loc[index, "feature_version"] = candidate.feature_version
        elif pd.isna(row.get("feature_version")):
            td3_metrics.loc[index, "feature_version"] = _infer_feature_version(
                str(candidate_name)
            )
    return _ensure_protocol_columns(td3_metrics), ingestion_info


def _ingest_td3_results_path(td3_results_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(td3_results_path)
    if not path.exists():
        raise FileNotFoundError(f"td3_results_path does not exist: {td3_results_path}")

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError("td3_results_path file must be a CSV.")
        return pd.read_csv(path), {
            "td3_results_path": str(path),
            "td3_results_path_type": "csv",
            "td3_ingestion_source_files": [str(path)],
            "td3_transaction_cost_note": None,
        }

    if path.is_dir():
        return _ingest_td3_experiment_directory(path)

    raise ValueError(f"td3_results_path must be a CSV file or directory: {td3_results_path}")


def _ingest_td3_experiment_directory(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    aggregate_path = path / "overall_aggregate_by_strategy_split.csv"
    robust_score_path = path / "robust_score_ranking.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(
            "TD3 experiment directory is missing overall_aggregate_by_strategy_split.csv"
        )

    aggregate = pd.read_csv(aggregate_path)
    required_columns = {"strategy", "split", "strategy_type"}
    missing_columns = required_columns.difference(aggregate.columns)
    if missing_columns:
        raise ValueError(
            "overall_aggregate_by_strategy_split.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    td3_rows = aggregate[
        (aggregate["split"] == "test")
        & (aggregate["strategy_type"].astype(str).str.lower() == "drl")
    ].copy()
    normalized = _normalize_td3_experiment_aggregate(td3_rows)

    source_files = [str(aggregate_path)]
    if robust_score_path.exists():
        robust_scores = pd.read_csv(robust_score_path)
        normalized = _merge_robust_score_ranking(normalized, robust_scores)
        source_files.append(str(robust_score_path))

    transaction_cost_note = None
    if "mean_transaction_cost" in aggregate.columns:
        transaction_cost_note = (
            "TD3 aggregate output provides mean_transaction_cost only; "
            "total_transaction_cost is left missing unless reconstructed from histories."
        )

    return normalized, {
        "td3_results_path": str(path),
        "td3_results_path_type": "directory",
        "td3_ingestion_source_files": source_files,
        "td3_transaction_cost_note": transaction_cost_note,
    }


def _normalize_td3_experiment_aggregate(aggregate: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "strategy": "strategy_name",
        "mean_cumulative_return": "cumulative_return",
        "mean_annualized_return": "annualized_return",
        "mean_annualized_volatility": "annualized_volatility",
        "mean_sharpe": "sharpe",
        "mean_sortino": "sortino",
        "mean_calmar": "calmar",
        "mean_max_drawdown": "max_drawdown",
        "mean_average_turnover": "average_turnover",
        "mean_average_max_weight": "average_max_weight",
        "mean_average_effective_number_of_assets": (
            "average_effective_number_of_assets"
        ),
        "mean_cash_weight": "mean_cash_weight",
        "cash_above_10_rate": "cash_above_10pct",
        "mean_transaction_cost": "average_transaction_cost",
    }
    available_map = {
        source: target
        for source, target in column_map.items()
        if source in aggregate.columns
    }
    normalized = aggregate.rename(columns=available_map).copy()
    normalized["candidate_name"] = normalized["strategy_name"]
    normalized["strategy_type"] = "td3"
    normalized["feature_version"] = normalized["strategy_name"].map(
        _infer_feature_version
    )
    normalized["total_transaction_cost"] = pd.NA
    return normalized


def _merge_robust_score_ranking(
    normalized: pd.DataFrame,
    robust_scores: pd.DataFrame,
) -> pd.DataFrame:
    if "type" in robust_scores.columns:
        robust_scores = robust_scores[
            robust_scores["type"].astype(str).str.lower() == "drl"
        ].copy()
    robust_columns = [
        "strategy",
        "robust_score",
        "pooled_dsr_n10",
        "pooled_dsr_n25",
        "pooled_dsr_n50",
        "mean_run_dsr_n25",
        "median_run_dsr_n25",
        "date_averaged_dsr_n25",
        "dsr_method",
    ]
    available_columns = [
        column for column in robust_columns if column in robust_scores.columns
    ]
    if "strategy" not in available_columns:
        return normalized

    robust_subset = robust_scores.loc[:, available_columns].rename(
        columns={"strategy": "strategy_name"}
    )
    return normalized.merge(
        robust_subset,
        on="strategy_name",
        how="left",
        suffixes=("", "_robust"),
    )


def _infer_feature_version(strategy_name: str) -> str:
    feature_versions = {
        "V2_reference_full": "v2",
        "V5_no_volatility_block": "v5",
        "V5_momentum_only_or_minimal_momentum_regime": "v5",
        "V6_financial_state": "v6",
    }
    return feature_versions.get(strategy_name, "unknown")


def _coerce_td3_results(
    td3_results: pd.DataFrame | list[dict[str, Any]],
) -> pd.DataFrame:
    if isinstance(td3_results, pd.DataFrame):
        return td3_results.copy()
    return pd.DataFrame(td3_results)


def _normalize_generic_td3_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "strategy_name" not in normalized.columns and "strategy" in normalized.columns:
        normalized = normalized.rename(columns={"strategy": "strategy_name"})
    return normalized


def _ensure_protocol_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in PROTOCOL_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, PROTOCOL_COLUMNS]


def _concat_protocol_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return _ensure_protocol_columns(pd.DataFrame(columns=PROTOCOL_COLUMNS))
    compact_frames = [
        frame.dropna(axis=1, how="all")
        for frame in non_empty_frames
    ]
    return _ensure_protocol_columns(pd.concat(compact_frames, ignore_index=True))


def _build_protocol_comparison_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = metrics.copy()
    if summary["robust_score"].notna().any():
        return summary.sort_values(
            ["robust_score", "sharpe", "cumulative_return"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return summary.sort_values(
        ["sharpe", "cumulative_return"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def _build_protocol_diagnostics(metrics: pd.DataFrame) -> pd.DataFrame:
    diagnostic_columns = [
        "strategy_name",
        "strategy_type",
        "candidate_name",
        "feature_version",
        "average_turnover",
        "total_transaction_cost",
        "average_transaction_cost",
        "average_max_weight",
        "average_effective_number_of_assets",
        "mean_cash_weight",
        "cash_above_10pct",
        "max_drawdown",
        "dsr_method",
    ]
    return metrics.loc[:, diagnostic_columns].copy()


def _build_model_selection_table(metrics: pd.DataFrame) -> pd.DataFrame:
    table = metrics.copy()
    sharpe_by_name = _series_by_strategy(table, "sharpe")
    drawdown_by_name = _series_by_strategy(table, "max_drawdown")

    equal_weight_sharpe = sharpe_by_name.get("Equal_Weight")
    equal_weight_risky_sharpe = sharpe_by_name.get("Equal_Weight_Risky")
    risk_parity_sharpe = sharpe_by_name.get("rolling_risk_parity_inverse_vol_12p")
    markowitz_sharpe = sharpe_by_name.get("rolling_markowitz_long_only_52p")
    equal_weight_drawdown = drawdown_by_name.get("Equal_Weight")

    table["beats_equal_weight"] = _greater_than(table["sharpe"], equal_weight_sharpe)
    table["beats_equal_weight_risky"] = _greater_than(
        table["sharpe"],
        equal_weight_risky_sharpe,
    )
    table["beats_risk_parity"] = _greater_than(table["sharpe"], risk_parity_sharpe)
    table["beats_markowitz_long_only"] = _greater_than(table["sharpe"], markowitz_sharpe)
    table["drawdown_not_worse_than_equal_weight"] = _greater_than(
        table["max_drawdown"],
        equal_weight_drawdown,
        or_equal=True,
    )
    table["turnover_acceptable"] = pd.to_numeric(
        table["average_turnover"],
        errors="coerce",
    ) <= 0.75

    if table["robust_score"].notna().any():
        table["robust_score_rank"] = pd.to_numeric(
            table["robust_score"],
            errors="coerce",
        ).rank(ascending=False, method="min")
    else:
        table["robust_score_rank"] = pd.NA

    table = table.sort_values(
        ["robust_score", "sharpe", "cumulative_return", "max_drawdown"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    table["final_protocol_rank"] = range(1, len(table) + 1)

    return table.loc[
        :,
        [
            "strategy_name",
            "strategy_type",
            "candidate_name",
            "feature_version",
            *MODEL_SELECTION_FLAGS,
            "sharpe",
            "robust_score",
            "max_drawdown",
            "average_turnover",
        ],
    ]


def _series_by_strategy(metrics: pd.DataFrame, column: str) -> dict[str, float]:
    numeric = pd.to_numeric(metrics[column], errors="coerce")
    return dict(zip(metrics["strategy_name"], numeric))


def _greater_than(
    values: pd.Series,
    benchmark: float | None,
    or_equal: bool = False,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if benchmark is None or pd.isna(benchmark):
        return pd.Series(False, index=values.index)
    if or_equal:
        return numeric >= float(benchmark)
    return numeric > float(benchmark)


def _write_protocol_outputs(
    output_path: Path,
    histories_dir: Path,
    benchmark_result: dict[str, Any],
    benchmark_metrics: pd.DataFrame,
    td3_metrics: pd.DataFrame,
    combined_metrics: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    model_selection: pd.DataFrame,
) -> dict[str, Any]:
    paths: dict[str, Any] = {
        "output_dir": str(output_path),
        "histories_dir": str(histories_dir),
        "histories": {},
    }
    output_map = {
        "protocol_comparison_metrics": (
            output_path / "protocol_comparison_metrics.csv",
            combined_metrics,
        ),
        "protocol_comparison_summary": (
            output_path / "protocol_comparison_summary.csv",
            comparison_summary,
        ),
        "protocol_comparison_diagnostics": (
            output_path / "protocol_comparison_diagnostics.csv",
            diagnostics,
        ),
        "protocol_model_selection_table": (
            output_path / "protocol_model_selection_table.csv",
            model_selection,
        ),
        "benchmark_metrics_table": (
            output_path / "benchmark_metrics_table.csv",
            benchmark_metrics,
        ),
        "td3_candidate_metrics_table": (
            output_path / "td3_candidate_metrics_table.csv",
            td3_metrics,
        ),
    }
    for key, (path, frame) in output_map.items():
        frame.to_csv(path, index=False)
        paths[key] = str(path)

    for benchmark_name, history_path in benchmark_result["paths"]["histories"].items():
        destination = histories_dir / Path(history_path).name
        shutil.copy2(history_path, destination)
        paths["histories"][benchmark_name] = str(destination)
    return paths


def _copy_td3_histories(
    candidates: list[ProtocolTD3Candidate],
    histories_dir: Path,
    history_paths: dict[str, str],
) -> None:
    for candidate in candidates:
        if not candidate.history_path:
            continue
        source = Path(candidate.history_path)
        if not source.exists():
            continue
        destination = histories_dir / f"{_safe_filename(candidate.name)}_history.csv"
        shutil.copy2(source, destination)
        history_paths[candidate.name] = str(destination)


def _build_protocol_metadata(
    returns_path: str,
    candidates: list[ProtocolTD3Candidate],
    smoke: bool,
    transaction_cost: float,
    td3_ingestion_info: dict[str, Any],
) -> dict[str, Any]:
    seeds = sorted({seed for candidate in candidates for seed in candidate.seeds})
    episodes = sorted(
        {
            int(candidate.episodes)
            for candidate in candidates
            if candidate.episodes is not None
        }
    )
    if smoke:
        seeds = seeds or [7]
        episodes = episodes or [1]

    metadata = {
        "data_path": returns_path,
        "config_paths": [
            candidate.config_path for candidate in candidates if candidate.config_path
        ],
        "feature_versions": [
            candidate.feature_version
            for candidate in candidates
            if candidate.feature_version
        ],
        "seeds": seeds,
        "episodes": episodes,
        "transaction_cost_rate": transaction_cost,
        "turnover_convention": TURNOVER_CONVENTION,
        "timing_convention": TIMING_CONVENTION,
        "DSR_method_policy": DEFAULT_DSR_POLICY,
        "git_commit_hash": _git_commit_hash(),
        "smoke_mode": bool(smoke),
    }
    metadata.update(td3_ingestion_info)
    return metadata


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


def _safe_filename(value: str) -> str:
    return value.replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine protocol benchmarks with ingested TD3 candidate results.",
    )
    parser.add_argument("--returns-path", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--td3-results-path")
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--initial-value", type=float, default=100000.0)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    result = run_protocol_td3_comparison(
        returns_path=args.returns_path,
        output_dir=args.output_dir,
        td3_results_path=args.td3_results_path,
        smoke=args.smoke,
        transaction_cost=args.transaction_cost,
        initial_value=args.initial_value,
        date_column=args.date_column,
    )
    print(result["comparison_summary"].to_string(index=False))
    print(f"Outputs written to {result['paths']['output_dir']}")


if __name__ == "__main__":
    main()
