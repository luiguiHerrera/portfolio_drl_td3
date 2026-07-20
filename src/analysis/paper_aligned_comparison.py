"""Build the paper comparison on one exact out-of-sample date index.

This reporting-only module never trains or evaluates a policy. It reads the
canonical TD3 test histories and deterministic benchmark histories, verifies
their timestamp coverage, filters them to one exact common index, and derives
all comparison metrics and downstream diagnostics from those aligned rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.analysis.asset_specific_constraint_pareto_report import (
    PARETO_FULL_OBJECTIVES,
    PARETO_REDUCED_OBJECTIVES,
    build_constraint_pass_fail_matrix,
    build_feasible_strategy_rankings,
    build_pareto_tables,
    build_standard_metric_rankings,
    canonical_mandate_profiles,
)
from src.analysis.mandate_aware_score import (
    assign_drawdown_bucket,
    calculate_recovery_required,
    get_drawdown_multiplier,
)
from src.analysis.mandate_profile_comparison_report import (
    build_profile_rankings,
    build_profile_winners,
    score_strategies_for_profiles,
)
from src.analysis.robust_score import (
    DEFAULT_COMPOSITE_WEIGHTS,
    compute_composite_robust_score,
    compute_deflated_sharpe_ratio,
)
from src.backtest.evaluate_policy import (
    annualized_return,
    annualized_volatility,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
)
from src.backtest.performance_metrics import calmar_ratio, sortino_ratio


PERIODS_PER_YEAR = 52
EXPECTED_ASSETS = ["SPY", "TLT", "GLD", "BTC-USD", "CASH"]
EXPECTED_TD3_STRATEGIES = 5
EXPECTED_BENCHMARKS = 14
INITIAL_VALUE = 100_000.0
EXPECTED_SEEDS = [7, 21, 42, 84, 101, 123, 202, 303, 404, 505]
DATE_COLUMN = "date"
RETURN_COLUMN = "financial_net_return"
WEIGHT_COLUMNS = [f"weight_{asset}" for asset in EXPECTED_ASSETS]
ASSET_TURNOVER_COLUMNS = [f"asset_turnover_{asset}" for asset in EXPECTED_ASSETS]
ASSET_COST_COLUMNS = [
    f"asset_transaction_cost_contribution_{asset}" for asset in EXPECTED_ASSETS
]
HISTORY_NUMERIC_COLUMNS = [
    "portfolio_return",
    RETURN_COLUMN,
    "turnover",
    "transaction_cost",
    *WEIGHT_COLUMNS,
    *ASSET_TURNOVER_COLUMNS,
    *ASSET_COST_COLUMNS,
    "diagnostic_max_weight",
    "diagnostic_effective_assets",
]
BASE_METRIC_COLUMNS = [
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "average_turnover",
    "total_transaction_cost",
    "mean_transaction_cost",
    "average_max_weight",
    "average_effective_number_of_assets",
    "mean_cash_weight",
    "cash_above_10_rate",
    "mean_btc_weight",
    "mean_cash_transaction_cost_contribution",
    "mean_btc_transaction_cost_contribution",
]
SCORE_COMPONENT_COLUMNS = [
    "dsr_score",
    "sortino_score",
    "calmar_score",
    "drawdown_score",
    "stability_score",
    "discipline_score",
    "robust_score",
    "mandate_aware_score",
]


@dataclass(frozen=True)
class ProtocolSources:
    """Canonical input and legacy-audit locations for one cash protocol."""

    protocol: str
    td3_dir: Path
    benchmark_dir: Path
    old_comparison_dir: Path
    old_constraint_dir: Path
    old_mandate_dir: Path
    statistical_dir: Path
    wrc_dir: Path
    cash_bps: float
    cash_return_assumption: str


def build_paper_aligned_comparison(
    repo_root: str | Path,
    external_root: str | Path,
    output_dir: str | Path = "outputs/paper_aligned_comparison",
    expected_observations: int = 228,
) -> dict[str, Any]:
    """Generate the aligned comparison package and validate every dependency."""
    repo = Path(repo_root).expanduser().resolve()
    external = Path(external_root).expanduser().resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = repo / output
    output = output.resolve()

    sources = protocol_sources(repo, external)
    _ensure_output_directories(output)
    generated_at = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit(repo)

    all_histories: list[pd.DataFrame] = []
    all_seed_returns: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    all_rankings: list[pd.DataFrame] = []
    all_robust: list[pd.DataFrame] = []
    all_pairwise: list[pd.DataFrame] = []
    all_constraints: list[pd.DataFrame] = []
    all_feasible: list[pd.DataFrame] = []
    all_profiles: list[pd.DataFrame] = []
    all_profile_winners: list[pd.DataFrame] = []
    all_pareto: list[pd.DataFrame] = []
    all_dominated: list[pd.DataFrame] = []
    all_standard_rankings: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    common_indices: dict[str, pd.DatetimeIndex] = {}
    protocol_metadata: dict[str, Any] = {}

    for source in sources:
        selected = _selected_td3_specs(source.td3_dir)
        td3_loaded: dict[str, dict[str, Any]] = {}
        timezone_kinds: set[str] = set()
        for spec in selected:
            loaded = load_td3_candidate_histories(
                protocol=source.protocol,
                td3_dir=source.td3_dir,
                base_candidate=spec["base_candidate"],
                cap_label=spec["cap_label"],
                repo_root=repo,
                external_root=external,
            )
            td3_loaded[loaded["strategy_name"]] = loaded
            inventory_rows.append(loaded["inventory"])
            lineage_rows.extend(loaded["lineage"])
            timezone_kinds.add(loaded["timezone_kind"])

        td3_indices = [item["canonical_index"] for item in td3_loaded.values()]
        canonical_td3 = td3_indices[0]
        if not all(index.equals(canonical_td3) for index in td3_indices):
            raise ValueError(f"{source.protocol}: selected TD3 strategies do not share one OOS index.")

        benchmark_loaded: dict[str, dict[str, Any]] = {}
        for path in sorted((source.benchmark_dir / "histories").glob("*_history.csv")):
            loaded = load_benchmark_history(
                protocol=source.protocol,
                path=path,
                repo_root=repo,
                external_root=external,
            )
            benchmark_loaded[loaded["strategy_name"]] = loaded
            inventory_rows.append(loaded["inventory"])
            lineage_rows.extend(loaded["lineage"])
            timezone_kinds.add(loaded["timezone_kind"])

        if len(td3_loaded) != EXPECTED_TD3_STRATEGIES:
            raise ValueError(
                f"{source.protocol}: expected {EXPECTED_TD3_STRATEGIES} TD3 strategies, "
                f"found {len(td3_loaded)}."
            )
        if len(benchmark_loaded) != EXPECTED_BENCHMARKS:
            raise ValueError(
                f"{source.protocol}: expected {EXPECTED_BENCHMARKS} benchmarks, "
                f"found {len(benchmark_loaded)}."
            )
        if len(timezone_kinds) != 1:
            raise ValueError(
                f"{source.protocol}: incompatible naive/aware timestamp sources: "
                f"{sorted(timezone_kinds)}"
            )

        common = exact_common_index(
            canonical_td3,
            [item["canonical_index"] for item in benchmark_loaded.values()],
            expected_observations=expected_observations,
            label=source.protocol,
        )
        common_indices[source.protocol] = common

        protocol_histories: list[pd.DataFrame] = []
        protocol_seed_returns: list[pd.DataFrame] = []
        protocol_metrics: list[dict[str, Any]] = []
        for loaded in td3_loaded.values():
            aligned, seed_returns, seed_diagnostics = align_td3_candidate(
                loaded,
                common,
            )
            protocol_histories.append(aligned)
            protocol_seed_returns.append(seed_returns)
            protocol_metrics.append(
                compute_aligned_metrics(
                    aligned,
                    protocol=source.protocol,
                    strategy_name=loaded["strategy_name"],
                    strategy_type="TD3",
                    base_candidate=loaded["base_candidate"],
                    cap_label=loaded["cap_label"],
                    stability=seed_diagnostics,
                )
            )

        for loaded in benchmark_loaded.values():
            aligned = align_benchmark_history(loaded, common)
            protocol_histories.append(aligned)
            protocol_metrics.append(
                compute_aligned_metrics(
                    aligned,
                    protocol=source.protocol,
                    strategy_name=loaded["strategy_name"],
                    strategy_type="benchmark",
                    base_candidate=pd.NA,
                    cap_label=pd.NA,
                    stability=None,
                )
            )

        histories = pd.concat(protocol_histories, ignore_index=True, sort=False)
        seed_returns = pd.concat(protocol_seed_returns, ignore_index=True, sort=False)
        metrics = pd.DataFrame(protocol_metrics)
        ranking = score_aligned_universe(metrics)
        robust = ranking[
            [
                "protocol",
                "strategy_name",
                "strategy_type",
                *SCORE_COMPONENT_COLUMNS,
                "rank_robust",
                "rank_mandate_aware",
            ]
        ].copy()
        pairwise = build_td3_benchmark_deltas(ranking)

        constraint_input = ranking.copy()
        constraint_input["strategy_type"] = constraint_input["strategy_type"].str.lower()
        constraint_input["drawdown_severity"] = constraint_input["max_drawdown"].abs()
        constraints = build_constraint_pass_fail_matrix(constraint_input)
        feasible = build_feasible_strategy_rankings(constraints)
        standard = build_standard_metric_rankings(constraint_input)
        pareto, dominated = build_pareto_tables(constraint_input)
        profile_scores = score_strategies_for_profiles(constraint_input)
        profile_rankings = build_profile_rankings(profile_scores)
        profile_winners = build_profile_winners(profile_rankings)
        for frame in [constraints, feasible, standard, pareto, dominated, profile_rankings, profile_winners]:
            frame.insert(0, "protocol", source.protocol)

        _update_inventory_alignment(inventory_rows, source.protocol, common)
        all_histories.append(histories)
        all_seed_returns.append(seed_returns)
        all_metrics.append(metrics)
        all_rankings.append(ranking)
        all_robust.append(robust)
        all_pairwise.append(pairwise)
        all_constraints.append(constraints)
        all_feasible.append(feasible)
        all_profiles.append(profile_rankings)
        all_profile_winners.append(profile_winners)
        all_pareto.append(pareto)
        all_dominated.append(dominated)
        all_standard_rankings.append(standard)

        protocol_metadata[source.protocol] = {
            "cash_bps": source.cash_bps,
            "cash_return_assumption": source.cash_return_assumption,
            "n_td3_strategies": len(td3_loaded),
            "n_benchmarks": len(benchmark_loaded),
            "n_strategies": len(metrics),
            "common_observations": len(common),
            "common_start_date": common.min().date().isoformat(),
            "common_end_date": common.max().date().isoformat(),
            "frequency": "weekly Friday",
            "timezone_kind": next(iter(timezone_kinds)),
            "selected_td3": selected,
        }

    histories = pd.concat(all_histories, ignore_index=True, sort=False)
    seed_returns = pd.concat(all_seed_returns, ignore_index=True, sort=False)
    metrics = pd.concat(all_metrics, ignore_index=True, sort=False)
    rankings = pd.concat(all_rankings, ignore_index=True, sort=False)
    robust_scores = pd.concat(all_robust, ignore_index=True, sort=False)
    pairwise = pd.concat(all_pairwise, ignore_index=True, sort=False)
    constraints = pd.concat(all_constraints, ignore_index=True, sort=False)
    feasible = pd.concat(all_feasible, ignore_index=True, sort=False)
    profiles = pd.concat(all_profiles, ignore_index=True, sort=False)
    profile_winners = pd.concat(all_profile_winners, ignore_index=True, sort=False)
    pareto = pd.concat(all_pareto, ignore_index=True, sort=False)
    dominated = pd.concat(all_dominated, ignore_index=True, sort=False)
    standard_rankings = pd.concat(all_standard_rankings, ignore_index=True, sort=False)
    inventory = pd.DataFrame(inventory_rows)

    audit = build_old_new_audit(sources, rankings, constraints, profile_winners, pareto)
    statistical_reference = load_statistical_reference(sources)
    methodology = build_methodology_metadata(
        generated_at=generated_at,
        git_commit=git_commit,
        expected_observations=expected_observations,
        protocol_metadata=protocol_metadata,
    )

    outputs = write_aligned_outputs(
        output=output,
        common_indices=common_indices,
        inventory=inventory,
        lineage_rows=lineage_rows,
        histories=histories,
        seed_returns=seed_returns,
        metrics=metrics,
        rankings=rankings,
        robust_scores=robust_scores,
        pairwise=pairwise,
        constraints=constraints,
        feasible=feasible,
        profiles=profiles,
        profile_winners=profile_winners,
        pareto=pareto,
        dominated=dominated,
        standard_rankings=standard_rankings,
        audit=audit,
        statistical_reference=statistical_reference,
        methodology=methodology,
        protocol_metadata=protocol_metadata,
    )
    validation = validate_output_directory(output, expected_observations=expected_observations)
    validation_path = output / "metadata/validation_summary.json"
    validation_path.write_text(json.dumps(_json_safe(validation), indent=2), encoding="utf-8")
    outputs["validation_summary"] = str(validation_path)
    return {
        "output_dir": str(output),
        "outputs": outputs,
        "validation": validation,
        "rankings": rankings,
        "metrics": metrics,
        "constraints": constraints,
        "profiles": profiles,
        "profile_winners": profile_winners,
        "pareto": pareto,
        "audit": audit,
    }


def protocol_sources(repo: Path, external: Path) -> list[ProtocolSources]:
    """Return source locations without embedding machine-local paths in code."""
    return [
        ProtocolSources(
            protocol="zero_cash",
            td3_dir=repo / "outputs/tables/final_corrected_limited_td3_60ep_10seeds",
            benchmark_dir=external / "final_corrected_zero_cash_benchmark_comparison/benchmarks",
            old_comparison_dir=external / "final_corrected_zero_cash_benchmark_comparison",
            old_constraint_dir=external / "final_corrected_zero_cash_constraint_pareto",
            old_mandate_dir=external / "final_corrected_zero_cash_mandate_profile_comparison",
            statistical_dir=external / "final_corrected_zero_cash_statistical_validation",
            wrc_dir=external / "final_corrected_zero_cash_white_reality_check",
            cash_bps=0.0,
            cash_return_assumption="synthetic zero return",
        ),
        ProtocolSources(
            protocol="bil_cash",
            td3_dir=external / "final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds",
            benchmark_dir=external / "final_corrected_bil_cash_benchmark_comparison/benchmarks",
            old_comparison_dir=external / "final_corrected_bil_cash_benchmark_comparison",
            old_constraint_dir=external / "final_corrected_bil_cash_constraint_pareto",
            old_mandate_dir=external / "final_corrected_bil_cash_mandate_profile_comparison",
            statistical_dir=external / "final_corrected_bil_cash_statistical_validation",
            wrc_dir=external / "final_corrected_bil_cash_white_reality_check",
            cash_bps=2.0,
            cash_return_assumption="BIL proxy return",
        ),
    ]


def normalize_history_dates(frame: pd.DataFrame, source: str | Path) -> tuple[pd.DataFrame, str]:
    """Normalize one history to sorted naive UTC dates and reject ambiguity."""
    if DATE_COLUMN not in frame.columns:
        raise ValueError(f"{source}: missing {DATE_COLUMN!r} column.")
    raw = frame[DATE_COLUMN]
    try:
        parsed = pd.to_datetime(raw, errors="raise", utc=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: unparseable or mixed-zone timestamps.") from exc
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype) and parsed.dtype == object:
        raise ValueError(f"{source}: mixed or incompatible timezone values.")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        timezone_kind = "aware"
        normalized = parsed.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    else:
        timezone_kind = "naive"
        normalized = parsed.dt.normalize()
    result = frame.copy()
    result[DATE_COLUMN] = normalized
    if result[DATE_COLUMN].duplicated().any():
        duplicates = result.loc[result[DATE_COLUMN].duplicated(), DATE_COLUMN].dt.date.astype(str).tolist()
        raise ValueError(f"{source}: duplicate dates: {duplicates[:5]}")
    result = result.sort_values(DATE_COLUMN).reset_index(drop=True)
    if not result[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError(f"{source}: dates are not chronological after sorting.")
    if not (result[DATE_COLUMN].dt.dayofweek == 4).all():
        raise ValueError(f"{source}: expected weekly Friday observations.")
    return result, timezone_kind


def exact_common_index(
    canonical_td3: pd.DatetimeIndex,
    other_indices: Iterable[pd.DatetimeIndex],
    *,
    expected_observations: int,
    label: str,
) -> pd.DatetimeIndex:
    """Intersect exact timestamps and require the full canonical TD3 index."""
    common = canonical_td3.copy()
    for index in other_indices:
        common = common.intersection(index, sort=False)
    common = common.sort_values()
    if len(common) != expected_observations:
        raise ValueError(
            f"{label}: expected {expected_observations} common dates, found {len(common)}."
        )
    if not common.equals(canonical_td3):
        missing = canonical_td3.difference(common)
        extra = common.difference(canonical_td3)
        raise ValueError(
            f"{label}: common index does not equal canonical TD3 index; "
            f"missing={list(missing[:5])}, extra={list(extra[:5])}."
        )
    if common.has_duplicates:
        raise ValueError(f"{label}: common index contains duplicates.")
    return common


def load_td3_candidate_histories(
    *,
    protocol: str,
    td3_dir: Path,
    base_candidate: str,
    cap_label: str,
    repo_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    """Load 4-fold x 10-seed histories and validate full-OOS seed indices."""
    directory = td3_dir / "per_candidate" / base_candidate
    pattern = f"F*_{base_candidate}_cap_{cap_label}_seed_*/test_policy_history.csv"
    paths = sorted(directory.glob(pattern))
    if len(paths) != 40:
        raise ValueError(
            f"{protocol} {base_candidate} cap {cap_label}: expected 40 histories, found {len(paths)}."
        )
    frames: list[pd.DataFrame] = []
    lineage: list[dict[str, Any]] = []
    timezone_kinds: set[str] = set()
    rows_per_file: list[int] = []
    for path in paths:
        match = re.match(r"^F(?P<fold>\d+)_.*_seed_(?P<seed>\d+)$", path.parent.name)
        if match is None:
            raise ValueError(f"Cannot parse fold/seed from {path.parent.name}")
        frame = pd.read_csv(path)
        frame, timezone_kind = normalize_history_dates(frame, path)
        validate_history_columns(frame, path)
        frame["fold"] = int(match.group("fold"))
        frame["seed"] = int(match.group("seed"))
        frame = add_row_diagnostics(frame, path)
        frames.append(frame)
        rows_per_file.append(len(frame))
        timezone_kinds.add(timezone_kind)
        lineage.append(
            source_lineage_record(
                protocol=protocol,
                strategy_name=f"{base_candidate}_cap_{cap_label}",
                strategy_type="TD3",
                path=path,
                frame=frame,
                repo_root=repo_root,
                external_root=external_root,
            )
        )
    if len(timezone_kinds) != 1:
        raise ValueError(f"{protocol} {base_candidate}: incompatible timezone kinds.")
    stacked = pd.concat(frames, ignore_index=True, sort=False)
    seed_indices: dict[int, pd.DatetimeIndex] = {}
    for seed, group in stacked.groupby("seed", sort=True):
        dates = pd.DatetimeIndex(group[DATE_COLUMN].sort_values(), name=DATE_COLUMN)
        if dates.has_duplicates:
            raise ValueError(f"{protocol} {base_candidate}: seed {seed} has duplicate OOS dates.")
        seed_indices[int(seed)] = dates
    if sorted(seed_indices) != EXPECTED_SEEDS:
        raise ValueError(
            f"{protocol} {base_candidate}: expected seeds {EXPECTED_SEEDS}, found {sorted(seed_indices)}."
        )
    canonical = seed_indices[EXPECTED_SEEDS[0]]
    if not all(index.equals(canonical) for index in seed_indices.values()):
        raise ValueError(f"{protocol} {base_candidate}: seed OOS indices are not identical.")
    date_counts = stacked.groupby(DATE_COLUMN).size()
    if not (date_counts == len(EXPECTED_SEEDS)).all():
        raise ValueError(f"{protocol} {base_candidate}: each date must contain ten seed observations.")
    strategy_name = f"{base_candidate}_cap_{cap_label}"
    inventory = {
        "protocol": protocol,
        "strategy_name": strategy_name,
        "strategy_type": "TD3",
        "source_pattern_original": str(directory / pattern),
        "portable_source_hint": _portable_path(directory, repo_root, external_root) + f"/{pattern}",
        "source_files": len(paths),
        "source_rows": int(len(stacked)),
        "source_unique_dates": int(len(canonical)),
        "source_start_date": canonical.min().date().isoformat(),
        "source_end_date": canonical.max().date().isoformat(),
        "source_rows_per_file_min": min(rows_per_file),
        "source_rows_per_file_max": max(rows_per_file),
        "replications_per_date_min": int(date_counts.min()),
        "replications_per_date_max": int(date_counts.max()),
        "return_column": RETURN_COLUMN,
        "return_semantics": "net return after asset-specific transaction costs",
        "history_semantics": "returns, weights, turnover, costs, equity and drawdown",
        "timezone_kind": next(iter(timezone_kinds)),
        "duplicate_dates": 0,
        "missing_returns": 0,
        "source_set_sha256": _hash_set([row["sha256"] for row in lineage]),
    }
    return {
        "protocol": protocol,
        "strategy_name": strategy_name,
        "base_candidate": base_candidate,
        "cap_label": cap_label,
        "stacked": stacked,
        "canonical_index": canonical,
        "timezone_kind": next(iter(timezone_kinds)),
        "inventory": inventory,
        "lineage": lineage,
    }


def load_benchmark_history(
    *,
    protocol: str,
    path: Path,
    repo_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    """Load and validate one deterministic benchmark history."""
    frame = pd.read_csv(path)
    frame, timezone_kind = normalize_history_dates(frame, path)
    validate_history_columns(frame, path)
    frame = add_row_diagnostics(frame, path)
    index = pd.DatetimeIndex(frame[DATE_COLUMN], name=DATE_COLUMN)
    strategy_name = path.name.removesuffix("_history.csv")
    lineage = [
        source_lineage_record(
            protocol=protocol,
            strategy_name=strategy_name,
            strategy_type="benchmark",
            path=path,
            frame=frame,
            repo_root=repo_root,
            external_root=external_root,
        )
    ]
    inventory = {
        "protocol": protocol,
        "strategy_name": strategy_name,
        "strategy_type": "benchmark",
        "source_pattern_original": str(path),
        "portable_source_hint": _portable_path(path, repo_root, external_root),
        "source_files": 1,
        "source_rows": int(len(frame)),
        "source_unique_dates": int(len(index)),
        "source_start_date": index.min().date().isoformat(),
        "source_end_date": index.max().date().isoformat(),
        "source_rows_per_file_min": int(len(frame)),
        "source_rows_per_file_max": int(len(frame)),
        "replications_per_date_min": 1,
        "replications_per_date_max": 1,
        "return_column": RETURN_COLUMN,
        "return_semantics": "net return after asset-specific transaction costs",
        "history_semantics": "returns, weights, turnover, costs, equity and drawdown",
        "timezone_kind": timezone_kind,
        "duplicate_dates": 0,
        "missing_returns": 0,
        "source_set_sha256": lineage[0]["sha256"],
    }
    return {
        "protocol": protocol,
        "strategy_name": strategy_name,
        "frame": frame,
        "canonical_index": index,
        "timezone_kind": timezone_kind,
        "inventory": inventory,
        "lineage": lineage,
    }


def validate_history_columns(frame: pd.DataFrame, source: str | Path) -> None:
    """Require net returns, financial diagnostics, and asset-specific cost mode."""
    required = [
        RETURN_COLUMN,
        "turnover",
        "transaction_cost",
        "transaction_cost_mode",
        *WEIGHT_COLUMNS,
        *ASSET_TURNOVER_COLUMNS,
        *ASSET_COST_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source}: missing required history columns: {missing}")
    numeric = frame[[RETURN_COLUMN, "turnover", "transaction_cost", *WEIGHT_COLUMNS]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if numeric.isna().any().any():
        raise ValueError(f"{source}: missing/non-numeric values in required history columns.")
    if set(frame["transaction_cost_mode"].dropna().astype(str)) != {"asset_specific"}:
        raise ValueError(f"{source}: history is not asset-specific-cost consistent.")


def add_row_diagnostics(frame: pd.DataFrame, source: str | Path) -> pd.DataFrame:
    """Recompute row-level concentration diagnostics from stored weights."""
    result = frame.copy()
    weights = result[WEIGHT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{source}: portfolio weights do not sum to one.")
    if (weights < -1e-10).any().any():
        raise ValueError(f"{source}: negative long-only weights detected.")
    result["diagnostic_max_weight"] = weights.max(axis=1)
    result["diagnostic_effective_assets"] = 1.0 / weights.pow(2).sum(axis=1)
    return result


def align_td3_candidate(
    loaded: dict[str, Any],
    common_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build one 228-date mean history after validating every full-OOS seed."""
    stacked = loaded["stacked"].copy()
    seed_rows: list[pd.DataFrame] = []
    seed_sharpes: list[float] = []
    seed_drawdowns: list[float] = []
    seed_dsrs: list[float] = []
    for seed, group in stacked.groupby("seed", sort=True):
        group = group.sort_values(DATE_COLUMN)
        index = pd.DatetimeIndex(group[DATE_COLUMN], name=DATE_COLUMN)
        if not index.equals(common_index):
            raise ValueError(
                f"{loaded['protocol']} {loaded['strategy_name']} seed {seed}: "
                "index does not equal the common index."
            )
        returns = pd.Series(group[RETURN_COLUMN].to_numpy(dtype=float), index=index)
        seed_sharpes.append(sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR))
        seed_drawdowns.append(max_drawdown(returns))
        seed_dsrs.append(
            compute_deflated_sharpe_ratio(
                returns,
                n_trials=25,
                periods_per_year=PERIODS_PER_YEAR,
            )
        )
        seed_rows.append(
            pd.DataFrame(
                {
                    "protocol": loaded["protocol"],
                    "strategy_name": loaded["strategy_name"],
                    "seed": int(seed),
                    DATE_COLUMN: common_index,
                    RETURN_COLUMN: returns.to_numpy(dtype=float),
                }
            )
        )
    numeric = [column for column in HISTORY_NUMERIC_COLUMNS if column in stacked.columns]
    averaged = stacked.groupby(DATE_COLUMN, sort=True)[numeric].mean()
    if not averaged.index.equals(common_index):
        raise ValueError(f"{loaded['strategy_name']}: averaged TD3 index mismatch.")
    aligned = reset_aligned_equity(averaged.reset_index())
    aligned.insert(0, "strategy_type", "TD3")
    aligned.insert(0, "strategy_name", loaded["strategy_name"])
    aligned.insert(0, "protocol", loaded["protocol"])
    aligned["transaction_cost_mode"] = "asset_specific"
    seed_returns = pd.concat(seed_rows, ignore_index=True)
    stability = {
        "std_sharpe": float(np.std(seed_sharpes, ddof=0)),
        "worst_max_drawdown": float(np.min(seed_drawdowns)),
        "dsr_score": float(np.median(seed_dsrs)),
        "median_aligned_seed_dsr_n25": float(np.median(seed_dsrs)),
        "date_averaged_dsr_n25": compute_deflated_sharpe_ratio(
            pd.Series(aligned[RETURN_COLUMN].to_numpy(dtype=float), index=common_index),
            n_trials=25,
            periods_per_year=PERIODS_PER_YEAR,
        ),
        "dsr_method": "median_aligned_seed_history_n25",
        "n_aligned_seeds": len(seed_sharpes),
    }
    return aligned, seed_returns, stability


def align_benchmark_history(
    loaded: dict[str, Any],
    common_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Filter one benchmark by exact timestamp and reset aligned-window equity."""
    source = loaded["frame"].set_index(DATE_COLUMN)
    missing = common_index.difference(source.index)
    if len(missing):
        raise ValueError(f"{loaded['strategy_name']}: missing common dates: {list(missing[:5])}")
    aligned = source.loc[common_index].reset_index()
    if not pd.DatetimeIndex(aligned[DATE_COLUMN], name=DATE_COLUMN).equals(common_index):
        raise ValueError(f"{loaded['strategy_name']}: benchmark index mismatch after filtering.")
    aligned = reset_aligned_equity(aligned)
    aligned.insert(0, "strategy_type", "benchmark")
    aligned.insert(0, "strategy_name", loaded["strategy_name"])
    aligned.insert(0, "protocol", loaded["protocol"])
    return aligned


def reset_aligned_equity(frame: pd.DataFrame) -> pd.DataFrame:
    """Reset NAV and drawdown at the first common date without altering returns."""
    result = frame.copy()
    returns = pd.to_numeric(result[RETURN_COLUMN], errors="raise")
    result["portfolio_value"] = INITIAL_VALUE * (1.0 + returns).cumprod()
    result["drawdown"] = result["portfolio_value"] / result["portfolio_value"].cummax() - 1.0
    return result


def compute_aligned_metrics(
    history: pd.DataFrame,
    *,
    protocol: str,
    strategy_name: str,
    strategy_type: str,
    base_candidate: Any,
    cap_label: Any,
    stability: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute all comparison inputs from one exact aligned history."""
    index = pd.DatetimeIndex(history[DATE_COLUMN], name=DATE_COLUMN)
    returns = pd.Series(history[RETURN_COLUMN].to_numpy(dtype=float), index=index)
    row_max = pd.to_numeric(history["diagnostic_max_weight"], errors="raise")
    row_effective = pd.to_numeric(history["diagnostic_effective_assets"], errors="raise")
    result: dict[str, Any] = {
        "protocol": protocol,
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "base_candidate": base_candidate,
        "cap_label": cap_label,
        "transaction_cost_mode": "asset_specific",
        "n_observations": len(history),
        "start_date": index.min().date().isoformat(),
        "end_date": index.max().date().isoformat(),
        "frequency": "weekly Friday",
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(returns, PERIODS_PER_YEAR),
        "annualized_volatility": annualized_volatility(returns, PERIODS_PER_YEAR),
        "sharpe": sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR),
        "sortino": sortino_ratio(returns, periods_per_year=PERIODS_PER_YEAR),
        "calmar": calmar_ratio(returns, periods_per_year=PERIODS_PER_YEAR),
        "max_drawdown": max_drawdown(returns),
        "average_turnover": float(pd.to_numeric(history["turnover"], errors="raise").mean()),
        "total_transaction_cost": float(
            pd.to_numeric(history["transaction_cost"], errors="raise").sum()
        ),
        "mean_transaction_cost": float(
            pd.to_numeric(history["transaction_cost"], errors="raise").mean()
        ),
        "average_max_weight": float(row_max.mean()),
        "average_effective_number_of_assets": float(row_effective.mean()),
        "mean_cash_weight": float(pd.to_numeric(history["weight_CASH"], errors="raise").mean()),
        "cash_above_10_rate": float(
            (pd.to_numeric(history["weight_CASH"], errors="raise") > 0.10).mean()
        ),
        "mean_btc_weight": float(
            pd.to_numeric(history["weight_BTC-USD"], errors="raise").mean()
        ),
        "mean_cash_transaction_cost_contribution": float(
            pd.to_numeric(
                history["asset_transaction_cost_contribution_CASH"],
                errors="raise",
            ).mean()
        ),
        "mean_btc_transaction_cost_contribution": float(
            pd.to_numeric(
                history["asset_transaction_cost_contribution_BTC-USD"],
                errors="raise",
            ).mean()
        ),
    }
    if stability is None:
        result.update(
            {
                "std_sharpe": 0.0,
                "worst_max_drawdown": result["max_drawdown"],
                "dsr_score": compute_deflated_sharpe_ratio(
                    returns,
                    n_trials=25,
                    periods_per_year=PERIODS_PER_YEAR,
                ),
                "median_aligned_seed_dsr_n25": np.nan,
                "date_averaged_dsr_n25": compute_deflated_sharpe_ratio(
                    returns,
                    n_trials=25,
                    periods_per_year=PERIODS_PER_YEAR,
                ),
                "dsr_method": "aligned_single_history_n25",
                "n_aligned_seeds": 1,
            }
        )
    else:
        result.update(stability)
    return result


def score_aligned_universe(metrics: pd.DataFrame) -> pd.DataFrame:
    """Recompute cross-sectional scores and ranks inside one aligned protocol."""
    if metrics["protocol"].nunique() != 1:
        raise ValueError("Scores must be computed separately within one cash protocol.")
    if not (pd.to_numeric(metrics["n_observations"]) == metrics["n_observations"].iloc[0]).all():
        raise ValueError("Cannot score metrics with different observation counts.")
    scoring = metrics.rename(
        columns={
            "strategy_name": "strategy",
            "worst_max_drawdown": "worst_drawdown",
            "average_turnover": "turnover",
            "average_effective_number_of_assets": "effective_assets",
        }
    ).copy()
    scoring["type"] = scoring["strategy_type"].str.lower()
    scored = compute_composite_robust_score(scoring).rename(
        columns={
            "strategy": "strategy_name",
            "worst_drawdown": "worst_max_drawdown",
            "turnover": "average_turnover",
            "effective_assets": "average_effective_number_of_assets",
        }
    )
    scored["mandate_bucket"] = scored["max_drawdown"].apply(assign_drawdown_bucket)
    scored["recovery_required"] = scored["max_drawdown"].apply(calculate_recovery_required)
    scored["drawdown_multiplier"] = scored["max_drawdown"].apply(get_drawdown_multiplier)
    scored["mandate_aware_score"] = scored["robust_score"] * scored["drawdown_multiplier"]
    scored.loc[scored["mandate_bucket"] == "not_eligible", "mandate_aware_score"] = 0.0
    for rank_column, metric in [
        ("rank_mandate_aware", "mandate_aware_score"),
        ("rank_robust", "robust_score"),
        ("rank_sharpe", "sharpe"),
    ]:
        scored[rank_column] = scored[metric].rank(method="min", ascending=False).astype(int)
    return scored.sort_values(
        ["rank_mandate_aware", "rank_robust", "rank_sharpe", "strategy_name"]
    ).reset_index(drop=True)


def build_td3_benchmark_deltas(ranking: pd.DataFrame) -> pd.DataFrame:
    """Return all aligned TD3-versus-benchmark metric differences."""
    protocol = str(ranking["protocol"].iloc[0])
    td3 = ranking[ranking["strategy_type"] == "TD3"]
    benchmarks = ranking[ranking["strategy_type"] == "benchmark"]
    rows: list[dict[str, Any]] = []
    for _, candidate in td3.iterrows():
        for _, benchmark in benchmarks.iterrows():
            rows.append(
                {
                    "protocol": protocol,
                    "td3_strategy": candidate["strategy_name"],
                    "benchmark_strategy": benchmark["strategy_name"],
                    "n_observations": int(candidate["n_observations"]),
                    "start_date": candidate["start_date"],
                    "end_date": candidate["end_date"],
                    **{
                        f"delta_{metric}": float(candidate[metric]) - float(benchmark[metric])
                        for metric in [
                            "annualized_return",
                            "annualized_volatility",
                            "sharpe",
                            "max_drawdown",
                            "robust_score",
                            "mandate_aware_score",
                        ]
                    },
                }
            )
    return pd.DataFrame(rows)


def load_statistical_reference(sources: list[ProtocolSources]) -> pd.DataFrame:
    """Select existing V7/Trend pairwise and WRC rows without recomputation."""
    rows: list[dict[str, Any]] = []
    candidate = "V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80"
    benchmark = "trend_spy_cash_12p"
    for source in sources:
        pairwise_path = source.statistical_dir / "statistical_validation_pairwise_bootstrap.csv"
        wrc_path = source.wrc_dir / "white_reality_check_summary.csv"
        pairwise = pd.read_csv(pairwise_path)
        selected = pairwise[
            (pairwise["candidate"] == candidate)
            & (pairwise["benchmark"] == benchmark)
            & (pairwise["metric"] == "sharpe")
        ]
        if len(selected) != 1:
            raise ValueError(f"{source.protocol}: expected one V7/Trend pairwise ratio row.")
        wrc = pd.read_csv(wrc_path)
        selected_wrc = wrc[wrc["benchmark"] == benchmark]
        if len(selected_wrc) != 1:
            raise ValueError(f"{source.protocol}: expected one Trend WRC row.")
        row = selected.iloc[0]
        wrc_row = selected_wrc.iloc[0]
        rows.append(
            {
                "protocol": source.protocol,
                "candidate": candidate,
                "benchmark": benchmark,
                "estimator": "CAGR divided by annualized volatility (legacy field name: sharpe)",
                "candidate_estimate": row["candidate_estimate"],
                "benchmark_estimate": row["benchmark_estimate"],
                "bootstrap_mean_delta": row["mean_delta"],
                "lower_5pct_delta": row["lower_5pct_delta"],
                "upper_95pct_delta": row["upper_95pct_delta"],
                "probability_candidate_beats": row["probability_candidate_beats"],
                "n_aligned_periods": int(row["n_aligned_periods"]),
                "n_bootstrap": int(row["n_bootstrap"]),
                "block_size": int(row["block_size"]),
                "wrc_p_value": wrc_row["p_value"],
                "wrc_n_candidates": int(wrc_row["n_candidates"]),
                "wrc_block_length": int(wrc_row["block_length"]),
                "wrc_n_bootstrap": int(wrc_row["n_bootstrap"]),
                "pairwise_source_sha256": _sha256(pairwise_path),
                "wrc_source_sha256": _sha256(wrc_path),
            }
        )
    return pd.DataFrame(rows)


def build_old_new_audit(
    sources: list[ProtocolSources],
    rankings: pd.DataFrame,
    constraints: pd.DataFrame,
    profile_winners: pd.DataFrame,
    pareto: pd.DataFrame,
) -> dict[str, pd.DataFrame | str]:
    """Compare legacy non-aligned outputs with the aligned package."""
    metric_rows: list[pd.DataFrame] = []
    constraint_rows: list[pd.DataFrame] = []
    winner_rows: list[pd.DataFrame] = []
    pareto_rows: list[pd.DataFrame] = []
    for source in sources:
        prefix = f"final_corrected_{source.protocol}"
        old_ranking_path = source.old_comparison_dir / f"{prefix}_combined_ranking.csv"
        old = pd.read_csv(old_ranking_path)
        new = rankings[rankings["protocol"] == source.protocol].copy()
        compare = old.merge(new, on="strategy_name", suffixes=("_old", "_aligned"), how="outer")
        if "protocol" in compare:
            compare["protocol"] = source.protocol
        else:
            compare.insert(0, "protocol", source.protocol)
        for metric in [
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
            "robust_score",
            "mandate_aware_score",
        ]:
            compare[f"delta_{metric}"] = (
                pd.to_numeric(compare[f"{metric}_aligned"], errors="coerce")
                - pd.to_numeric(compare[f"{metric}_old"], errors="coerce")
            )
        for rank in ["rank_mandate_aware", "rank_robust", "rank_sharpe"]:
            compare[f"change_{rank}"] = (
                pd.to_numeric(compare[f"{rank}_aligned"], errors="coerce")
                - pd.to_numeric(compare[f"{rank}_old"], errors="coerce")
            )
        compare["change_explanation"] = (
            np.where(
                compare["strategy_type_old"].astype(str).str.upper() == "TD3",
                "TD3 metrics are recomputed from ten full-OOS seed histories with identical 228-date indices and their date-average; nonlinear metrics therefore differ from legacy means of fold/seed metrics. Robust and mandate scores are renormalized in the aligned universe.",
                "Benchmark metrics are recomputed on the exact 228-date OOS window rather than full available history. Robust and mandate scores are renormalized in the aligned universe.",
            )
        )
        metric_rows.append(compare)

        old_constraints = pd.read_csv(source.old_constraint_dir / "constraint_pass_fail_matrix.csv")
        new_constraints = constraints[constraints["protocol"] == source.protocol]
        constraint_compare = old_constraints.merge(
            new_constraints,
            on=["profile", "strategy_name"],
            suffixes=("_old", "_aligned"),
            how="outer",
        )
        if "protocol" in constraint_compare:
            constraint_compare["protocol"] = source.protocol
        else:
            constraint_compare.insert(0, "protocol", source.protocol)
        constraint_compare["feasibility_changed"] = (
            constraint_compare["feasible_old"].astype(str)
            != constraint_compare["feasible_aligned"].astype(str)
        )
        constraint_rows.append(constraint_compare)

        old_winners = pd.read_csv(source.old_mandate_dir / "mandate_profile_winners.csv")
        new_winners = profile_winners[profile_winners["protocol"] == source.protocol]
        winner_compare = old_winners.merge(
            new_winners,
            on="profile",
            suffixes=("_old", "_aligned"),
            how="outer",
        )
        if "protocol" in winner_compare:
            winner_compare["protocol"] = source.protocol
        else:
            winner_compare.insert(0, "protocol", source.protocol)
        winner_compare["overall_winner_changed"] = (
            winner_compare["overall_winner_old"].astype(str)
            != winner_compare["overall_winner_aligned"].astype(str)
        )
        winner_rows.append(winner_compare)

        old_pareto = pd.read_csv(source.old_constraint_dir / "pareto_frontier.csv")
        new_pareto = pareto[pareto["protocol"] == source.protocol]
        old_membership = old_pareto[["frontier_type", "strategy_name"]].drop_duplicates().assign(
            on_frontier_old=True
        )
        new_membership = new_pareto[["frontier_type", "strategy_name"]].drop_duplicates().assign(
            on_frontier_aligned=True
        )
        pareto_compare = old_membership.merge(
            new_membership,
            on=["frontier_type", "strategy_name"],
            how="outer",
        ).fillna(False)
        if "protocol" in pareto_compare:
            pareto_compare["protocol"] = source.protocol
        else:
            pareto_compare.insert(0, "protocol", source.protocol)
        pareto_compare["membership_changed"] = (
            pareto_compare["on_frontier_old"] != pareto_compare["on_frontier_aligned"]
        )
        pareto_rows.append(pareto_compare)

    metric_audit = pd.concat(metric_rows, ignore_index=True, sort=False)
    constraint_audit = pd.concat(constraint_rows, ignore_index=True, sort=False)
    winner_audit = pd.concat(winner_rows, ignore_index=True, sort=False)
    pareto_audit = pd.concat(pareto_rows, ignore_index=True, sort=False)
    summary = build_audit_summary(metric_audit, constraint_audit, winner_audit, pareto_audit)
    return {
        "metrics_and_ranking": metric_audit,
        "constraint_changes": constraint_audit,
        "mandate_winner_changes": winner_audit,
        "pareto_membership_changes": pareto_audit,
        "summary": summary,
    }


def build_audit_summary(
    metric_audit: pd.DataFrame,
    constraint_audit: pd.DataFrame,
    winner_audit: pd.DataFrame,
    pareto_audit: pd.DataFrame,
) -> str:
    """Summarize mechanical changes without favoring either strategy type."""
    lines = [
        "# Old versus aligned comparison audit",
        "",
        "All changes below result from exact-window metric recomputation and, for custom scores, "
        "cross-sectional renormalization within each aligned 19-strategy protocol universe.",
        "",
    ]
    for protocol in ["zero_cash", "bil_cash"]:
        rows = metric_audit[metric_audit["protocol"] == protocol]
        old_top = rows.sort_values("rank_mandate_aware_old").iloc[0]
        new_top = rows.sort_values("rank_mandate_aware_aligned").iloc[0]
        lines.extend(
            [
                f"## {protocol}",
                "",
                f"- Previous mandate-aware leader: `{old_top['strategy_name']}`.",
                f"- Aligned mandate-aware leader: `{new_top['strategy_name']}`.",
                f"- Hard-feasibility changes: {int(constraint_audit[(constraint_audit['protocol'] == protocol) & constraint_audit['feasibility_changed']].shape[0])} rows.",
                f"- Mandate-profile winner changes: {int(winner_audit[(winner_audit['protocol'] == protocol) & winner_audit['overall_winner_changed']].shape[0])} profiles.",
                f"- Pareto membership changes: {int(pareto_audit[(pareto_audit['protocol'] == protocol) & pareto_audit['membership_changed']].shape[0])} strategy/frontier rows.",
                "",
            ]
        )
    return "\n".join(lines)


def write_aligned_outputs(
    *,
    output: Path,
    common_indices: dict[str, pd.DatetimeIndex],
    inventory: pd.DataFrame,
    lineage_rows: list[dict[str, Any]],
    histories: pd.DataFrame,
    seed_returns: pd.DataFrame,
    metrics: pd.DataFrame,
    rankings: pd.DataFrame,
    robust_scores: pd.DataFrame,
    pairwise: pd.DataFrame,
    constraints: pd.DataFrame,
    feasible: pd.DataFrame,
    profiles: pd.DataFrame,
    profile_winners: pd.DataFrame,
    pareto: pd.DataFrame,
    dominated: pd.DataFrame,
    standard_rankings: pd.DataFrame,
    audit: dict[str, pd.DataFrame | str],
    statistical_reference: pd.DataFrame,
    methodology: dict[str, Any],
    protocol_metadata: dict[str, Any],
) -> dict[str, str]:
    """Write the minimal reconstructible aligned-result package."""
    date_index = pd.concat(
        [
            pd.DataFrame(
                {
                    "protocol": protocol,
                    DATE_COLUMN: index,
                    "observation_number": np.arange(1, len(index) + 1),
                }
            )
            for protocol, index in common_indices.items()
        ],
        ignore_index=True,
    )
    returns = histories[["protocol", "strategy_name", "strategy_type", DATE_COLUMN, RETURN_COLUMN]].copy()
    benchmark_returns = returns[returns["strategy_type"] == "benchmark"].copy()
    paths: dict[str, Path] = {
        "aligned_date_index": output / "alignment/aligned_date_index.csv",
        "alignment_inventory": output / "alignment/alignment_inventory.csv",
        "alignment_metadata": output / "alignment/alignment_metadata.json",
        "aligned_strategy_returns": output / "histories/aligned_strategy_returns.csv",
        "aligned_benchmark_returns": output / "histories/aligned_benchmark_returns.csv",
        "aligned_strategy_histories": output / "histories/aligned_strategy_histories.csv",
        "aligned_td3_seed_returns": output / "histories/aligned_td3_seed_returns.csv",
        "aligned_strategy_metrics": output / "metrics/aligned_strategy_metrics.csv",
        "aligned_benchmark_metrics": output / "metrics/aligned_benchmark_metrics.csv",
        "aligned_combined_ranking": output / "ranking/aligned_combined_ranking.csv",
        "aligned_robust_scores": output / "ranking/aligned_robust_scores.csv",
        "aligned_td3_vs_benchmarks": output / "ranking/aligned_td3_vs_benchmarks.csv",
        "aligned_statistical_reference": output / "ranking/aligned_statistical_reference.csv",
        "aligned_constraint_pass_fail_matrix": output / "mandates/aligned_constraint_pass_fail_matrix.csv",
        "aligned_feasible_strategy_rankings": output / "mandates/aligned_feasible_strategy_rankings.csv",
        "aligned_mandate_profile_rankings": output / "mandates/aligned_mandate_profile_rankings.csv",
        "aligned_mandate_profile_winners": output / "mandates/aligned_mandate_profile_winners.csv",
        "aligned_standard_metric_rankings": output / "ranking/aligned_standard_metric_rankings.csv",
        "aligned_pareto_frontier": output / "pareto/aligned_pareto_frontier.csv",
        "aligned_pareto_dominated": output / "pareto/aligned_pareto_dominated_strategies.csv",
        "old_vs_aligned_metrics": output / "audit/old_vs_aligned_metrics_and_ranking.csv",
        "mandate_pass_changes": output / "audit/mandate_pass_changes.csv",
        "mandate_winner_changes": output / "audit/mandate_winner_changes.csv",
        "pareto_membership_changes": output / "audit/pareto_membership_changes.csv",
        "audit_summary": output / "audit/old_vs_aligned_summary.md",
        "methodology": output / "metadata/methodology.json",
        "source_lineage": output / "metadata/source_lineage.json",
        "paper_macros": output / "paper/aligned_results_macros.tex",
        "paper_combined_table_rows": output / "paper/aligned_combined_table_rows.tex",
        "paper_statistical_table_rows": output / "paper/aligned_statistical_table_rows.tex",
    }
    date_index.to_csv(paths["aligned_date_index"], index=False, date_format="%Y-%m-%d")
    inventory.to_csv(paths["alignment_inventory"], index=False)
    returns.to_csv(paths["aligned_strategy_returns"], index=False, date_format="%Y-%m-%d")
    benchmark_returns.to_csv(paths["aligned_benchmark_returns"], index=False, date_format="%Y-%m-%d")
    histories.to_csv(paths["aligned_strategy_histories"], index=False, date_format="%Y-%m-%d")
    seed_returns.to_csv(paths["aligned_td3_seed_returns"], index=False, date_format="%Y-%m-%d")
    metrics.to_csv(paths["aligned_strategy_metrics"], index=False)
    metrics[metrics["strategy_type"] == "benchmark"].to_csv(
        paths["aligned_benchmark_metrics"], index=False
    )
    rankings.to_csv(paths["aligned_combined_ranking"], index=False)
    robust_scores.to_csv(paths["aligned_robust_scores"], index=False)
    pairwise.to_csv(paths["aligned_td3_vs_benchmarks"], index=False)
    statistical_reference.to_csv(paths["aligned_statistical_reference"], index=False)
    constraints.to_csv(paths["aligned_constraint_pass_fail_matrix"], index=False)
    feasible.to_csv(paths["aligned_feasible_strategy_rankings"], index=False)
    profiles.to_csv(paths["aligned_mandate_profile_rankings"], index=False)
    profile_winners.to_csv(paths["aligned_mandate_profile_winners"], index=False)
    standard_rankings.to_csv(paths["aligned_standard_metric_rankings"], index=False)
    pareto.to_csv(paths["aligned_pareto_frontier"], index=False)
    dominated.to_csv(paths["aligned_pareto_dominated"], index=False)
    assert isinstance(audit["metrics_and_ranking"], pd.DataFrame)
    assert isinstance(audit["constraint_changes"], pd.DataFrame)
    assert isinstance(audit["mandate_winner_changes"], pd.DataFrame)
    assert isinstance(audit["pareto_membership_changes"], pd.DataFrame)
    audit["metrics_and_ranking"].to_csv(paths["old_vs_aligned_metrics"], index=False)
    audit["constraint_changes"].to_csv(paths["mandate_pass_changes"], index=False)
    audit["mandate_winner_changes"].to_csv(paths["mandate_winner_changes"], index=False)
    audit["pareto_membership_changes"].to_csv(paths["pareto_membership_changes"], index=False)
    paths["audit_summary"].write_text(str(audit["summary"]), encoding="utf-8")
    paths["methodology"].write_text(json.dumps(_json_safe(methodology), indent=2), encoding="utf-8")
    paths["source_lineage"].write_text(
        json.dumps(_json_safe({"sources": lineage_rows}), indent=2), encoding="utf-8"
    )
    paper_macros, paper_rows, statistical_rows = build_paper_latex_artifacts(
        rankings=rankings,
        constraints=constraints,
        profile_winners=profile_winners,
        pareto=pareto,
        statistical_reference=statistical_reference,
    )
    paths["paper_macros"].write_text(paper_macros, encoding="utf-8")
    paths["paper_combined_table_rows"].write_text(paper_rows, encoding="utf-8")
    paths["paper_statistical_table_rows"].write_text(statistical_rows, encoding="utf-8")
    alignment_metadata = {
        "method": "exact timestamp intersection anchored to canonical TD3 OOS index",
        "no_tail_selection": True,
        "no_forward_fill": True,
        "no_extrapolation": True,
        "no_source_overwrite": True,
        "protocols": protocol_metadata,
    }
    paths["alignment_metadata"].write_text(
        json.dumps(_json_safe(alignment_metadata), indent=2), encoding="utf-8"
    )
    return {key: str(value) for key, value in paths.items()}


def build_paper_latex_artifacts(
    *,
    rankings: pd.DataFrame,
    constraints: pd.DataFrame,
    profile_winners: pd.DataFrame,
    pareto: pd.DataFrame,
    statistical_reference: pd.DataFrame,
) -> tuple[str, str, str]:
    """Generate manuscript numbers directly from the validated CSV frames."""
    selected: dict[str, dict[str, pd.Series]] = {}
    for protocol in ["zero_cash", "bil_cash"]:
        rows = rankings[rankings["protocol"] == protocol]
        selected[protocol] = {
            "gld": rows[rows["strategy_name"] == "BuyHold_GLD"].iloc[0],
            "v7": rows[
                rows["strategy_name"]
                == "V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80"
            ].iloc[0],
            "trend": rows[rows["strategy_name"] == "trend_spy_cash_12p"].iloc[0],
        }
    macros = {
        "AlignedWeeks": int(rankings["n_observations"].iloc[0]),
        "AlignedStartDate": str(rankings["start_date"].iloc[0]),
        "AlignedEndDate": str(rankings["end_date"].iloc[0]),
        "ZeroBestTDThreeMandateRank": int(selected["zero_cash"]["v7"]["rank_mandate_aware"]),
        "BilBestTDThreeMandateRank": int(selected["bil_cash"]["v7"]["rank_mandate_aware"]),
        "ZeroBestTDThreeSharpeRank": int(selected["zero_cash"]["v7"]["rank_sharpe"]),
        "BilBestTDThreeSharpeRank": int(selected["bil_cash"]["v7"]["rank_sharpe"]),
        "ZeroTDThreeSharpe": f"{selected['zero_cash']['v7']['sharpe']:.4f}",
        "BilTDThreeSharpe": f"{selected['bil_cash']['v7']['sharpe']:.4f}",
        "ZeroTrendSharpe": f"{selected['zero_cash']['trend']['sharpe']:.4f}",
        "BilTrendSharpe": f"{selected['bil_cash']['trend']['sharpe']:.4f}",
        "AggressiveFeasibleCount": int(
            constraints[
                (constraints["protocol"] == "zero_cash")
                & (constraints["profile"] == "aggressive")
                & constraints["feasible"]
            ].shape[0]
        ),
        "TDThreeFeasibleCount": int(
            constraints[(constraints["strategy_type"] == "td3") & constraints["feasible"]].shape[0]
        ),
        "ZeroParetoTDThreeCount": int(
            pareto[
                (pareto["protocol"] == "zero_cash")
                & (pareto["frontier_type"] == "reduced")
                & (pareto["strategy_type"] == "td3")
            ]["strategy_name"].nunique()
        ),
        "BilParetoTDThreeCount": int(
            pareto[
                (pareto["protocol"] == "bil_cash")
                & (pareto["frontier_type"] == "reduced")
                & (pareto["strategy_type"] == "td3")
            ]["strategy_name"].nunique()
        ),
    }
    for protocol, prefix in [("zero_cash", "Zero"), ("bil_cash", "Bil")]:
        stat = statistical_reference[statistical_reference["protocol"] == protocol].iloc[0]
        macros.update(
            {
                f"{prefix}BootstrapMeanDelta": f"{stat['bootstrap_mean_delta']:.4f}",
                f"{prefix}BootstrapLower": f"{stat['lower_5pct_delta']:.4f}",
                f"{prefix}BootstrapUpper": f"{stat['upper_95pct_delta']:.4f}",
                f"{prefix}BootstrapProbability": f"{stat['probability_candidate_beats']:.3f}",
                f"{prefix}WrcPValue": f"{stat['wrc_p_value']:.4f}",
            }
        )
    macro_text = "% Generated by scripts/build_paper_aligned_comparison.py; do not edit.\n"
    macro_text += "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    )
    macro_text += "\n"

    row_specs = [
        ("zero_cash", "gld", "\\cashzero{}", "BuyHold GLD", "Benchmark"),
        ("zero_cash", "v7", "\\cashzero{}", "V7 macro+GARCH cap 0.80", "TD3"),
        ("zero_cash", "trend", "\\cashzero{}", "Trend SPY/CASH", "Benchmark"),
        ("bil_cash", "gld", "\\cashbil{}", "BuyHold GLD", "Benchmark"),
        ("bil_cash", "v7", "\\cashbil{}", "V7 macro+GARCH cap 0.80", "TD3"),
        ("bil_cash", "trend", "\\cashbil{}", "Trend SPY/CASH", "Benchmark"),
    ]
    lines = ["% Generated by scripts/build_paper_aligned_comparison.py; do not edit."]
    for protocol, key, cash, label, strategy_type in row_specs:
        row = selected[protocol][key]
        lines.append(
            f"{cash} & {label} & {strategy_type} & {int(row['rank_mandate_aware'])} & "
            f"{row['annualized_return']:.4f} & {row['annualized_volatility']:.4f} & "
            f"{row['sharpe']:.4f} & {row['max_drawdown']:.4f} & "
            f"{row['mandate_aware_score']:.4f} & {row['robust_score']:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    stat_lines = ["% Generated from existing aligned bootstrap/WRC outputs; do not edit."]
    for protocol, cash in [("zero_cash", "\\cashzero{}"), ("bil_cash", "\\cashbil{}")]:
        stat = statistical_reference[statistical_reference["protocol"] == protocol].iloc[0]
        stat_lines.append(
            f"{cash} & TD3 V7 macro+GARCH cap 0.80 vs Trend SPY/CASH & "
            f"{stat['bootstrap_mean_delta']:.4f} & "
            f"[{stat['lower_5pct_delta']:.4f}, {stat['upper_95pct_delta']:.4f}] & "
            f"{stat['probability_candidate_beats']:.3f} & {stat['wrc_p_value']:.4f} & "
            "No statistical superiority claim is supported. \\\\"
        )
    stat_lines.append("\\bottomrule")
    return macro_text, "\n".join(lines) + "\n", "\n".join(stat_lines) + "\n"


def validate_output_directory(
    output_dir: str | Path,
    *,
    expected_observations: int = 228,
) -> dict[str, Any]:
    """Fail on any temporal, metric, score, mandate, Pareto or metadata mismatch."""
    output = Path(output_dir)
    dates = pd.read_csv(output / "alignment/aligned_date_index.csv", parse_dates=[DATE_COLUMN])
    returns = pd.read_csv(
        output / "histories/aligned_strategy_returns.csv", parse_dates=[DATE_COLUMN]
    )
    histories = pd.read_csv(
        output / "histories/aligned_strategy_histories.csv", parse_dates=[DATE_COLUMN]
    )
    metrics = pd.read_csv(output / "metrics/aligned_strategy_metrics.csv")
    ranking = pd.read_csv(output / "ranking/aligned_combined_ranking.csv")
    constraints = pd.read_csv(output / "mandates/aligned_constraint_pass_fail_matrix.csv")
    profiles = pd.read_csv(output / "mandates/aligned_mandate_profile_rankings.csv")
    pareto = pd.read_csv(output / "pareto/aligned_pareto_frontier.csv")
    alignment_metadata = json.loads(
        (output / "alignment/alignment_metadata.json").read_text(encoding="utf-8")
    )
    methodology = json.loads((output / "metadata/methodology.json").read_text(encoding="utf-8"))

    errors: list[str] = []
    strategy_counts: dict[str, int] = {}
    all_indices_identical = True
    duplicate_dates = 0
    missing_values = int(returns[RETURN_COLUMN].isna().sum())
    for protocol, protocol_dates in dates.groupby("protocol", sort=True):
        canonical = pd.DatetimeIndex(protocol_dates.sort_values(DATE_COLUMN)[DATE_COLUMN], name=DATE_COLUMN)
        if len(canonical) != expected_observations:
            errors.append(f"{protocol}: aligned date index has {len(canonical)} rows.")
        protocol_returns = returns[returns["protocol"] == protocol]
        strategy_counts[protocol] = protocol_returns["strategy_name"].nunique()
        for strategy_name, group in protocol_returns.groupby("strategy_name", sort=True):
            group_index = pd.DatetimeIndex(group.sort_values(DATE_COLUMN)[DATE_COLUMN], name=DATE_COLUMN)
            duplicate_dates += int(group[DATE_COLUMN].duplicated().sum())
            if not group_index.equals(canonical):
                all_indices_identical = False
                errors.append(f"{protocol} {strategy_name}: return index mismatch.")
        history_subset = histories[histories["protocol"] == protocol]
        if history_subset["strategy_name"].nunique() != strategy_counts[protocol]:
            errors.append(f"{protocol}: history/return strategy count mismatch.")

    metric_pass = _validate_metrics_against_histories(histories, metrics, errors)
    ranking_pass = _validate_ranking_against_metrics(metrics, ranking, errors)
    mandate_pass = _validate_mandates(ranking, constraints, profiles, errors)
    pareto_pass = _validate_pareto(ranking, pareto, errors)
    metadata_pass = True
    for protocol, metadata in alignment_metadata["protocols"].items():
        expected_rows = int(
            dates[dates["protocol"] == protocol].shape[0]
        )
        if metadata["common_observations"] != expected_rows:
            errors.append(f"{protocol}: alignment metadata observation mismatch.")
            metadata_pass = False
    if methodology.get("expected_observations") != expected_observations:
        errors.append("Methodology expected-observations mismatch.")
        metadata_pass = False
    if duplicate_dates:
        errors.append(f"Aligned histories contain {duplicate_dates} duplicate strategy dates.")
    if missing_values:
        errors.append(f"Aligned return table contains {missing_values} missing returns.")
    if not all_indices_identical:
        errors.append("Not all aligned strategy indices are identical.")
    if any(count != EXPECTED_TD3_STRATEGIES + EXPECTED_BENCHMARKS for count in strategy_counts.values()):
        errors.append(f"Unexpected strategy counts: {strategy_counts}")
    if errors:
        raise ValueError("Aligned comparison validation failed:\n- " + "\n- ".join(errors))
    first_protocol = sorted(dates["protocol"].unique())[0]
    first_dates = dates[dates["protocol"] == first_protocol].sort_values(DATE_COLUMN)
    return {
        "status": "PASS",
        "n_strategies_compared_per_protocol": strategy_counts,
        "common_observations": expected_observations,
        "common_start_date": first_dates[DATE_COLUMN].min().date().isoformat(),
        "common_end_date": first_dates[DATE_COLUMN].max().date().isoformat(),
        "frequency": "weekly Friday",
        "all_indices_identical": "PASS",
        "duplicate_dates": duplicate_dates,
        "missing_values": missing_values,
        "metrics_derived_from_aligned_histories": "PASS" if metric_pass else "FAIL",
        "ranking_derived_from_aligned_metrics": "PASS" if ranking_pass else "FAIL",
        "mandates_derived_from_aligned_metrics": "PASS" if mandate_pass else "FAIL",
        "pareto_derived_from_aligned_metrics": "PASS" if pareto_pass else "FAIL",
        "metadata_consistent_with_csv": "PASS" if metadata_pass else "FAIL",
    }


def _validate_metrics_against_histories(
    histories: pd.DataFrame,
    metrics: pd.DataFrame,
    errors: list[str],
) -> bool:
    passed = True
    for _, row in metrics.iterrows():
        history = histories[
            (histories["protocol"] == row["protocol"])
            & (histories["strategy_name"] == row["strategy_name"])
        ].sort_values(DATE_COLUMN)
        returns = pd.Series(pd.to_numeric(history[RETURN_COLUMN], errors="coerce").to_numpy())
        expected = {
            "cumulative_return": cumulative_return(returns),
            "annualized_return": annualized_return(returns, PERIODS_PER_YEAR),
            "annualized_volatility": annualized_volatility(returns, PERIODS_PER_YEAR),
            "sharpe": sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR),
            "sortino": sortino_ratio(returns, periods_per_year=PERIODS_PER_YEAR),
            "calmar": calmar_ratio(returns, periods_per_year=PERIODS_PER_YEAR),
            "max_drawdown": max_drawdown(returns),
            "average_turnover": pd.to_numeric(history["turnover"], errors="coerce").mean(),
            "mean_transaction_cost": pd.to_numeric(
                history["transaction_cost"], errors="coerce"
            ).mean(),
            "average_max_weight": pd.to_numeric(
                history["diagnostic_max_weight"], errors="coerce"
            ).mean(),
            "average_effective_number_of_assets": pd.to_numeric(
                history["diagnostic_effective_assets"], errors="coerce"
            ).mean(),
        }
        for metric, value in expected.items():
            if not np.isclose(float(row[metric]), float(value), rtol=1e-10, atol=1e-12):
                errors.append(f"{row['protocol']} {row['strategy_name']}: metric mismatch {metric}.")
                passed = False
    return passed


def _validate_ranking_against_metrics(
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    errors: list[str],
) -> bool:
    passed = True
    for protocol in sorted(metrics["protocol"].unique()):
        expected = score_aligned_universe(metrics[metrics["protocol"] == protocol].copy())
        actual = ranking[ranking["protocol"] == protocol].copy()
        merged = expected.merge(actual, on="strategy_name", suffixes=("_expected", "_actual"))
        for metric in [*SCORE_COMPONENT_COLUMNS, "rank_robust", "rank_mandate_aware", "rank_sharpe"]:
            left = pd.to_numeric(merged[f"{metric}_expected"], errors="coerce")
            right = pd.to_numeric(merged[f"{metric}_actual"], errors="coerce")
            if not np.allclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True):
                errors.append(f"{protocol}: ranking mismatch for {metric}.")
                passed = False
    return passed


def _validate_mandates(
    ranking: pd.DataFrame,
    constraints: pd.DataFrame,
    profiles: pd.DataFrame,
    errors: list[str],
) -> bool:
    passed = True
    for protocol in sorted(ranking["protocol"].unique()):
        source = ranking[ranking["protocol"] == protocol].copy()
        source["strategy_type"] = source["strategy_type"].str.lower()
        expected_constraints = build_constraint_pass_fail_matrix(source)
        actual_constraints = constraints[constraints["protocol"] == protocol]
        merged = expected_constraints.merge(
            actual_constraints,
            on=["profile", "strategy_name"],
            suffixes=("_expected", "_actual"),
        )
        for column in [
            "feasible",
            "max_drawdown_pass",
            "annualized_volatility_pass",
            "average_effective_number_of_assets_pass",
            "average_turnover_pass",
        ]:
            if not (
                merged[f"{column}_expected"].astype(bool).to_numpy()
                == merged[f"{column}_actual"].astype(bool).to_numpy()
            ).all():
                errors.append(f"{protocol}: constraint mismatch for {column}.")
                passed = False
        expected_profiles = build_profile_rankings(score_strategies_for_profiles(source))
        actual_profiles = profiles[profiles["protocol"] == protocol]
        profile_merge = expected_profiles.merge(
            actual_profiles,
            on=["profile", "strategy_name"],
            suffixes=("_expected", "_actual"),
        )
        if not np.allclose(
            profile_merge["profile_score_expected"],
            profile_merge["profile_score_actual"],
            rtol=1e-10,
            atol=1e-12,
        ):
            errors.append(f"{protocol}: mandate profile scores mismatch.")
            passed = False
    return passed


def _validate_pareto(
    ranking: pd.DataFrame,
    pareto: pd.DataFrame,
    errors: list[str],
) -> bool:
    passed = True
    for protocol in sorted(ranking["protocol"].unique()):
        source = ranking[ranking["protocol"] == protocol].copy()
        source["strategy_type"] = source["strategy_type"].str.lower()
        source["drawdown_severity"] = source["max_drawdown"].abs()
        expected, _ = build_pareto_tables(source)
        actual = pareto[pareto["protocol"] == protocol]
        expected_set = set(map(tuple, expected[["frontier_type", "strategy_name"]].to_numpy()))
        actual_set = set(map(tuple, actual[["frontier_type", "strategy_name"]].to_numpy()))
        if expected_set != actual_set:
            errors.append(f"{protocol}: Pareto membership mismatch.")
            passed = False
    return passed


def build_methodology_metadata(
    *,
    generated_at: str,
    git_commit: str | None,
    expected_observations: int,
    protocol_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Describe formulas, thresholds, lineage and experimental-unit caveats."""
    return {
        "generated_at_utc": generated_at,
        "runner": "scripts/build_paper_aligned_comparison.py",
        "module": "src.analysis.paper_aligned_comparison",
        "git_commit": git_commit,
        "expected_observations": expected_observations,
        "periods_per_year": PERIODS_PER_YEAR,
        "initial_value": INITIAL_VALUE,
        "date_alignment": {
            "canonical_index": "union of four TD3 OOS folds after exact per-seed validation",
            "operation": "timestamp intersection, never tail-row selection",
            "expected_frequency": "weekly Friday",
            "no_forward_fill": True,
            "no_extrapolation": True,
            "benchmark_warm_up": (
                "Benchmark signals and costs are retained from their original causal full histories; "
                "only reported rows are filtered to the common OOS index."
            ),
        },
        "td3_aggregation": {
            "per_seed": "four disjoint OOS folds concatenated to the same 228 timestamps",
            "reported_history": "arithmetic mean by date across ten aligned seeds",
            "seeds": EXPECTED_SEEDS,
            "independence_warning": (
                "Folds, seeds, candidates and histories share one market record and are not "
                "independent economic realizations."
            ),
        },
        "return_metrics": {
            "source": "src.backtest.evaluate_policy and src.backtest.performance_metrics",
            "return_column": RETURN_COLUMN,
            "risk_free_rate": 0.0,
        },
        "robust_score": {
            "weights": DEFAULT_COMPOSITE_WEIGHTS,
            "dsr_trials": 25,
            "td3_dsr": "median DSR across ten full 228-date aligned seed histories",
            "benchmark_dsr": "DSR of the single 228-date aligned history",
            "stability": "TD3 seed Sharpe dispersion and worst seed drawdown; deterministic benchmark dispersion is zero",
            "normalization_scope": "separate 19-strategy universe inside each cash protocol",
        },
        "mandate_aware_score": {
            "drawdown_buckets": {
                "clean_mandate": "max_drawdown >= -0.20",
                "eligible_yellow": "-0.25 <= max_drawdown < -0.20",
                "eligible_red": "-0.30 <= max_drawdown < -0.25",
                "not_eligible": "max_drawdown < -0.30",
            },
            "multiplier": "max(0, 1 - abs(drawdown)/(1-abs(drawdown)))",
        },
        "hard_mandates": {
            name: limits.to_dict() for name, limits in canonical_mandate_profiles().items()
        },
        "pareto": {
            "full_objectives": PARETO_FULL_OBJECTIVES,
            "reduced_objectives": PARETO_REDUCED_OBJECTIVES,
            "numeric_tolerance": 0.0,
            "ties": "exact ties do not dominate one another",
        },
        "transaction_costs_bps": {
            "SPY": 2.0,
            "TLT": 2.0,
            "GLD": 2.0,
            "BTC-USD": 10.0,
            "CASH": {"zero_cash": 0.0, "bil_cash": 2.0},
        },
        "td3_experiment": {
            "episodes": 60,
            "feature_families": ["V3", "V4", "V5", "V7", "V8"],
            "cap_selection": "legacy within-TD3 best_by_mandate_aware_score, fixed before aligned benchmark ranking",
            "no_retraining": True,
        },
        "protocols": protocol_metadata,
    }


def source_lineage_record(
    *,
    protocol: str,
    strategy_name: str,
    strategy_type: str,
    path: Path,
    frame: pd.DataFrame,
    repo_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    """Record source origin and content hash without copying the full input."""
    dates = pd.DatetimeIndex(frame[DATE_COLUMN])
    return {
        "protocol": protocol,
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "source_path_original": str(path),
        "portable_source_hint": _portable_path(path, repo_root, external_root),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(),
        "return_column": RETURN_COLUMN,
        "transaction_cost_mode": "asset_specific",
    }


def _selected_td3_specs(td3_dir: Path) -> list[dict[str, str]]:
    path = td3_dir / "cap_sensitivity_best_caps.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected-cap table: {path}")
    selected = pd.read_csv(path)
    specs = []
    for _, row in selected.iterrows():
        specs.append(
            {
                "base_candidate": str(row["base_candidate"]),
                "cap_label": _cap_slug(row["best_by_mandate_aware_score"]),
            }
        )
    return specs


def _cap_slug(value: Any) -> str:
    text = str(value).strip().lower()
    if text == "uncapped":
        return "uncapped"
    return f"0p{int(round(float(text) * 100)):02d}"


def _update_inventory_alignment(
    rows: list[dict[str, Any]],
    protocol: str,
    common: pd.DatetimeIndex,
) -> None:
    for row in rows:
        if row["protocol"] != protocol or "aligned_observations" in row:
            continue
        row["aligned_observations"] = len(common)
        row["aligned_start_date"] = common.min().date().isoformat()
        row["aligned_end_date"] = common.max().date().isoformat()
        row["aligned_index_equals_td3"] = True


def _portable_path(path: Path, repo_root: Path, external_root: Path) -> str:
    resolved = path.resolve()
    try:
        return "repo://" + resolved.relative_to(repo_root).as_posix()
    except ValueError:
        pass
    try:
        return "external://" + resolved.relative_to(external_root).as_posix()
    except ValueError:
        return "unmapped://" + resolved.name


def _ensure_output_directories(output: Path) -> None:
    for name in [
        "alignment",
        "histories",
        "metrics",
        "ranking",
        "mandates",
        "pareto",
        "metadata",
        "audit",
        "paper",
    ]:
        (output / name).mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_set(values: Iterable[str]) -> str:
    return hashlib.sha256("".join(sorted(values)).encode("utf-8")).hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (str, bool, dict, list)) else False:
        return None
    return value
