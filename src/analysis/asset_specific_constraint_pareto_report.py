"""Constraint-first and Pareto report for asset-specific-cost results.

This reporting layer avoids using custom robust/mandate scores as primary
evidence. It applies explicit hard mandate filters, ranks feasible strategies
with standard metrics, and computes Pareto non-dominated sets over the combined
asset-specific-cost TD3 + benchmark universe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.risk.mandate_profiles import get_default_mandate_profiles


DEFAULT_COMBINED_RANKING_PATH = (
    "outputs/tables/asset_specific_cost_benchmark_comparison/"
    "asset_specific_cost_combined_ranking.csv"
)
DEFAULT_TD3_REPORT_DIR = "outputs/tables/asset_specific_cost_full_final_report"
DEFAULT_BENCHMARK_DIR = "outputs/tables/asset_specific_cost_benchmark_comparison/benchmarks"
DEFAULT_STATISTICAL_VALIDATION_DIR = "outputs/tables/asset_specific_cost_statistical_validation"
DEFAULT_WRC_DIR = "outputs/tables/asset_specific_cost_white_reality_check"
DEFAULT_REGIME_ANALYSIS_DIR = "outputs/tables/asset_specific_cost_regime_analysis"
DEFAULT_MANDATE_PROFILE_DIR = "outputs/tables/asset_specific_cost_mandate_profile_comparison"
DEFAULT_OUTPUT_DIR = "outputs/tables/asset_specific_cost_constraint_pareto"
ASSET_SPECIFIC_COMBINED_METADATA_FILE = (
    "asset_specific_cost_benchmark_comparison_metadata.json"
)
FINAL_CORRECTED_METADATA_PATTERN = "final_corrected_*_benchmark_comparison_metadata.json"
MANDATE_PROFILE_SOURCE = "src/risk/mandate_profiles.py"
MANDATE_MAX_WEIGHT_NOTE = (
    "Max-weight caps are structural TD3 training/evaluation experiments, not official "
    "investor mandate constraints. The official mandate controls concentration through "
    "min_effective_assets; average_max_weight is reported only as a diagnostic."
)
PROFILE_ORDER = ["conservative", "moderate", "aggressive"]

KEY_STRATEGIES = [
    "V5_no_volatility_block_cap_0p50",
    "V4_real_garch_current_cap_0p50",
    "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
    "trend_spy_cash_12p",
]

STANDARD_RANK_COLUMNS = {
    "rank_by_sharpe": ("sharpe", False),
    "rank_by_calmar": ("calmar", False),
    "rank_by_sortino": ("sortino", False),
    "rank_by_max_drawdown": ("max_drawdown", False),
    "rank_by_turnover": ("average_turnover", True),
    "rank_by_effective_assets": ("average_effective_number_of_assets", False),
    "rank_by_average_max_weight": ("average_max_weight", True),
}

PARETO_FULL_OBJECTIVES = {
    "sharpe": "max",
    "annualized_return": "max",
    "calmar": "max",
    "drawdown_severity": "min",
    "average_turnover": "min",
    "mean_transaction_cost": "min",
    "average_max_weight": "min",
    "average_effective_number_of_assets": "max",
}

PARETO_REDUCED_OBJECTIVES = {
    "sharpe": "max",
    "drawdown_severity": "min",
    "average_turnover": "min",
    "average_max_weight": "min",
    "average_effective_number_of_assets": "max",
}


def canonical_mandate_profiles() -> dict[str, Any]:
    """Return the official project mandate profiles in reporting order."""
    profiles = get_default_mandate_profiles()
    return {name: profiles[name] for name in PROFILE_ORDER}


def build_constraint_pareto_report(
    combined_ranking_path: str = DEFAULT_COMBINED_RANKING_PATH,
    benchmark_dir: str = DEFAULT_BENCHMARK_DIR,
    statistical_validation_dir: str = DEFAULT_STATISTICAL_VALIDATION_DIR,
    white_reality_check_dir: str = DEFAULT_WRC_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    td3_report_dir: str = DEFAULT_TD3_REPORT_DIR,
    regime_analysis_dir: str = DEFAULT_REGIME_ANALYSIS_DIR,
    mandate_profile_dir: str = DEFAULT_MANDATE_PROFILE_DIR,
) -> dict[str, Any]:
    """Build the constraint-first and Pareto report."""
    combined_path = Path(combined_ranking_path)
    benchmark_path = Path(benchmark_dir)
    stat_path = Path(statistical_validation_dir)
    wrc_path = Path(white_reality_check_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    strategies, enrichment_notes = load_and_prepare_universe(combined_path, benchmark_path)
    metadata_validation = validate_combined_report_metadata(
        combined_path=combined_path,
        strategies=strategies,
    )
    pass_fail = build_constraint_pass_fail_matrix(strategies)
    feasible_rankings = build_feasible_strategy_rankings(pass_fail)
    standard_rankings = build_standard_metric_rankings(strategies)
    pareto_frontier, pareto_dominated = build_pareto_tables(strategies)
    summary = build_summary_markdown(
        feasible_rankings=feasible_rankings,
        pass_fail=pass_fail,
        pareto_frontier=pareto_frontier,
        wrc_dir=wrc_path,
        statistical_validation_dir=stat_path,
    )

    paths = {
        "feasible_strategy_rankings": output_path / "feasible_strategy_rankings.csv",
        "constraint_pass_fail_matrix": output_path / "constraint_pass_fail_matrix.csv",
        "standard_metric_rankings": output_path / "standard_metric_rankings.csv",
        "pareto_frontier": output_path / "pareto_frontier.csv",
        "pareto_dominated_strategies": output_path / "pareto_dominated_strategies.csv",
        "summary": output_path / "constraint_pareto_summary.md",
        "metadata": output_path / "constraint_pareto_metadata.json",
    }
    feasible_rankings.to_csv(paths["feasible_strategy_rankings"], index=False)
    pass_fail.to_csv(paths["constraint_pass_fail_matrix"], index=False)
    standard_rankings.to_csv(paths["standard_metric_rankings"], index=False)
    pareto_frontier.to_csv(paths["pareto_frontier"], index=False)
    pareto_dominated.to_csv(paths["pareto_dominated_strategies"], index=False)
    paths["summary"].write_text(summary, encoding="utf-8")

    profiles = canonical_mandate_profiles()
    metadata = {
        "runner": "src.analysis.asset_specific_constraint_pareto_report",
        "combined_ranking_path": str(combined_path),
        "td3_report_dir": td3_report_dir,
        "benchmark_dir": str(benchmark_path),
        "statistical_validation_dir": str(stat_path),
        "white_reality_check_dir": str(wrc_path),
        "regime_analysis_dir": regime_analysis_dir,
        "mandate_profile_dir": mandate_profile_dir,
        "output_dir": str(output_path),
        "mandate_profile_source": MANDATE_PROFILE_SOURCE,
        "canonical_mandate_profiles": {
            name: profile.to_dict()
            for name, profile in profiles.items()
        },
        "max_weight_mandate_note": MANDATE_MAX_WEIGHT_NOTE,
        "primary_evidence_note": (
            "Hard mandate filters, standard metric rankings, and Pareto dominance are primary here; "
            "custom robust_score and mandate_aware_score are not used for primary ranking."
        ),
        "pareto_full_objectives": PARETO_FULL_OBJECTIVES,
        "pareto_reduced_objectives": PARETO_REDUCED_OBJECTIVES,
        "enrichment_notes": enrichment_notes,
        "input_validation": metadata_validation,
        "n_strategies": int(len(strategies)),
    }
    paths["metadata"].write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")

    return {
        "strategies": strategies,
        "pass_fail": pass_fail,
        "feasible_rankings": feasible_rankings,
        "standard_rankings": standard_rankings,
        "pareto_frontier": pareto_frontier,
        "pareto_dominated": pareto_dominated,
        "summary": summary,
        "metadata": metadata,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def load_and_prepare_universe(
    combined_ranking_path: Path,
    benchmark_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Load combined ranking and enrich missing benchmark diagnostics from histories."""
    if not combined_ranking_path.exists():
        raise FileNotFoundError(f"Missing combined ranking: {combined_ranking_path}")
    data = pd.read_csv(combined_ranking_path)
    if data.empty:
        raise ValueError("Combined ranking is empty.")
    data["strategy_type"] = data["strategy_type"].astype(str).str.lower()
    if (
        "average_effective_number_of_assets" not in data.columns
        and "average_effective_assets" in data.columns
    ):
        data["average_effective_number_of_assets"] = data["average_effective_assets"]
    required = [
        "strategy_name",
        "strategy_type",
        "transaction_cost_mode",
        "sharpe",
        "calmar",
        "sortino",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "average_turnover",
        "mean_transaction_cost",
        "average_max_weight",
        "average_effective_number_of_assets",
    ]
    missing_base = [column for column in required if column not in data.columns]
    if missing_base:
        raise ValueError(f"Combined ranking missing required columns: {missing_base}")
    modes = set(data["transaction_cost_mode"].dropna().astype(str).unique().tolist())
    if modes != {"asset_specific"}:
        raise ValueError(f"Expected only asset_specific transaction_cost_mode, found {sorted(modes)}")

    notes = enrich_benchmark_diagnostics_from_histories(data, benchmark_dir)
    validate_required_diagnostics(data)
    data["drawdown_severity"] = data["max_drawdown"].abs()
    return data, notes


def validate_combined_report_metadata(
    combined_path: Path,
    strategies: pd.DataFrame,
) -> dict[str, Any]:
    """Validate corrected/asset-specific comparison metadata when available."""
    metadata_path = _metadata_path_for_combined_ranking(combined_path)
    td3_count = int((strategies["strategy_type"] == "td3").sum())
    benchmark_count = int((strategies["strategy_type"] == "benchmark").sum())
    result = {
        "selected_td3_count": td3_count,
        "benchmark_count": benchmark_count,
        "metadata_path": str(metadata_path) if metadata_path else None,
        "transaction_cost_mode": None,
        "asset_transaction_cost_bps": None,
        "cash_bps": None,
    }
    if metadata_path is None:
        return result
    if td3_count != 5:
        raise ValueError(f"Expected 5 selected TD3 strategies, found {td3_count}.")
    if benchmark_count != 14:
        raise ValueError(f"Expected 14 benchmark strategies, found {benchmark_count}.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mode = metadata.get("transaction_cost_mode") or metadata.get("cost_model", {}).get(
        "transaction_cost_mode"
    )
    bps = metadata.get("asset_transaction_cost_bps") or metadata.get("cost_model", {}).get(
        "asset_transaction_cost_bps"
    )
    bps = _normalize_bps(bps)
    if mode != "asset_specific":
        raise ValueError(f"Combined comparison metadata is not asset-specific: {mode}")
    expected_core = {"SPY": 2.0, "TLT": 2.0, "GLD": 2.0, "BTC-USD": 10.0}
    for asset, expected in expected_core.items():
        if bps.get(asset) != expected:
            raise ValueError(f"Unexpected {asset} cost bps: {bps.get(asset)}")
    if bps.get("CASH") not in {0.0, 2.0}:
        raise ValueError(f"Unexpected CASH cost bps: {bps.get('CASH')}")

    result.update(
        {
            "transaction_cost_mode": mode,
            "asset_transaction_cost_bps": bps,
            "cash_bps": bps.get("CASH"),
        }
    )
    return result


def _metadata_path_for_combined_ranking(combined_path: Path) -> Path | None:
    directory = combined_path.parent
    asset_path = directory / ASSET_SPECIFIC_COMBINED_METADATA_FILE
    if asset_path.exists():
        return asset_path
    corrected = sorted(directory.glob(FINAL_CORRECTED_METADATA_PATTERN))
    return corrected[0] if corrected else None


def _normalize_bps(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(asset): float(value[asset]) for asset in sorted(value)}


def enrich_benchmark_diagnostics_from_histories(
    data: pd.DataFrame,
    benchmark_dir: Path,
) -> list[str]:
    """Fill missing benchmark diagnostics from history files where possible."""
    notes: list[str] = []
    history_dir = benchmark_dir / "histories"
    benchmark_mask = data["strategy_type"].astype(str) == "benchmark"
    diagnostic_columns = [
        "average_turnover",
        "mean_transaction_cost",
        "average_max_weight",
        "average_effective_number_of_assets",
    ]
    needs_enrichment = data.loc[benchmark_mask, diagnostic_columns].isna().any(axis=1)
    if not needs_enrichment.any():
        return notes
    if not history_dir.exists():
        raise FileNotFoundError(f"Benchmark diagnostics missing and history directory not found: {history_dir}")

    for idx, row in data.loc[benchmark_mask & needs_enrichment].iterrows():
        strategy_name = str(row["strategy_name"])
        history_path = history_dir / f"{strategy_name}_history.csv"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing benchmark history for diagnostics: {history_path}")
        history = pd.read_csv(history_path)
        diagnostics = compute_history_diagnostics(history, strategy_name)
        for column, value in diagnostics.items():
            if column in data.columns and pd.isna(data.at[idx, column]):
                data.at[idx, column] = value
        notes.append(f"Filled missing benchmark diagnostics for {strategy_name} from {history_path}.")
    return notes


def compute_history_diagnostics(history: pd.DataFrame, strategy_name: str) -> dict[str, float]:
    """Compute benchmark diagnostics from one history table."""
    if "transaction_cost_mode" in history.columns:
        modes = set(history["transaction_cost_mode"].dropna().astype(str).unique().tolist())
        if modes != {"asset_specific"}:
            raise ValueError(f"{strategy_name} history is not asset-specific: {sorted(modes)}")
    else:
        raise ValueError(f"{strategy_name} history missing transaction_cost_mode.")
    weight_cols = [col for col in history.columns if col.startswith("weight_")]
    if not weight_cols:
        raise ValueError(f"{strategy_name} history has no weight columns for diagnostics.")
    weights = history[weight_cols].apply(pd.to_numeric, errors="coerce")
    max_weight = weights.max(axis=1).mean()
    effective_assets = (1.0 / weights.pow(2).sum(axis=1).replace(0.0, np.nan)).mean()
    diagnostics = {
        "average_max_weight": float(max_weight),
        "average_effective_number_of_assets": float(effective_assets),
    }
    if "turnover" in history.columns:
        diagnostics["average_turnover"] = float(pd.to_numeric(history["turnover"], errors="coerce").mean())
    if "transaction_cost" in history.columns:
        diagnostics["mean_transaction_cost"] = float(pd.to_numeric(history["transaction_cost"], errors="coerce").mean())
    return diagnostics


def validate_required_diagnostics(data: pd.DataFrame) -> None:
    required_numeric = [
        "sharpe",
        "calmar",
        "sortino",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "average_turnover",
        "mean_transaction_cost",
        "average_max_weight",
        "average_effective_number_of_assets",
    ]
    for column in required_numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    missing_rows = data[data[required_numeric].isna().any(axis=1)]
    if not missing_rows.empty:
        problems = missing_rows[["strategy_name", "strategy_type"]].to_dict("records")
        raise ValueError(f"Missing required diagnostics after enrichment: {problems}")


def build_constraint_pass_fail_matrix(strategies: pd.DataFrame) -> pd.DataFrame:
    """Return one pass/fail row per profile and strategy."""
    rows: list[dict[str, Any]] = []
    for profile, limits in canonical_mandate_profiles().items():
        for _, strategy in strategies.iterrows():
            checks = {
                "max_drawdown_pass": strategy["max_drawdown"] >= limits.max_drawdown,
                "annualized_volatility_pass": (
                    strategy["annualized_volatility"] <= limits.max_annualized_volatility
                ),
                "average_effective_number_of_assets_pass": (
                    strategy["average_effective_number_of_assets"]
                    >= limits.min_effective_assets
                ),
                "average_turnover_pass": strategy["average_turnover"] <= limits.max_average_turnover,
            }
            failed = [name.replace("_pass", "") for name, passed in checks.items() if not bool(passed)]
            rows.append(
                {
                    "profile": profile,
                    "strategy_name": strategy["strategy_name"],
                    "strategy_type": strategy["strategy_type"],
                    "strategy_group": strategy.get("strategy_group", np.nan),
                    "feasible": not failed,
                    "failed_constraints": ";".join(failed),
                    **checks,
                    "max_drawdown": strategy["max_drawdown"],
                    "max_drawdown_limit": limits.max_drawdown,
                    "annualized_volatility": strategy["annualized_volatility"],
                    "max_annualized_volatility_limit": limits.max_annualized_volatility,
                    "average_max_weight": strategy["average_max_weight"],
                    "average_effective_number_of_assets": strategy["average_effective_number_of_assets"],
                    "min_effective_assets_limit": limits.min_effective_assets,
                    "average_turnover": strategy["average_turnover"],
                    "max_average_turnover_limit": limits.max_average_turnover,
                    "sharpe": strategy["sharpe"],
                    "calmar": strategy["calmar"],
                    "sortino": strategy["sortino"],
                    "annualized_return": strategy["annualized_return"],
                }
            )
    return pd.DataFrame(rows)


def build_feasible_strategy_rankings(pass_fail: pd.DataFrame) -> pd.DataFrame:
    """Rank feasible strategies within each mandate profile."""
    feasible = pass_fail[pass_fail["feasible"]].copy()
    if feasible.empty:
        return feasible.assign(profile_rank=pd.Series(dtype="int64"))
    feasible = feasible.sort_values(
        ["profile", "sharpe", "calmar", "sortino", "annualized_return", "strategy_name"],
        ascending=[True, False, False, False, False, True],
    )
    feasible["profile_rank"] = feasible.groupby("profile").cumcount() + 1
    return feasible.reset_index(drop=True)


def build_standard_metric_rankings(strategies: pd.DataFrame) -> pd.DataFrame:
    """Add standard metric ranks over the common combined universe."""
    ranked = strategies.copy()
    for rank_col, (metric_col, ascending) in STANDARD_RANK_COLUMNS.items():
        ranked[rank_col] = ranked[metric_col].rank(method="min", ascending=ascending).astype(int)
    cols = [
        "strategy_name",
        "strategy_type",
        "strategy_group",
        "sharpe",
        "calmar",
        "sortino",
        "annualized_return",
        "max_drawdown",
        "average_turnover",
        "mean_transaction_cost",
        "average_effective_number_of_assets",
        "average_max_weight",
        *STANDARD_RANK_COLUMNS.keys(),
    ]
    return ranked[[col for col in cols if col in ranked.columns]].sort_values("rank_by_sharpe")


def build_pareto_tables(strategies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return combined full and reduced Pareto frontier/dominated tables."""
    frontiers = []
    dominated_tables = []
    for frontier_type, objectives in [
        ("full", PARETO_FULL_OBJECTIVES),
        ("reduced", PARETO_REDUCED_OBJECTIVES),
    ]:
        non_dominated_mask = pareto_non_dominated_mask(strategies, objectives)
        frontier = strategies.loc[non_dominated_mask].copy()
        dominated = strategies.loc[~non_dominated_mask].copy()
        frontier["frontier_type"] = frontier_type
        dominated["frontier_type"] = frontier_type
        frontier["objective_columns"] = ",".join(objectives.keys())
        dominated["objective_columns"] = ",".join(objectives.keys())
        frontiers.append(frontier)
        dominated_tables.append(dominated)
    output_cols = [
        "frontier_type",
        "strategy_name",
        "strategy_type",
        "strategy_group",
        "sharpe",
        "annualized_return",
        "calmar",
        "max_drawdown",
        "drawdown_severity",
        "average_turnover",
        "mean_transaction_cost",
        "average_max_weight",
        "average_effective_number_of_assets",
        "objective_columns",
    ]
    frontier_df = pd.concat(frontiers, ignore_index=True)
    dominated_df = pd.concat(dominated_tables, ignore_index=True)
    return (
        frontier_df[[col for col in output_cols if col in frontier_df.columns]].sort_values(
            ["frontier_type", "strategy_name"]
        ),
        dominated_df[[col for col in output_cols if col in dominated_df.columns]].sort_values(
            ["frontier_type", "strategy_name"]
        ),
    )


def pareto_non_dominated_mask(data: pd.DataFrame, objectives: dict[str, str]) -> np.ndarray:
    """Compute non-dominated mask for max/min objectives."""
    missing = [column for column in objectives if column not in data.columns]
    if missing:
        raise ValueError(f"Pareto objectives missing: {missing}")
    values = data[list(objectives.keys())].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("Pareto objectives contain missing values.")
    transformed = values.copy()
    for column, direction in objectives.items():
        if direction == "min":
            transformed[column] = -transformed[column]
        elif direction != "max":
            raise ValueError(f"Unsupported Pareto direction {direction!r} for {column}")
    arr = transformed.to_numpy(dtype=float)
    n = arr.shape[0]
    non_dominated = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(arr[j] >= arr[i]) and np.any(arr[j] > arr[i]):
                non_dominated[i] = False
                break
    return non_dominated


def build_summary_markdown(
    feasible_rankings: pd.DataFrame,
    pass_fail: pd.DataFrame,
    pareto_frontier: pd.DataFrame,
    wrc_dir: Path,
    statistical_validation_dir: Path,
) -> str:
    winners = summarize_feasible_winners(feasible_rankings)
    key_pass = summarize_key_passes(pass_fail)
    wrc_note = summarize_wrc(wrc_dir)
    stat_note = summarize_statistical_validation(statistical_validation_dir)
    lines = [
        "# Asset-Specific Constraint-First and Pareto Report",
        "",
        "This report does not use custom `robust_score` or `mandate_aware_score` as primary evidence.",
        "It uses hard mandate filters, standard metric ranking among feasible strategies, and Pareto non-dominance.",
        "Statistical validation remains separate.",
        "",
        "## Feasible Winners",
        "",
    ]
    lines.extend(
        [
            "The official hard filters are drawdown, annualized volatility, effective-assets minimum, and average turnover.",
            MANDATE_MAX_WEIGHT_NOTE,
            "",
        ]
    )
    for profile in PROFILE_ORDER:
        row = winners.get(profile)
        if row is None:
            lines.append(f"- {profile}: no feasible strategies.")
        else:
            lines.append(
                f"- {profile}: best feasible overall `{row['overall']}`; "
                f"best TD3 {_format_optional_strategy(row['td3'])}; "
                f"best benchmark {_format_optional_strategy(row['benchmark'])}; "
                f"feasible TD3 count {row['n_td3']}; feasible benchmark count {row['n_benchmarks']}."
            )
    lines.extend(["", "## Key Strategy Pass/Fail", ""])
    for strategy_name, profile_map in key_pass.items():
        bits = ", ".join(f"{profile}: {'pass' if passed else 'fail'}" for profile, passed in profile_map.items())
        lines.append(f"- `{strategy_name}` - {bits}.")
    full_frontier = pareto_frontier[pareto_frontier["frontier_type"] == "full"]["strategy_name"].tolist()
    reduced_frontier = pareto_frontier[pareto_frontier["frontier_type"] == "reduced"]["strategy_name"].tolist()
    lines.extend(
        [
            "",
            "## Pareto Frontier",
            "",
            f"- Full objective frontier: {', '.join(f'`{name}`' for name in full_frontier) if full_frontier else 'none'}.",
            f"- Reduced objective frontier: {', '.join(f'`{name}`' for name in reduced_frontier) if reduced_frontier else 'none'}.",
            "",
            "## Interpretation",
            "",
        ]
    )
    v5_rows = feasible_rankings[feasible_rankings["strategy_name"] == "V5_no_volatility_block_cap_0p50"]
    if not v5_rows.empty:
        profiles = ", ".join(v5_rows["profile"].astype(str).tolist())
        lines.append(f"`V5_no_volatility_block_cap_0p50` is feasible in: {profiles}.")
    else:
        lines.append("`V5_no_volatility_block_cap_0p50` does not pass any hard mandate filter.")
    conservative_top = winners.get("conservative", {})
    if conservative_top.get("td3") == "V4_real_garch_current_cap_0p50":
        lines.append("`V4_real_garch_current_cap_0p50` is the best feasible TD3 under the conservative filter.")
    elif conservative_top == {}:
        v4_conservative = pass_fail[
            (pass_fail["profile"] == "conservative")
            & (pass_fail["strategy_name"] == "V4_real_garch_current_cap_0p50")
        ]
        if not v4_conservative.empty:
            failed = str(v4_conservative.iloc[0]["failed_constraints"])
            lines.append(
                "`V4_real_garch_current_cap_0p50` is not feasible under the conservative hard filter "
                f"because it fails: {failed}."
            )
    benchmark_wins = [profile for profile, row in winners.items() if row.get("overall_type") == "benchmark"]
    if benchmark_wins:
        lines.append(f"Benchmarks win the feasible ranking in: {', '.join(benchmark_wins)}.")
    else:
        lines.append("Benchmarks do not win the top feasible slot here, but feasible benchmark rows remain part of the comparison.")
    lines.extend(
        [
            wrc_note,
            stat_note,
            "Do not claim superiority unless the separate statistical validation supports it.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_feasible_winners(feasible_rankings: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for profile in PROFILE_ORDER:
        rows = feasible_rankings[feasible_rankings["profile"] == profile].sort_values("profile_rank")
        if rows.empty:
            continue
        best = rows.iloc[0]
        td3_rows = rows[rows["strategy_type"] == "td3"]
        bench_rows = rows[rows["strategy_type"] == "benchmark"]
        summary[profile] = {
            "overall": best["strategy_name"],
            "overall_type": best["strategy_type"],
            "td3": td3_rows.iloc[0]["strategy_name"] if not td3_rows.empty else np.nan,
            "benchmark": bench_rows.iloc[0]["strategy_name"] if not bench_rows.empty else np.nan,
            "n_td3": int(len(td3_rows)),
            "n_benchmarks": int(len(bench_rows)),
        }
    return summary


def summarize_key_passes(pass_fail: pd.DataFrame) -> dict[str, dict[str, bool]]:
    result: dict[str, dict[str, bool]] = {}
    for strategy_name in KEY_STRATEGIES:
        rows = pass_fail[pass_fail["strategy_name"] == strategy_name]
        result[strategy_name] = {
            row["profile"]: bool(row["feasible"])
            for _, row in rows.iterrows()
        }
    return result


def _format_optional_strategy(value: Any) -> str:
    if pd.isna(value):
        return "none"
    return f"`{value}`"


def summarize_wrc(wrc_dir: Path) -> str:
    path = wrc_dir / "white_reality_check_summary.csv"
    if not path.exists():
        return "White Reality Check output was not found; no statistical superiority statement is made."
    data = pd.read_csv(path)
    if "p_value" not in data.columns or data.empty:
        return "White Reality Check output is unavailable or incomplete; no statistical superiority statement is made."
    min_p = pd.to_numeric(data["p_value"], errors="coerce").min()
    if pd.isna(min_p):
        return "White Reality Check p-values are unavailable; no statistical superiority statement is made."
    if min_p < 0.05:
        return f"White Reality Check minimum p-value is {min_p:.3f}; interpret with the reported benchmark-specific table."
    return f"White Reality Check minimum p-value is {min_p:.3f}; no searched TD3 superiority claim is supported."


def summarize_statistical_validation(stat_dir: Path) -> str:
    path = stat_dir / "statistical_validation_pairwise_bootstrap.csv"
    if not path.exists():
        return "Pairwise bootstrap output was not found; uncertainty remains undocumented here."
    data = pd.read_csv(path)
    if "probability_candidate_beats" not in data.columns or data.empty:
        return "Pairwise bootstrap output is unavailable or incomplete; uncertainty remains."
    probs = pd.to_numeric(data["probability_candidate_beats"], errors="coerce").dropna()
    if probs.empty:
        return "Pairwise bootstrap probabilities are unavailable; uncertainty remains."
    return f"Pairwise bootstrap beat probabilities range from {probs.min():.2f} to {probs.max():.2f}; read this as uncertainty, not proof."


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build constraint-first and Pareto report.")
    parser.add_argument("--combined-ranking-path", default=DEFAULT_COMBINED_RANKING_PATH)
    parser.add_argument("--td3-report-dir", default=DEFAULT_TD3_REPORT_DIR)
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--statistical-validation-dir", default=DEFAULT_STATISTICAL_VALIDATION_DIR)
    parser.add_argument("--white-reality-check-dir", default=DEFAULT_WRC_DIR)
    parser.add_argument("--regime-analysis-dir", default=DEFAULT_REGIME_ANALYSIS_DIR)
    parser.add_argument("--mandate-profile-dir", default=DEFAULT_MANDATE_PROFILE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = build_constraint_pareto_report(
        combined_ranking_path=args.combined_ranking_path,
        td3_report_dir=args.td3_report_dir,
        benchmark_dir=args.benchmark_dir,
        statistical_validation_dir=args.statistical_validation_dir,
        white_reality_check_dir=args.white_reality_check_dir,
        regime_analysis_dir=args.regime_analysis_dir,
        mandate_profile_dir=args.mandate_profile_dir,
        output_dir=args.output_dir,
    )
    winners = summarize_feasible_winners(result["feasible_rankings"])
    print("Feasible winners:")
    for profile, row in winners.items():
        print(f"{profile}: {row['overall']}")
    print("\nPareto frontier strategies:")
    print(result["pareto_frontier"][["frontier_type", "strategy_name"]].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
