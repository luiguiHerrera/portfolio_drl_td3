"""Compare asset-specific-cost-aware TD3 candidates against benchmarks.

This module is reporting-only. It combines the official TD3 asset-specific
cost report with deterministic benchmark histories regenerated under the same
asset-specific transaction cost model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.asset_specific_cost_final_report import EXPECTED_COST_MODEL
from src.analysis.mandate_aware_score import (
    assign_drawdown_bucket,
    calculate_recovery_required,
    get_drawdown_multiplier,
)
from src.analysis.robust_score import (
    compute_composite_robust_score,
    compute_deflated_sharpe_ratio,
)


DEFAULT_TD3_REPORT_DIR = "outputs/tables/asset_specific_cost_full_final_report"
DEFAULT_BENCHMARK_DIR = "outputs/tables/asset_specific_cost_benchmark_comparison/benchmarks"
DEFAULT_OUTPUT_DIR = "outputs/tables/asset_specific_cost_benchmark_comparison"
PERIODS_PER_YEAR = 52
BROKER_COST_CAVEAT = (
    "Broker/exchange-style trading-cost proxy only; does not model fiat ramps, "
    "withdrawals, custody frictions, taxes, market impact, or delays."
)
REQUIRED_BENCHMARK_HISTORY_COLUMNS = [
    "transaction_cost_mode",
    "transaction_cost",
    "turnover",
    "financial_net_return",
    "asset_turnover_SPY",
    "asset_turnover_TLT",
    "asset_turnover_GLD",
    "asset_turnover_BTC-USD",
    "asset_turnover_CASH",
    "asset_transaction_cost_contribution_SPY",
    "asset_transaction_cost_contribution_TLT",
    "asset_transaction_cost_contribution_GLD",
    "asset_transaction_cost_contribution_BTC-USD",
    "asset_transaction_cost_contribution_CASH",
]
RANKING_COLUMNS = [
    "rank_mandate_aware",
    "rank_robust",
    "rank_sharpe",
    "strategy_name",
    "strategy_type",
    "strategy_group",
    "base_candidate",
    "cap_label",
    "transaction_cost_mode",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "robust_score",
    "mandate_aware_score",
    "mandate_bucket",
    "recovery_required",
    "drawdown_multiplier",
    "max_drawdown",
    "worst_max_drawdown",
    "average_turnover",
    "mean_transaction_cost",
    "average_effective_number_of_assets",
    "average_max_weight",
    "mean_cash_weight",
    "mean_btc_weight",
    "mean_btc_transaction_cost_contribution",
    "dsr_score",
    "dsr_method",
]


def build_asset_specific_cost_benchmark_comparison_report(
    td3_report_dir: str = DEFAULT_TD3_REPORT_DIR,
    benchmark_dir: str = DEFAULT_BENCHMARK_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build combined TD3-vs-benchmark comparison under asset-specific costs."""
    td3_path = Path(td3_report_dir)
    benchmark_path = Path(benchmark_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    td3_metadata = _load_json(td3_path / "asset_specific_cost_metadata.json")
    benchmark_metadata = _load_json(benchmark_path / "benchmark_metadata.json")
    validate_cost_model_match(td3_metadata, benchmark_metadata)

    td3_rows = load_selected_td3_rows(td3_path)
    benchmark_rows, benchmark_warnings = load_benchmark_rows(benchmark_path)
    combined = pd.concat([td3_rows, benchmark_rows], ignore_index=True, sort=False)
    combined = add_combined_scores(combined)
    combined = add_ranks(combined)
    combined_ranking = select_ranking_columns(combined)
    td3_vs_benchmarks = build_td3_vs_benchmark_table(combined_ranking)
    metadata = build_metadata(
        td3_report_dir=str(td3_path),
        benchmark_dir=str(benchmark_path),
        output_dir=str(output_path),
        td3_metadata=td3_metadata,
        benchmark_metadata=benchmark_metadata,
        combined_ranking=combined_ranking,
        warnings=benchmark_warnings,
    )
    markdown = build_summary_markdown(combined_ranking, td3_vs_benchmarks, metadata)

    paths = {
        "combined_ranking": output_path
        / "asset_specific_cost_combined_ranking.csv",
        "td3_vs_benchmarks": output_path
        / "asset_specific_cost_td3_vs_benchmarks.csv",
        "summary": output_path
        / "asset_specific_cost_benchmark_comparison_summary.md",
        "metadata": output_path
        / "asset_specific_cost_benchmark_comparison_metadata.json",
    }
    combined_ranking.to_csv(paths["combined_ranking"], index=False)
    td3_vs_benchmarks.to_csv(paths["td3_vs_benchmarks"], index=False)
    paths["summary"].write_text(markdown, encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "combined_ranking": combined_ranking,
        "td3_vs_benchmarks": td3_vs_benchmarks,
        "metadata": metadata,
        "summary": markdown,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def load_selected_td3_rows(td3_report_dir: Path) -> pd.DataFrame:
    """Load selected official TD3 candidates and normalize column names."""
    path = td3_report_dir / "asset_specific_cost_selected_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected TD3 candidates: {path}")
    selected = pd.read_csv(path)
    if selected.empty:
        raise ValueError("Selected TD3 candidates table must not be empty.")
    result = selected.copy()
    result["strategy_name"] = result["candidate_name"]
    result["strategy_type"] = "td3"
    result["strategy_group"] = "td3_selected"
    result["transaction_cost_mode"] = "asset_specific"
    if "mean_transaction_cost" not in result and "average_transaction_cost" in result:
        result["mean_transaction_cost"] = result["average_transaction_cost"]
    return result


def load_benchmark_rows(benchmark_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load benchmark metrics and derive benchmark DSR diagnostics from histories."""
    metrics_path = benchmark_dir / "benchmark_metrics_table.csv"
    diagnostics_path = benchmark_dir / "benchmark_diagnostics.csv"
    histories_dir = benchmark_dir / "histories"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing benchmark metrics table: {metrics_path}")
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing benchmark diagnostics table: {diagnostics_path}")
    if not histories_dir.exists():
        raise FileNotFoundError(f"Missing benchmark histories directory: {histories_dir}")

    metrics = pd.read_csv(metrics_path)
    diagnostics = pd.read_csv(diagnostics_path)
    result = metrics.merge(
        diagnostics[["benchmark_name", "transaction_cost_mode"]],
        on="benchmark_name",
        how="left",
        suffixes=("", "_diagnostic"),
    )
    warnings: list[str] = []
    dsr_scores = []
    mean_btc_weights = []
    btc_cost_contributions = []
    worst_drawdowns = []
    for benchmark_name in result["benchmark_name"]:
        history_path = histories_dir / f"{_safe_filename(benchmark_name)}_history.csv"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing benchmark history: {history_path}")
        history = pd.read_csv(history_path)
        validate_benchmark_history(history, history_path)
        returns = pd.to_numeric(history["financial_net_return"], errors="coerce").dropna()
        dsr_scores.append(
            compute_deflated_sharpe_ratio(
                returns,
                n_trials=25,
                periods_per_year=PERIODS_PER_YEAR,
            )
        )
        mean_btc_weights.append(_mean_if_present(history, "weight_BTC-USD"))
        btc_cost_contributions.append(
            _mean_if_present(history, "asset_transaction_cost_contribution_BTC-USD")
        )
        worst_drawdowns.append(_min_if_present(history, "drawdown"))

    result["strategy_name"] = result["benchmark_name"]
    result["strategy_type"] = "benchmark"
    result["strategy_group"] = "benchmark"
    result["base_candidate"] = pd.NA
    result["cap_label"] = pd.NA
    result["mean_transaction_cost"] = result.get("average_transaction_cost", pd.NA)
    result["mean_btc_weight"] = mean_btc_weights
    result["mean_btc_transaction_cost_contribution"] = btc_cost_contributions
    result["worst_max_drawdown"] = pd.Series(worst_drawdowns).combine_first(
        result["max_drawdown"],
    )
    result["dsr_score"] = dsr_scores
    result["dsr_method"] = "history_dsr_n25"
    result["cash_above_10_rate"] = result.get("cash_above_10pct", 0.0)
    return result, warnings


def validate_cost_model_match(td3_metadata: dict, benchmark_metadata: dict) -> None:
    """Fail loudly if TD3 and benchmark cost models are not comparable."""
    td3_cost = td3_metadata.get("cost_model", {})
    benchmark_mode = benchmark_metadata.get("transaction_cost_mode")
    benchmark_bps = benchmark_metadata.get("asset_transaction_cost_bps")
    if td3_metadata.get("score_scope") != "combined_asset_specific_full_universe":
        raise ValueError("TD3 report metadata must have combined asset-specific score_scope.")
    if td3_cost.get("transaction_cost_mode") != "asset_specific":
        raise ValueError("TD3 report is not asset-specific cost aware.")
    if benchmark_mode != "asset_specific":
        raise ValueError("Benchmark report is not asset-specific cost aware.")
    if _normalize_bps(td3_cost.get("asset_transaction_cost_bps")) != _normalize_bps(
        benchmark_bps
    ):
        raise ValueError("TD3 and benchmark asset transaction cost bps mappings differ.")


def validate_benchmark_history(history: pd.DataFrame, path: Path) -> None:
    """Validate benchmark history cost diagnostics."""
    missing = [column for column in REQUIRED_BENCHMARK_HISTORY_COLUMNS if column not in history]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    modes = set(history["transaction_cost_mode"].dropna().astype(str).unique().tolist())
    if modes != {"asset_specific"}:
        raise ValueError(f"{path} has non asset-specific modes: {sorted(modes)}")


def add_combined_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute robust and mandate-aware scores over combined TD3+benchmark universe."""
    result = frame.copy()
    scoring_input = result.rename(
        columns={
            "strategy_name": "strategy",
            "worst_max_drawdown": "worst_drawdown",
            "average_turnover": "turnover",
            "average_effective_number_of_assets": "effective_assets",
        },
    )
    scoring_input["type"] = scoring_input["strategy_type"]
    scored = compute_composite_robust_score(scoring_input)
    result = scored.rename(
        columns={
            "strategy": "strategy_name",
            "worst_drawdown": "worst_max_drawdown",
            "turnover": "average_turnover",
            "effective_assets": "average_effective_number_of_assets",
        },
    )
    result["mandate_bucket"] = result["max_drawdown"].apply(assign_drawdown_bucket)
    result["recovery_required"] = result["max_drawdown"].apply(calculate_recovery_required)
    result["drawdown_multiplier"] = result["max_drawdown"].apply(get_drawdown_multiplier)
    result["mandate_aware_score"] = result["robust_score"] * result["drawdown_multiplier"]
    result.loc[result["mandate_bucket"] == "not_eligible", "mandate_aware_score"] = 0.0
    return result


def add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Add combined-universe ranks."""
    result = frame.copy()
    result["rank_mandate_aware"] = (
        pd.to_numeric(result["mandate_aware_score"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype(int)
    )
    result["rank_robust"] = (
        pd.to_numeric(result["robust_score"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype(int)
    )
    result["rank_sharpe"] = (
        pd.to_numeric(result["sharpe"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return result.sort_values(
        ["rank_mandate_aware", "rank_robust", "rank_sharpe"],
    ).reset_index(drop=True)


def select_ranking_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Select stable output columns."""
    result = frame.copy()
    for column in RANKING_COLUMNS:
        if column not in result:
            result[column] = pd.NA
    return result.loc[:, RANKING_COLUMNS]


def build_td3_vs_benchmark_table(combined_ranking: pd.DataFrame) -> pd.DataFrame:
    """Build pairwise TD3-vs-benchmark metric deltas."""
    td3 = combined_ranking.loc[combined_ranking["strategy_type"] == "td3"].copy()
    benchmarks = combined_ranking.loc[
        combined_ranking["strategy_type"] == "benchmark"
    ].copy()
    rows = []
    for _, td3_row in td3.iterrows():
        for _, bench in benchmarks.iterrows():
            rows.append(
                {
                    "td3_strategy": td3_row["strategy_name"],
                    "benchmark_strategy": bench["strategy_name"],
                    "td3_mandate_aware_score": td3_row["mandate_aware_score"],
                    "benchmark_mandate_aware_score": bench["mandate_aware_score"],
                    "delta_mandate_aware_score": _num(td3_row["mandate_aware_score"])
                    - _num(bench["mandate_aware_score"]),
                    "td3_robust_score": td3_row["robust_score"],
                    "benchmark_robust_score": bench["robust_score"],
                    "delta_robust_score": _num(td3_row["robust_score"])
                    - _num(bench["robust_score"]),
                    "td3_sharpe": td3_row["sharpe"],
                    "benchmark_sharpe": bench["sharpe"],
                    "delta_sharpe": _num(td3_row["sharpe"]) - _num(bench["sharpe"]),
                    "td3_max_drawdown": td3_row["max_drawdown"],
                    "benchmark_max_drawdown": bench["max_drawdown"],
                    "delta_max_drawdown": _num(td3_row["max_drawdown"])
                    - _num(bench["max_drawdown"]),
                    "td3_beats_benchmark_by_mandate": _num(
                        td3_row["mandate_aware_score"]
                    )
                    > _num(bench["mandate_aware_score"]),
                    "td3_beats_benchmark_by_sharpe": _num(td3_row["sharpe"])
                    > _num(bench["sharpe"]),
                    "td3_has_lower_drawdown_than_benchmark": _num(
                        td3_row["max_drawdown"]
                    )
                    > _num(bench["max_drawdown"]),
                }
            )
    return pd.DataFrame(rows)


def build_metadata(
    td3_report_dir: str,
    benchmark_dir: str,
    output_dir: str,
    td3_metadata: dict,
    benchmark_metadata: dict,
    combined_ranking: pd.DataFrame,
    warnings: list[str],
) -> dict:
    """Build reproducibility metadata."""
    return {
        "runner": "src.analysis.asset_specific_cost_benchmark_comparison_report",
        "td3_report_dir": td3_report_dir,
        "benchmark_dir": benchmark_dir,
        "output_dir": output_dir,
        "cost_model": EXPECTED_COST_MODEL,
        "td3_score_scope": td3_metadata.get("score_scope"),
        "combined_score_scope": "selected_td3_plus_asset_specific_benchmarks",
        "scoring_note": (
            "robust_score and mandate_aware_score are recomputed over selected "
            "TD3 candidates plus deterministic benchmarks. TD3 DSR inputs are "
            "taken from the official TD3 report; benchmark DSR inputs are "
            "computed from regenerated benchmark histories."
        ),
        "n_td3": int((combined_ranking["strategy_type"] == "td3").sum()),
        "n_benchmarks": int((combined_ranking["strategy_type"] == "benchmark").sum()),
        "benchmark_names": benchmark_metadata.get("benchmark_names", []),
        "cost_caveat": BROKER_COST_CAVEAT,
        "warnings": warnings,
    }


def build_summary_markdown(
    combined_ranking: pd.DataFrame,
    td3_vs_benchmarks: pd.DataFrame,
    metadata: dict,
) -> str:
    """Build concise markdown interpretation."""
    best_overall = combined_ranking.sort_values("rank_mandate_aware").iloc[0]
    td3 = combined_ranking.loc[combined_ranking["strategy_type"] == "td3"]
    benchmarks = combined_ranking.loc[combined_ranking["strategy_type"] == "benchmark"]
    best_td3 = td3.sort_values("rank_mandate_aware").iloc[0]
    best_benchmark = benchmarks.sort_values("rank_mandate_aware").iloc[0]
    official_td3_leader = combined_ranking.loc[
        combined_ranking["strategy_name"]
        == "V3_real_macro_vintage_clean_no_dxy_cap_0p70"
    ]
    top_td3_name = str(best_td3["strategy_name"])
    top_td3_pairs = td3_vs_benchmarks.loc[
        td3_vs_benchmarks["td3_strategy"] == top_td3_name
    ]
    top_td3_beats_any_benchmark_sharpe = bool(
        top_td3_pairs["td3_beats_benchmark_by_sharpe"].any()
    )
    top_td3_beats_all_benchmark_sharpe = bool(
        top_td3_pairs["td3_beats_benchmark_by_sharpe"].all()
    )
    top_td3_lower_drawdown_any = bool(
        top_td3_pairs["td3_has_lower_drawdown_than_benchmark"].any()
    )
    top_td3_lower_drawdown_all = bool(
        top_td3_pairs["td3_has_lower_drawdown_than_benchmark"].all()
    )
    top_td3_beats_any_mandate = bool(
        top_td3_pairs["td3_beats_benchmark_by_mandate"].any()
    )
    top_td3_beats_all_mandate = bool(
        top_td3_pairs["td3_beats_benchmark_by_mandate"].all()
    )
    lines = [
        "# Asset-Specific-Cost TD3 vs Benchmark Comparison",
        "",
        "This report combines selected TD3 candidates and deterministic benchmarks "
        "under one common asset-specific transaction-cost framework.",
        "",
        "Cost model: SPY/TLT/GLD = 2 bps, BTC-USD = 10 bps, CASH = 0 bps.",
        "",
        "## Main Results",
        "",
        f"- Best overall by mandate-aware score: `{best_overall['strategy_name']}` "
        f"({best_overall['strategy_type']}, score {_fmt(best_overall['mandate_aware_score'])}).",
        f"- Best TD3 by recomputed combined-universe mandate-aware score: `{best_td3['strategy_name']}` "
        f"(score {_fmt(best_td3['mandate_aware_score'])}, Sharpe {_fmt(best_td3['sharpe'])}, "
        f"max drawdown {_fmt(best_td3['max_drawdown'])}).",
        f"- Best benchmark by mandate-aware score: `{best_benchmark['strategy_name']}` "
        f"(score {_fmt(best_benchmark['mandate_aware_score'])}, Sharpe {_fmt(best_benchmark['sharpe'])}, "
        f"max drawdown {_fmt(best_benchmark['max_drawdown'])}).",
        "",
        "## TD3 Leader Checks",
        "",
        f"- Top TD3 beats at least one benchmark by Sharpe: {top_td3_beats_any_benchmark_sharpe}.",
        f"- Top TD3 beats all benchmarks by Sharpe: {top_td3_beats_all_benchmark_sharpe}.",
        f"- Top TD3 has lower drawdown than at least one benchmark: {top_td3_lower_drawdown_any}.",
        f"- Top TD3 has lower drawdown than all benchmarks: {top_td3_lower_drawdown_all}.",
        f"- Top TD3 beats at least one benchmark by recomputed mandate-aware score: {top_td3_beats_any_mandate}.",
        f"- Top TD3 beats all benchmarks by recomputed mandate-aware score: {top_td3_beats_all_mandate}.",
        "",
        "## Interpretation",
        "",
        "The comparison is cost-consistent across TD3 and benchmarks, but it is "
        "still a reporting layer. It does not establish statistical superiority. "
        "Statistical validation, White Reality Check, regime analysis, and "
        "mandate-profile analysis should be regenerated under this same "
        "asset-specific cost model before paper-level claims are updated.",
        "",
        f"Scoring note: {metadata['scoring_note']}",
    ]
    if not official_td3_leader.empty:
        row = official_td3_leader.iloc[0]
        lines.insert(
            15,
            f"- Official TD3-only leader `V3_real_macro_vintage_clean_no_dxy_cap_0p70` "
            f"ranks {int(row['rank_mandate_aware'])} by recomputed combined-universe "
            f"mandate-aware score (score {_fmt(row['mandate_aware_score'])}).",
        )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_bps(value: dict | None) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(asset): float(cost) for asset, cost in sorted(value.items())}


def _safe_filename(value: str) -> str:
    return value.replace("/", "_")


def _mean_if_present(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _min_if_present(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").min())


def _num(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _fmt(value: Any) -> str:
    numeric = _num(value)
    if not np.isfinite(numeric):
        return "NA"
    return f"{numeric:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build asset-specific-cost TD3 vs benchmark comparison report.",
    )
    parser.add_argument("--td3-report-dir", default=DEFAULT_TD3_REPORT_DIR)
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = build_asset_specific_cost_benchmark_comparison_report(
        td3_report_dir=args.td3_report_dir,
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
    )
    print(report["combined_ranking"].head(20).to_string(index=False))
    print(f"Outputs written to {report['paths']['combined_ranking']}")


if __name__ == "__main__":
    main()
