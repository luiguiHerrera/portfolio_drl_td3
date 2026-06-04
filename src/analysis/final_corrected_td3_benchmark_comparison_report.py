"""Build corrected TD3-vs-benchmark comparison reports.

This module is reporting-only. It combines recovered corrected TD3 cap
sensitivity outputs with deterministic benchmark histories generated under the
matching cash-return assumption. It does not train or evaluate TD3 policies.
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
    calculate_recovery_required,
    get_drawdown_multiplier,
)
from src.analysis.robust_score import (
    compute_composite_robust_score,
    compute_deflated_sharpe_ratio,
)


ZERO_TD3_DIR = "outputs/tables/final_corrected_limited_td3_60ep_10seeds"
ZERO_BENCHMARK_DIR = (
    "/Users/thiagoherrera/Projects/portfolio_drl_outputs/"
    "final_corrected_zero_cash_benchmark_comparison/benchmarks"
)
ZERO_OUTPUT_DIR = (
    "/Users/thiagoherrera/Projects/portfolio_drl_outputs/"
    "final_corrected_zero_cash_benchmark_comparison"
)
BIL_TD3_DIR = (
    "/Users/thiagoherrera/Projects/portfolio_drl_outputs/"
    "final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds"
)
BIL_BENCHMARK_DIR = (
    "/Users/thiagoherrera/Projects/portfolio_drl_outputs/"
    "final_corrected_bil_cash_benchmark_comparison/benchmarks"
)
BIL_OUTPUT_DIR = (
    "/Users/thiagoherrera/Projects/portfolio_drl_outputs/"
    "final_corrected_bil_cash_benchmark_comparison"
)
PERIODS_PER_YEAR = 52
EXPECTED_ASSETS = ["SPY", "TLT", "GLD", "BTC-USD", "CASH"]
EXPECTED_BENCHMARK_COUNT = 14
EXPECTED_TD3_COUNT = 5
BROKER_COST_CAVEAT = (
    "Broker/exchange-style trading-cost proxy only; does not model fiat ramps, "
    "exchange transfers, withdrawal fees, custody frictions, taxes, market "
    "impact, or delays."
)
REQUIRED_BENCHMARK_HISTORY_COLUMNS = [
    "transaction_cost_mode",
    "transaction_cost",
    "turnover",
    "financial_net_return",
    "weight_SPY",
    "weight_TLT",
    "weight_GLD",
    "weight_BTC-USD",
    "weight_CASH",
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
    "average_effective_assets",
    "average_max_weight",
    "mean_cash_weight",
    "total_transaction_cost",
    "mean_transaction_cost",
    "mean_btc_weight",
    "mean_cash_transaction_cost_contribution",
    "mean_btc_transaction_cost_contribution",
    "dsr_score",
    "dsr_method",
]


def build_report_pair(
    zero_td3_dir: str = ZERO_TD3_DIR,
    zero_benchmark_dir: str = ZERO_BENCHMARK_DIR,
    zero_output_dir: str = ZERO_OUTPUT_DIR,
    bil_td3_dir: str = BIL_TD3_DIR,
    bil_benchmark_dir: str = BIL_BENCHMARK_DIR,
    bil_output_dir: str = BIL_OUTPUT_DIR,
) -> dict[str, dict[str, Any]]:
    """Build both zero-CASH and BIL-CASH comparison reports."""
    zero = build_single_report(
        cash_label="zero_cash",
        td3_dir=zero_td3_dir,
        benchmark_dir=zero_benchmark_dir,
        output_dir=zero_output_dir,
        expected_cash_bps=0.0,
        expected_returns_path_contains="returns_weekly_latest.csv",
    )
    bil = build_single_report(
        cash_label="bil_cash",
        td3_dir=bil_td3_dir,
        benchmark_dir=bil_benchmark_dir,
        output_dir=bil_output_dir,
        expected_cash_bps=2.0,
        expected_returns_path_contains="returns_weekly_latest_cash_bil_proxy.csv",
    )
    return {"zero_cash": zero, "bil_cash": bil}


def build_single_report(
    cash_label: str,
    td3_dir: str,
    benchmark_dir: str,
    output_dir: str,
    expected_cash_bps: float,
    expected_returns_path_contains: str,
) -> dict[str, Any]:
    """Build one corrected TD3-vs-benchmark report."""
    td3_path = Path(td3_dir).expanduser()
    benchmark_path = Path(benchmark_dir).expanduser()
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    benchmark_metadata = _load_json(benchmark_path / "benchmark_metadata.json")
    validate_benchmark_cost_model(
        benchmark_metadata=benchmark_metadata,
        expected_cash_bps=expected_cash_bps,
        expected_returns_path_contains=expected_returns_path_contains,
    )
    td3_rows = load_selected_td3_rows(td3_path)
    benchmark_rows = load_benchmark_rows(benchmark_path)
    validate_counts(td3_rows, benchmark_rows)

    combined = pd.concat([td3_rows, benchmark_rows], ignore_index=True, sort=False)
    combined = add_combined_scores(combined)
    combined = add_ranks(combined)
    ranking = select_ranking_columns(combined)
    td3_vs_benchmarks = build_td3_vs_benchmark_table(ranking)
    metadata = build_metadata(
        cash_label=cash_label,
        td3_dir=str(td3_path),
        benchmark_dir=str(benchmark_path),
        output_dir=str(output_path),
        expected_cash_bps=expected_cash_bps,
        benchmark_metadata=benchmark_metadata,
        ranking=ranking,
    )
    summary = build_summary_markdown(
        cash_label=cash_label,
        ranking=ranking,
        td3_vs_benchmarks=td3_vs_benchmarks,
        metadata=metadata,
    )

    prefix = f"final_corrected_{cash_label}"
    paths = {
        "combined_ranking": output_path / f"{prefix}_combined_ranking.csv",
        "td3_vs_benchmarks": output_path / f"{prefix}_td3_vs_benchmarks.csv",
        "summary": output_path / f"{prefix}_benchmark_comparison_summary.md",
        "metadata": output_path / f"{prefix}_benchmark_comparison_metadata.json",
    }
    ranking.to_csv(paths["combined_ranking"], index=False)
    td3_vs_benchmarks.to_csv(paths["td3_vs_benchmarks"], index=False)
    paths["summary"].write_text(summary, encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "combined_ranking": ranking,
        "td3_vs_benchmarks": td3_vs_benchmarks,
        "metadata": metadata,
        "summary": summary,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def load_selected_td3_rows(td3_dir: Path) -> pd.DataFrame:
    """Load best-by-mandate TD3 cap rows from recovered cap sensitivity files."""
    best_path = td3_dir / "cap_sensitivity_best_caps.csv"
    all_path = td3_dir / "cap_sensitivity_all_results.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing TD3 best caps table: {best_path}")
    if not all_path.exists():
        raise FileNotFoundError(f"Missing TD3 all-results table: {all_path}")
    best = pd.read_csv(best_path)
    all_results = pd.read_csv(all_path)
    if best.empty or all_results.empty:
        raise ValueError("TD3 cap sensitivity files must not be empty.")

    rows = []
    for _, best_row in best.iterrows():
        base_candidate = str(best_row["base_candidate"])
        cap_label = _cap_to_label(best_row["best_by_mandate_aware_score"])
        matches = all_results.loc[
            (all_results["base_candidate"].astype(str) == base_candidate)
            & (all_results["cap_label"].astype(str).map(_normalize_cap_label) == cap_label)
        ]
        if matches.empty:
            raise ValueError(
                f"No TD3 all-results row for {base_candidate} selected cap {cap_label}."
            )
        rows.append(matches.sort_values("mandate_aware_score", ascending=False).iloc[0])

    result = pd.DataFrame(rows).reset_index(drop=True)
    result["strategy_name"] = result["candidate_name"]
    result["strategy_type"] = "TD3"
    result["transaction_cost_mode"] = "asset_specific"
    result["average_effective_assets"] = result[
        "average_effective_number_of_assets"
    ]
    result["total_transaction_cost"] = pd.NA
    result["dsr_method"] = result.get("dsr_method", "td3_recovered_report_score_inputs")
    if "dsr_score" not in result:
        result["dsr_score"] = np.nan
    if "mean_btc_weight" not in result:
        result["mean_btc_weight"] = np.nan
    if "mean_cash_transaction_cost_contribution" not in result:
        result["mean_cash_transaction_cost_contribution"] = np.nan
    if "mean_btc_transaction_cost_contribution" not in result:
        result["mean_btc_transaction_cost_contribution"] = np.nan
    return result


def load_benchmark_rows(benchmark_dir: Path) -> pd.DataFrame:
    """Load deterministic benchmark metrics and validate asset-specific histories."""
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
    if metrics.empty:
        raise ValueError("Benchmark metrics table must not be empty.")
    result = metrics.merge(
        diagnostics[["benchmark_name", "transaction_cost_mode"]],
        on="benchmark_name",
        how="left",
    )

    dsr_scores = []
    mean_btc_weights = []
    cash_cost_contributions = []
    btc_cost_contributions = []
    worst_drawdowns = []
    for benchmark_name in result["benchmark_name"]:
        history_path = histories_dir / f"{_safe_filename(str(benchmark_name))}_history.csv"
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
        cash_cost_contributions.append(
            _mean_if_present(history, "asset_transaction_cost_contribution_CASH")
        )
        btc_cost_contributions.append(
            _mean_if_present(history, "asset_transaction_cost_contribution_BTC-USD")
        )
        worst_drawdowns.append(_min_if_present(history, "drawdown"))

    result["strategy_name"] = result["benchmark_name"]
    result["strategy_type"] = "benchmark"
    result["base_candidate"] = pd.NA
    result["cap_label"] = pd.NA
    result["average_effective_assets"] = result[
        "average_effective_number_of_assets"
    ]
    result["mean_transaction_cost"] = result["average_transaction_cost"]
    result["worst_max_drawdown"] = pd.Series(worst_drawdowns).combine_first(
        result["max_drawdown"],
    )
    result["mean_btc_weight"] = mean_btc_weights
    result["mean_cash_transaction_cost_contribution"] = cash_cost_contributions
    result["mean_btc_transaction_cost_contribution"] = btc_cost_contributions
    result["dsr_score"] = dsr_scores
    result["dsr_method"] = "benchmark_history_dsr_n25"
    result["cash_above_10_rate"] = result.get("cash_above_10pct", 0.0)
    return result


def validate_benchmark_cost_model(
    benchmark_metadata: dict[str, Any],
    expected_cash_bps: float,
    expected_returns_path_contains: str,
) -> None:
    """Validate benchmark report cash assumption and transaction-cost mapping."""
    if benchmark_metadata.get("transaction_cost_mode") != "asset_specific":
        raise ValueError("Benchmark report is not asset-specific-cost aware.")
    returns_path = str(benchmark_metadata.get("returns_path", ""))
    if expected_returns_path_contains not in returns_path:
        raise ValueError(
            "Benchmark returns path does not match requested cash assumption: "
            f"{returns_path}"
        )
    mapping = _normalize_bps(benchmark_metadata.get("asset_transaction_cost_bps"))
    expected = {
        "SPY": 2.0,
        "TLT": 2.0,
        "GLD": 2.0,
        "BTC-USD": 10.0,
        "CASH": float(expected_cash_bps),
    }
    if mapping != expected:
        raise ValueError(
            f"Benchmark asset cost map mismatch. Expected {expected}, found {mapping}."
        )


def validate_benchmark_history(history: pd.DataFrame, path: Path) -> None:
    """Validate benchmark history diagnostics."""
    missing = [column for column in REQUIRED_BENCHMARK_HISTORY_COLUMNS if column not in history]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    modes = set(history["transaction_cost_mode"].dropna().astype(str).unique())
    if modes != {"asset_specific"}:
        raise ValueError(f"{path} contains non-asset-specific modes: {sorted(modes)}")


def validate_counts(td3_rows: pd.DataFrame, benchmark_rows: pd.DataFrame) -> None:
    """Validate expected selected TD3 and benchmark counts."""
    if len(td3_rows) != EXPECTED_TD3_COUNT:
        raise ValueError(f"Expected {EXPECTED_TD3_COUNT} selected TD3 rows, found {len(td3_rows)}.")
    if len(benchmark_rows) != EXPECTED_BENCHMARK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_BENCHMARK_COUNT} benchmark rows, found {len(benchmark_rows)}."
        )


def add_combined_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute robust and mandate-aware scores over the combined universe."""
    result = frame.copy()
    scoring_input = result.rename(
        columns={
            "strategy_name": "strategy",
            "worst_max_drawdown": "worst_drawdown",
            "average_turnover": "turnover",
            "average_effective_assets": "effective_assets",
        },
    )
    scoring_input["type"] = scoring_input["strategy_type"].str.lower()
    scored = compute_composite_robust_score(scoring_input)
    result = scored.rename(
        columns={
            "strategy": "strategy_name",
            "worst_drawdown": "worst_max_drawdown",
            "turnover": "average_turnover",
            "effective_assets": "average_effective_assets",
        },
    )
    result["mandate_bucket"] = result["max_drawdown"].apply(assign_drawdown_bucket)
    result["recovery_required"] = result["max_drawdown"].apply(
        calculate_recovery_required
    )
    result["drawdown_multiplier"] = result["max_drawdown"].apply(
        get_drawdown_multiplier
    )
    result["mandate_aware_score"] = result["robust_score"] * result[
        "drawdown_multiplier"
    ]
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
    """Build TD3-vs-benchmark pairwise metric deltas."""
    td3 = combined_ranking.loc[combined_ranking["strategy_type"] == "TD3"]
    benchmarks = combined_ranking.loc[combined_ranking["strategy_type"] == "benchmark"]
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
    cash_label: str,
    td3_dir: str,
    benchmark_dir: str,
    output_dir: str,
    expected_cash_bps: float,
    benchmark_metadata: dict[str, Any],
    ranking: pd.DataFrame,
) -> dict[str, Any]:
    """Build reproducibility metadata."""
    return {
        "runner": "src.analysis.final_corrected_td3_benchmark_comparison_report",
        "cash_label": cash_label,
        "td3_dir": td3_dir,
        "benchmark_dir": benchmark_dir,
        "output_dir": output_dir,
        "transaction_cost_mode": "asset_specific",
        "asset_transaction_cost_bps": {
            "SPY": 2.0,
            "TLT": 2.0,
            "GLD": 2.0,
            "BTC-USD": 10.0,
            "CASH": float(expected_cash_bps),
        },
        "benchmark_returns_path": benchmark_metadata.get("returns_path"),
        "benchmark_names": benchmark_metadata.get("benchmark_names", []),
        "n_td3_selected": int((ranking["strategy_type"] == "TD3").sum()),
        "n_benchmarks": int((ranking["strategy_type"] == "benchmark").sum()),
        "score_scope": f"{cash_label}_selected_td3_plus_benchmarks",
        "scoring_note": (
            "robust_score and mandate_aware_score are recomputed over selected "
            "TD3 candidates plus deterministic benchmarks for this cash assumption. "
            "This is a ranking diagnostic, not statistical superiority evidence."
        ),
        "validation": {
            "no_cross_cash_mixing": True,
            "benchmark_count_expected": EXPECTED_BENCHMARK_COUNT,
            "selected_td3_count_expected": EXPECTED_TD3_COUNT,
            "histories_asset_specific_cost_consistent": True,
        },
        "cost_caveat": BROKER_COST_CAVEAT,
    }


def build_summary_markdown(
    cash_label: str,
    ranking: pd.DataFrame,
    td3_vs_benchmarks: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    """Build concise markdown interpretation."""
    label = "Zero-CASH" if cash_label == "zero_cash" else "BIL-CASH"
    best_overall = ranking.sort_values("rank_mandate_aware").iloc[0]
    td3 = ranking.loc[ranking["strategy_type"] == "TD3"]
    benchmarks = ranking.loc[ranking["strategy_type"] == "benchmark"]
    best_td3 = td3.sort_values("rank_mandate_aware").iloc[0]
    best_benchmark = benchmarks.sort_values("rank_mandate_aware").iloc[0]
    top_td3_pairs = td3_vs_benchmarks.loc[
        td3_vs_benchmarks["td3_strategy"] == best_td3["strategy_name"]
    ]
    beats_all_benchmarks = bool(top_td3_pairs["td3_beats_benchmark_by_mandate"].all())
    beats_any_benchmark = bool(top_td3_pairs["td3_beats_benchmark_by_mandate"].any())
    lines = [
        f"# Final Corrected {label} TD3 vs Benchmark Comparison",
        "",
        "This report combines recovered corrected TD3 selected caps with deterministic "
        "benchmark histories under the same cash-return and asset-specific "
        "transaction-cost assumption.",
        "",
        "## Main Results",
        "",
        f"- Best overall by recomputed mandate-aware score: `{best_overall['strategy_name']}` "
        f"({best_overall['strategy_type']}, score {_fmt(best_overall['mandate_aware_score'])}).",
        f"- Best TD3: `{best_td3['strategy_name']}` "
        f"(score {_fmt(best_td3['mandate_aware_score'])}, robust {_fmt(best_td3['robust_score'])}, "
        f"Sharpe {_fmt(best_td3['sharpe'])}, max drawdown {_fmt(best_td3['max_drawdown'])}).",
        f"- Best benchmark: `{best_benchmark['strategy_name']}` "
        f"(score {_fmt(best_benchmark['mandate_aware_score'])}, robust {_fmt(best_benchmark['robust_score'])}, "
        f"Sharpe {_fmt(best_benchmark['sharpe'])}, max drawdown {_fmt(best_benchmark['max_drawdown'])}).",
        f"- TD3 beats every benchmark by recomputed mandate-aware score: {beats_all_benchmarks}.",
        f"- TD3 beats at least one benchmark by recomputed mandate-aware score: {beats_any_benchmark}.",
        "",
        "## Interpretation",
        "",
        "This is a cost-consistent comparison report, not statistical superiority "
        "evidence. It should be paired with bootstrap/WRC validation before any "
        "paper-level claim about benchmark outperformance.",
        "",
        f"Scoring note: {metadata['scoring_note']}",
        f"Cost caveat: {metadata['cost_caveat']}",
    ]
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_bps(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(asset): float(value[asset]) for asset in sorted(value)}


def _safe_filename(value: str) -> str:
    return value.replace("/", "_")


def _cap_to_label(value: Any) -> str:
    text = str(value)
    if text.lower() == "nan":
        return "uncapped"
    if text == "uncapped":
        return "uncapped"
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return text
    return f"{float(numeric):.2f}"


def _normalize_cap_label(value: Any) -> str:
    text = str(value)
    if text.lower() in {"nan", "none"}:
        return "uncapped"
    if text == "uncapped":
        return "uncapped"
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return f"{float(numeric):.2f}"
    return text


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
        description="Build final corrected Zero-CASH and BIL-CASH TD3 vs benchmark reports.",
    )
    parser.add_argument("--zero-td3-dir", default=ZERO_TD3_DIR)
    parser.add_argument("--zero-benchmark-dir", default=ZERO_BENCHMARK_DIR)
    parser.add_argument("--zero-output-dir", default=ZERO_OUTPUT_DIR)
    parser.add_argument("--bil-td3-dir", default=BIL_TD3_DIR)
    parser.add_argument("--bil-benchmark-dir", default=BIL_BENCHMARK_DIR)
    parser.add_argument("--bil-output-dir", default=BIL_OUTPUT_DIR)
    args = parser.parse_args()

    result = build_report_pair(
        zero_td3_dir=args.zero_td3_dir,
        zero_benchmark_dir=args.zero_benchmark_dir,
        zero_output_dir=args.zero_output_dir,
        bil_td3_dir=args.bil_td3_dir,
        bil_benchmark_dir=args.bil_benchmark_dir,
        bil_output_dir=args.bil_output_dir,
    )
    for label, report in result.items():
        ranking = report["combined_ranking"]
        best = ranking.sort_values("rank_mandate_aware").iloc[0]
        print(
            f"{label}: best overall {best['strategy_name']} "
            f"({best['strategy_type']}), mandate-aware {_fmt(best['mandate_aware_score'])}"
        )


if __name__ == "__main__":
    main()
