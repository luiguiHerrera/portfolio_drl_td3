"""Regime analysis for final constrained TD3 reports.

This module is reporting-only. It reads existing benchmark histories and TD3
test policy histories, date-averages TD3 fold/seed returns, and computes
regime-level metrics without retraining or changing production scoring logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.mandate_aware_score import assign_drawdown_bucket, get_drawdown_multiplier


DEFAULT_FINAL_REPORT_DIR = "outputs/tables/final_constrained_td3_report_with_v3_v4_v7_v8_60ep_10seeds"
DEFAULT_OUTPUT_DIR = "outputs/tables/regime_analysis_final_v3_v4"
PERIODS_PER_YEAR = 52
RETURN_COLUMNS = ("financial_net_return", "net_return", "portfolio_return")
ASSET_SPECIFIC_SELECTED_FILE = "asset_specific_cost_selected_candidates.csv"
ASSET_SPECIFIC_METADATA_FILE = "asset_specific_cost_metadata.json"
ASSET_SPECIFIC_COMBINED_RANKING_FILE = "asset_specific_cost_combined_ranking.csv"

REGIME_DEFINITIONS = [
    ("covid_shock_recovery", "COVID shock / recovery", "2020-02-01", "2020-12-31"),
    ("post_covid_liquidity", "Post-COVID liquidity", "2021-01-01", "2021-12-31"),
    ("inflation_hiking_shock", "Inflation / hiking shock", "2022-01-01", "2022-12-31"),
    ("ai_risk_on_recovery", "AI / risk-on recovery", "2023-01-01", "2024-06-30"),
    ("recent_late_test_window", "Recent / late test window", "2024-07-01", "2026-05-15"),
    ("calendar_2022", "Calendar 2022", "2022-01-01", "2022-12-31"),
    ("calendar_2023", "Calendar 2023", "2023-01-01", "2023-12-31"),
    ("calendar_2024", "Calendar 2024", "2024-01-01", "2024-12-31"),
    ("calendar_2025", "Calendar 2025", "2025-01-01", "2025-12-31"),
    ("calendar_2026_ytd", "Calendar 2026 YTD", "2026-01-01", "2026-05-15"),
]

DEFAULT_BENCHMARKS = [
    "BuyHold_GLD",
    "trend_spy_cash_12p",
    "rolling_risk_parity_inverse_vol_12p",
    "rolling_markowitz_min_variance_52p",
    "defensive_risk_off_12p",
    "60_40_SPY_TLT",
]

PAIRWISE_COMPARISONS = [
    ("V5_no_volatility_block_cap_0p50", "trend_spy_cash_12p"),
    ("V5_no_volatility_block_cap_0p50", "BuyHold_GLD"),
    ("V5_no_volatility_block_cap_0p50", "Equal_Weight"),
    ("V5_no_volatility_block_cap_0p50", "V3_real_macro_vintage_clean_no_dxy_cap_0p70"),
    ("V5_no_volatility_block_cap_0p50", "V4_real_garch_current_cap_0p50"),
    ("V3_real_macro_vintage_clean_no_dxy_cap_0.50", "BuyHold_GLD"),
    ("V3_real_macro_vintage_clean_no_dxy_cap_0.50", "trend_spy_cash_12p"),
    ("V3_real_macro_vintage_clean_no_dxy_cap_0.50", "V4_cap_0.50"),
    ("V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50", "BuyHold_GLD"),
    ("V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50", "trend_spy_cash_12p"),
    ("V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50", "V4_cap_0.50"),
    (
        "V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50",
        "V3_real_macro_vintage_clean_no_dxy_cap_0.50",
    ),
    ("V3_cap_0.60", "BuyHold_GLD"),
    ("V3_cap_0.60", "trend_spy_cash_12p"),
    ("V4_cap_0.50", "BuyHold_GLD"),
    ("V4_cap_0.50", "trend_spy_cash_12p"),
    ("V3_cap_0.60", "V4_cap_0.50"),
]


def build_regime_analysis_report(
    final_report_dir: str = DEFAULT_FINAL_REPORT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    v3_cap_sensitivity_dir: str | None = None,
    v3_vintage_cap_sensitivity_dir: str | None = None,
    v3_clean_no_dxy_cap_sensitivity_dir: str | None = None,
    v4_cap_sensitivity_dir: str | None = None,
    v7_cap_sensitivity_dir: str | None = None,
    v7_clean_no_dxy_garch_cap_sensitivity_dir: str | None = None,
    v8_cap_sensitivity_dir: str | None = None,
    benchmark_dir: str | None = None,
    asset_specific_only: bool | None = None,
) -> dict[str, Any]:
    """Build and write regime analysis outputs."""
    final_path = Path(final_report_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    histories, history_sources, warnings = locate_and_load_histories(
        final_path,
        v3_cap_sensitivity_dir=v3_cap_sensitivity_dir,
        v3_vintage_cap_sensitivity_dir=v3_vintage_cap_sensitivity_dir,
        v3_clean_no_dxy_cap_sensitivity_dir=v3_clean_no_dxy_cap_sensitivity_dir,
        v4_cap_sensitivity_dir=v4_cap_sensitivity_dir,
        v7_cap_sensitivity_dir=v7_cap_sensitivity_dir,
        v7_clean_no_dxy_garch_cap_sensitivity_dir=(
            v7_clean_no_dxy_garch_cap_sensitivity_dir
        ),
        v8_cap_sensitivity_dir=v8_cap_sensitivity_dir,
        benchmark_dir=benchmark_dir,
        asset_specific_only=asset_specific_only,
    )
    metrics = build_regime_strategy_metrics(histories)
    rankings = build_regime_strategy_rankings(metrics)
    pairwise = build_regime_pairwise_comparisons(
        metrics,
        _pairwise_comparisons_for_histories(histories),
    )
    winners = build_regime_winners_summary(metrics, rankings)
    metadata = build_metadata(
        final_report_dir=final_report_dir,
        output_dir=output_dir,
        histories=histories,
        history_sources=history_sources,
        warnings=warnings,
    )
    summary = build_summary_markdown(metrics, rankings, pairwise, winners, warnings)

    paths = {
        "metrics": output_path / "regime_strategy_metrics.csv",
        "rankings": output_path / "regime_strategy_rankings.csv",
        "pairwise": output_path / "regime_pairwise_comparisons.csv",
        "winners": output_path / "regime_winners_summary.csv",
        "metadata": output_path / "regime_analysis_metadata.json",
        "summary": output_path / "regime_analysis_summary.md",
    }
    metrics.to_csv(paths["metrics"], index=False)
    rankings.to_csv(paths["rankings"], index=False)
    pairwise.to_csv(paths["pairwise"], index=False)
    winners.to_csv(paths["winners"], index=False)
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    paths["summary"].write_text(summary, encoding="utf-8")

    return {
        "histories": histories,
        "history_sources": history_sources,
        "warnings": warnings,
        "metrics": metrics,
        "rankings": rankings,
        "pairwise": pairwise,
        "winners": winners,
        "metadata": metadata,
        "summary": summary,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def locate_and_load_histories(
    final_report_dir: Path,
    *,
    v3_cap_sensitivity_dir: str | None = None,
    v3_vintage_cap_sensitivity_dir: str | None = None,
    v3_clean_no_dxy_cap_sensitivity_dir: str | None = None,
    v4_cap_sensitivity_dir: str | None = None,
    v7_cap_sensitivity_dir: str | None = None,
    v7_clean_no_dxy_garch_cap_sensitivity_dir: str | None = None,
    v8_cap_sensitivity_dir: str | None = None,
    benchmark_dir: str | None = None,
    asset_specific_only: bool | None = None,
) -> tuple[dict[str, pd.Series], dict[str, Any], list[str]]:
    """Load benchmark histories and date-averaged TD3 histories."""
    selected, metadata, report_mode = _load_selected_and_metadata(final_report_dir)
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
    require_asset_specific = (
        report_mode == "asset_specific" if asset_specific_only is None else asset_specific_only
    )
    histories: dict[str, pd.Series] = {}
    sources: dict[str, Any] = {}
    warnings: list[str] = []

    for _, row in selected.iterrows():
        strategy_name = str(row["strategy_name"])
        base_candidate = str(row["base_candidate"])
        cap = row.get("selected_cap")
        source_dir = _source_dir_for_candidate(base_candidate, metadata)
        if source_dir is None:
            warnings.append(f"No source directory configured for {strategy_name}.")
            continue
        try:
            series, files = load_td3_date_averaged_history(
                Path(source_dir),
                base_candidate,
                cap,
                require_asset_specific=require_asset_specific,
            )
        except FileNotFoundError as exc:
            warnings.append(str(exc))
            continue
        histories[strategy_name] = series
        sources[strategy_name] = {
            "history_type": "td3_date_averaged_test_policy_history",
            "source_dir": source_dir,
            "n_history_files": len(files),
            "history_files_sample": [str(path) for path in files[:5]],
        }

    benchmark_history_dir = _benchmark_history_dir(metadata)
    for benchmark_name in _benchmark_names_for_report(final_report_dir):
        path = benchmark_history_dir / f"{benchmark_name}_history.csv"
        if not path.exists():
            warnings.append(f"Missing benchmark history for {benchmark_name}: {path}")
            continue
        series = read_return_history(path, require_asset_specific=require_asset_specific)
        histories[benchmark_name] = series
        sources[benchmark_name] = {
            "history_type": "benchmark_single_history",
            "source_file": str(path),
            "n_history_files": 1,
        }

    return histories, sources, warnings


def load_td3_date_averaged_history(
    source_dir: Path,
    base_candidate: str,
    cap: Any,
    require_asset_specific: bool = False,
) -> tuple[pd.Series, list[Path]]:
    """Load and date-average TD3 test returns for one base candidate and cap."""
    per_candidate_dir = source_dir / "per_candidate" / base_candidate
    if not per_candidate_dir.exists():
        raise FileNotFoundError(f"Missing per-candidate history dir: {per_candidate_dir}")
    cap_label = cap_to_label(cap)
    pattern = f"*{base_candidate}_cap_{cap_label}_seed_*/test_policy_history.csv"
    files = sorted(per_candidate_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No test_policy_history files found for {base_candidate} cap {cap_label} in {per_candidate_dir}"
        )
    frames = []
    for path in files:
        series = read_return_history(path, require_asset_specific=require_asset_specific)
        frames.append(series.rename("return").reset_index())
    stacked = pd.concat(frames, ignore_index=True)
    averaged = stacked.groupby("date", as_index=True)["return"].mean().sort_index()
    averaged.name = "return"
    return averaged, files


def read_return_history(path: Path, require_asset_specific: bool = False) -> pd.Series:
    """Read a dated return series from a policy or benchmark history CSV."""
    frame = pd.read_csv(path)
    if require_asset_specific:
        _validate_asset_specific_history(frame, path)
    if "date" not in frame.columns:
        raise ValueError(f"History file lacks date column: {path}")
    return_column = next((column for column in RETURN_COLUMNS if column in frame.columns), None)
    if return_column is None:
        raise ValueError(f"History file lacks usable return column: {path}")
    dates = pd.to_datetime(frame["date"])
    returns = pd.to_numeric(frame[return_column], errors="coerce")
    result = pd.Series(returns.to_numpy(dtype=float), index=dates, name="return")
    result = result.dropna().sort_index()
    result.index.name = "date"
    return result


def build_regime_strategy_metrics(histories: dict[str, pd.Series]) -> pd.DataFrame:
    """Calculate regime metrics for each strategy."""
    rows = []
    for regime_id, regime_name, start, end in REGIME_DEFINITIONS:
        for strategy_name, returns in histories.items():
            sliced = slice_returns(returns, start, end)
            row = calculate_return_metrics(sliced)
            row.update(
                {
                    "regime_id": regime_id,
                    "regime_name": regime_name,
                    "start_date": start,
                    "end_date": end,
                    "strategy_name": strategy_name,
                }
            )
            rows.append(row)
    columns = [
        "regime_id",
        "regime_name",
        "start_date",
        "end_date",
        "strategy_name",
        "n_observations",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "average_return",
        "hit_rate",
        "worst_period_return",
        "mandate_bucket",
        "mandate_style_score",
        "simple_composite_score",
    ]
    return pd.DataFrame(rows).loc[:, columns]


def build_regime_strategy_rankings(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank strategies within each regime by several criteria."""
    ranking = metrics.copy()
    ranking["rank_sharpe"] = ranking.groupby("regime_id")["sharpe"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    ranking["rank_max_drawdown"] = ranking.groupby("regime_id")["max_drawdown"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    ranking["rank_cumulative_return"] = ranking.groupby("regime_id")[
        "cumulative_return"
    ].rank(ascending=False, method="min", na_option="bottom")
    ranking["rank_mandate_style_score"] = ranking.groupby("regime_id")[
        "mandate_style_score"
    ].rank(ascending=False, method="min", na_option="bottom")
    ranking["rank_simple_composite"] = ranking.groupby("regime_id")[
        "simple_composite_score"
    ].rank(ascending=False, method="min", na_option="bottom")
    rank_columns = [
        "rank_sharpe",
        "rank_max_drawdown",
        "rank_cumulative_return",
        "rank_mandate_style_score",
        "rank_simple_composite",
    ]
    for column in rank_columns:
        ranking[column] = ranking[column].astype("Int64")
    return ranking.sort_values(
        ["regime_id", "rank_mandate_style_score", "rank_sharpe", "strategy_name"]
    ).reset_index(drop=True)


def build_regime_pairwise_comparisons(
    metrics: pd.DataFrame,
    comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    """Build regime-level pairwise comparisons."""
    indexed = metrics.set_index(["regime_id", "strategy_name"])
    rows = []
    for regime_id in metrics["regime_id"].drop_duplicates():
        regime_name = metrics.loc[metrics["regime_id"] == regime_id, "regime_name"].iloc[0]
        for left, right in comparisons:
            left_key = (regime_id, left)
            right_key = (regime_id, right)
            if left_key not in indexed.index or right_key not in indexed.index:
                rows.append(
                    {
                        "regime_id": regime_id,
                        "regime_name": regime_name,
                        "left_strategy": left,
                        "right_strategy": right,
                        "comparison_available": False,
                        "reason": "missing_strategy_metrics",
                    }
                )
                continue
            left_row = indexed.loc[left_key]
            right_row = indexed.loc[right_key]
            left_n = int(left_row["n_observations"])
            right_n = int(right_row["n_observations"])
            available = left_n > 0 and right_n > 0
            rows.append(
                {
                    "regime_id": regime_id,
                    "regime_name": regime_name,
                    "left_strategy": left,
                    "right_strategy": right,
                    "comparison_available": available,
                    "reason": "" if available else "insufficient_observations",
                    "left_n_observations": left_n,
                    "right_n_observations": right_n,
                    "delta_cumulative_return": _delta(left_row, right_row, "cumulative_return"),
                    "delta_annualized_return": _delta(left_row, right_row, "annualized_return"),
                    "delta_sharpe": _delta(left_row, right_row, "sharpe"),
                    "delta_max_drawdown": _delta(left_row, right_row, "max_drawdown"),
                    "delta_mandate_style_score": _delta(
                        left_row,
                        right_row,
                        "mandate_style_score",
                    ),
                    "left_beats_right_by_sharpe": _beats(left_row, right_row, "sharpe"),
                    "left_beats_right_by_drawdown": _beats(
                        left_row,
                        right_row,
                        "max_drawdown",
                    ),
                    "left_beats_right_by_mandate_style": _beats(
                        left_row,
                        right_row,
                        "mandate_style_score",
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_regime_winners_summary(
    metrics: pd.DataFrame,
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize winners by regime."""
    rows = []
    for regime_id, group in rankings.groupby("regime_id", sort=False):
        valid = group[group["n_observations"] > 0].copy()
        regime_name = group["regime_name"].iloc[0]
        if valid.empty:
            rows.append(
                {
                    "regime_id": regime_id,
                    "regime_name": regime_name,
                    "n_strategies_with_data": 0,
                    "best_by_sharpe": pd.NA,
                    "best_by_max_drawdown": pd.NA,
                    "best_by_cumulative_return": pd.NA,
                    "best_by_mandate_style_score": pd.NA,
                    "best_by_simple_composite": pd.NA,
                    "v3_rank_mandate_style": pd.NA,
                    "v4_rank_mandate_style": pd.NA,
                    "v3_minus_v4_mandate_style": pd.NA,
                }
            )
            continue
        by_strategy = valid.set_index("strategy_name")
        primary_td3 = _primary_td3_name(valid)
        secondary_td3 = _secondary_td3_name(valid)
        rows.append(
            {
                "regime_id": regime_id,
                "regime_name": regime_name,
                "n_strategies_with_data": int(len(valid)),
                "best_by_sharpe": _best_strategy(valid, "sharpe"),
                "best_by_max_drawdown": _best_strategy(valid, "max_drawdown"),
                "best_by_cumulative_return": _best_strategy(valid, "cumulative_return"),
                "best_by_mandate_style_score": _best_strategy(valid, "mandate_style_score"),
                "best_by_simple_composite": _best_strategy(valid, "simple_composite_score"),
                "v3_rank_mandate_style": _rank_for(valid, primary_td3, "rank_mandate_style_score"),
                "v4_rank_mandate_style": _rank_for(valid, secondary_td3, "rank_mandate_style_score"),
                "v3_minus_v4_mandate_style": (
                    _value_for(by_strategy, primary_td3, "mandate_style_score")
                    - _value_for(by_strategy, secondary_td3, "mandate_style_score")
                ),
            }
        )
    return pd.DataFrame(rows)


def calculate_return_metrics(returns: pd.Series) -> dict[str, Any]:
    """Calculate core performance metrics from a dated return series."""
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    n_obs = int(len(returns))
    if n_obs == 0:
        return _empty_metrics()
    cumulative_return = float((1.0 + returns).prod() - 1.0)
    annualized_return = _annualized_return(cumulative_return, n_obs)
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if n_obs > 1 else 0.0
    sharpe = _safe_ratio(annualized_return, annualized_volatility)
    downside = returns[returns < 0.0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(downside) > 1 else 0.0
    sortino = _safe_ratio(annualized_return, downside_vol)
    max_drawdown = calculate_max_drawdown(returns)
    calmar = _safe_ratio(annualized_return, abs(max_drawdown))
    hit_rate = float((returns > 0.0).mean())
    average_return = float(returns.mean())
    worst_period_return = float(returns.min())
    mandate_bucket = assign_drawdown_bucket(max_drawdown)
    mandate_style_score = _mandate_style_score(sharpe, sortino, calmar, max_drawdown)
    simple_composite_score = _simple_composite_score(
        sharpe=sharpe,
        cumulative_return=cumulative_return,
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
    )
    return {
        "n_observations": n_obs,
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "average_return": average_return,
        "hit_rate": hit_rate,
        "worst_period_return": worst_period_return,
        "mandate_bucket": mandate_bucket,
        "mandate_style_score": mandate_style_score,
        "simple_composite_score": simple_composite_score,
    }


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate max drawdown from periodic returns."""
    if len(returns) == 0:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def slice_returns(returns: pd.Series, start: str, end: str) -> pd.Series:
    """Slice returns inclusively by date."""
    return returns.loc[(returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))]


def build_metadata(
    final_report_dir: str,
    output_dir: str,
    histories: dict[str, pd.Series],
    history_sources: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Build metadata for regime analysis outputs."""
    return {
        "runner": "src.analysis.regime_analysis_report",
        "final_report_dir": final_report_dir,
        "output_dir": output_dir,
        "history_policy": (
            "TD3 histories are date-averaged across available test fold/seed "
            "policy histories; benchmarks use single deterministic protocol histories."
        ),
        "regimes": [
            {
                "regime_id": regime_id,
                "regime_name": regime_name,
                "start_date": start,
                "end_date": end,
            }
            for regime_id, regime_name, start, end in REGIME_DEFINITIONS
        ],
        "strategies_loaded": sorted(histories),
        "history_sources": history_sources,
        "warnings": warnings,
        "asset_specific_validation": "histories validated when asset-specific mode is requested",
        "reporting_only_note": "No TD3 training is run by this report.",
    }


def build_summary_markdown(
    metrics: pd.DataFrame,
    rankings: pd.DataFrame,
    pairwise: pd.DataFrame,
    winners: pd.DataFrame,
    warnings: list[str],
) -> str:
    """Build concise regime analysis markdown summary."""
    primary = _primary_td3_name(rankings)
    secondary = _secondary_td3_name(rankings)
    v3_wins = _regime_ids_where_strategy_leads(rankings, primary)
    v4_wins = _regime_ids_where_strategy_leads(rankings, secondary)
    benchmark_wins = winners[
        winners["best_by_mandate_style_score"].astype(str).isin(DEFAULT_BENCHMARKS)
    ]
    pair_available = pairwise[pairwise["comparison_available"] == True].copy()
    v3_vs_gld = _pairwise_sentence(pair_available, primary, "BuyHold_GLD")
    v4_vs_gld = _pairwise_sentence(pair_available, secondary, "BuyHold_GLD")
    v3_vs_trend = _pairwise_sentence(pair_available, primary, "trend_spy_cash_12p")
    v4_vs_trend = _pairwise_sentence(pair_available, secondary, "trend_spy_cash_12p")
    return "\n".join(
        [
            "# Regime Analysis Report",
            "",
            "## Method",
            "",
            (
                "TD3 histories are date-averaged across available test fold/seed "
                "policy histories. Benchmarks use the deterministic protocol history files."
            ),
            "",
            "## Main Findings",
            "",
            f"- `{primary}` leads by mandate-style score in: {_format_regime_list(v3_wins)}.",
            f"- `{secondary}` leads by mandate-style score in: {_format_regime_list(v4_wins)}.",
            (
                f"- Benchmarks lead by mandate-style score in {len(benchmark_wins)} "
                "regime/calendar slices."
            ),
            f"- `{primary}` versus BuyHold_GLD: {v3_vs_gld}",
            f"- `{secondary}` versus BuyHold_GLD: {v4_vs_gld}",
            f"- `{primary}` versus trend_spy_cash_12p: {v3_vs_trend}",
            f"- `{secondary}` versus trend_spy_cash_12p: {v4_vs_trend}",
            "",
            "## Interpretation",
            "",
            (
                "The constrained TD3 candidates are regime-sensitive rather than "
                "universally dominant. Their strongest evidence is in the out-of-sample "
                "test windows where histories are available, while early regimes such as "
                "COVID and 2021 mostly lack TD3 test-history coverage under the current "
                "walk-forward outputs."
            ),
            (
                "This supports a cautious claim: constrained TD3 can become competitive "
                "in mandate-aware comparisons, but benchmark rules remain important and "
                "some regimes still favor simpler strategies."
            ),
            "",
            "## Caveats",
            "",
            "- This report does not retrain TD3.",
            "- TD3 regime returns are date-averaged across fold/seed histories.",
            "- Regimes with no TD3 test-history observations are reported as unavailable.",
            *[f"- Warning: {warning}" for warning in warnings],
            "",
        ]
    )


def cap_to_label(cap: Any) -> str:
    """Convert selected cap to directory label used by experiment outputs."""
    if pd.isna(cap):
        return "uncapped"
    return f"{float(cap):.2f}".replace(".", "p")


def _load_selected_and_metadata(final_report_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    asset_selected = final_report_dir / ASSET_SPECIFIC_SELECTED_FILE
    asset_metadata = final_report_dir / ASSET_SPECIFIC_METADATA_FILE
    if asset_selected.exists() and asset_metadata.exists():
        selected = pd.read_csv(asset_selected).copy()
        selected["strategy_name"] = selected["candidate_name"]
        selected["selected_cap"] = selected["max_weight_cap"]
        metadata = json.loads(asset_metadata.read_text(encoding="utf-8"))
        return selected, metadata, "asset_specific"

    metadata_path = final_report_dir / "final_constrained_td3_metadata.json"
    selected_path = final_report_dir / "final_constrained_td3_selected_candidates.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing final report metadata: {metadata_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected candidates table: {selected_path}")
    return (
        pd.read_csv(selected_path),
        json.loads(metadata_path.read_text(encoding="utf-8")),
        "legacy",
    )


def _source_dir_for_candidate(base_candidate: str, metadata: dict[str, Any]) -> str | None:
    if metadata.get("runner") == "src.analysis.asset_specific_cost_final_report":
        source_dirs = metadata.get("source_dirs", {})
        if base_candidate == "V7_real_macro_vintage_clean_no_dxy_garch":
            return source_dirs.get("v7")
        if base_candidate == "V8_ewma_garch_vol_current":
            return source_dirs.get("v8")
        return source_dirs.get("v2_v6")
    source_key = {
        "V3_real_macro_current": "v3_cap_sensitivity_dir",
        "V3_real_macro_vintage": "v3_vintage_cap_sensitivity_dir",
        "V3_real_macro_vintage_clean_no_dxy": "v3_clean_no_dxy_cap_sensitivity_dir",
        "V4_real_garch_current": "v4_cap_sensitivity_dir",
        "V7_real_macro_garch_current": "v7_cap_sensitivity_dir",
        "V7_real_macro_vintage_clean_no_dxy_garch": (
            "v7_clean_no_dxy_garch_cap_sensitivity_dir"
        ),
        "V8_ewma_garch_vol_current": "v8_cap_sensitivity_dir",
    }.get(base_candidate, "cap_sensitivity_dir")
    return metadata.get(source_key)


def _with_cap_sensitivity_overrides(
    metadata: dict[str, Any],
    *,
    v3_cap_sensitivity_dir: str | None = None,
    v3_vintage_cap_sensitivity_dir: str | None = None,
    v3_clean_no_dxy_cap_sensitivity_dir: str | None = None,
    v4_cap_sensitivity_dir: str | None = None,
    v7_cap_sensitivity_dir: str | None = None,
    v7_clean_no_dxy_garch_cap_sensitivity_dir: str | None = None,
    v8_cap_sensitivity_dir: str | None = None,
) -> dict[str, Any]:
    updated = dict(metadata)
    overrides = {
        "v3_cap_sensitivity_dir": v3_cap_sensitivity_dir,
        "v3_vintage_cap_sensitivity_dir": v3_vintage_cap_sensitivity_dir,
        "v3_clean_no_dxy_cap_sensitivity_dir": v3_clean_no_dxy_cap_sensitivity_dir,
        "v4_cap_sensitivity_dir": v4_cap_sensitivity_dir,
        "v7_cap_sensitivity_dir": v7_cap_sensitivity_dir,
        "v7_clean_no_dxy_garch_cap_sensitivity_dir": (
            v7_clean_no_dxy_garch_cap_sensitivity_dir
        ),
        "v8_cap_sensitivity_dir": v8_cap_sensitivity_dir,
    }
    for key, value in overrides.items():
        if value:
            updated[key] = value
    return updated


def _benchmark_history_dir(metadata: dict[str, Any]) -> Path:
    root = Path(metadata["benchmark_comparison_dir"])
    if (root / "histories").exists():
        return root / "histories"
    return root / "benchmarks" / "histories"


def _benchmark_names_for_report(final_report_dir: Path) -> list[str]:
    combined_path = (
        final_report_dir.parent
        / "asset_specific_cost_benchmark_comparison"
        / ASSET_SPECIFIC_COMBINED_RANKING_FILE
    )
    if combined_path.exists():
        ranking = pd.read_csv(combined_path)
        benchmarks = ranking[ranking["strategy_type"].astype(str) == "benchmark"]
        names = benchmarks["strategy_name"].dropna().astype(str).unique().tolist()
        key_names = [
            "trend_spy_cash_12p",
            "BuyHold_GLD",
            "Equal_Weight",
            *DEFAULT_BENCHMARKS,
        ]
        ordered = [name for name in key_names if name in names]
        ordered.extend([name for name in names if name not in ordered])
        return list(dict.fromkeys(ordered))
    return list(DEFAULT_BENCHMARKS)


def _validate_asset_specific_history(frame: pd.DataFrame, path: Path) -> None:
    if "transaction_cost_mode" not in frame.columns:
        raise ValueError(f"Asset-specific validation failed; missing transaction_cost_mode: {path}")
    modes = set(frame["transaction_cost_mode"].dropna().astype(str).unique().tolist())
    if modes != {"asset_specific"}:
        raise ValueError(f"Scalar/non asset-specific history detected in {path}: {sorted(modes)}")
    required = [
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
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Asset-specific history missing diagnostics in {path}: {missing}")


def _pairwise_comparisons_for_histories(
    histories: dict[str, pd.Series],
) -> list[tuple[str, str]]:
    comparisons = list(PAIRWISE_COMPARISONS)
    primary = _primary_td3_name_from_names(histories)
    benchmarks = ["trend_spy_cash_12p", "BuyHold_GLD", "Equal_Weight"]
    if primary:
        comparisons.extend((primary, benchmark) for benchmark in benchmarks)
    for candidate in [
        "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
        "V4_real_garch_current_cap_0p50",
        "V8_ewma_garch_vol_current_cap_0p80",
        "V7_real_macro_vintage_clean_no_dxy_garch_cap_0p50",
    ]:
        if candidate in histories:
            comparisons.extend((candidate, benchmark) for benchmark in benchmarks)
    return list(dict.fromkeys(comparisons))


def _primary_td3_name(frame: pd.DataFrame) -> str:
    names = set(frame["strategy_name"].astype(str))
    preferred = [
        "V5_no_volatility_block_cap_0p50",
        "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
        "V3_cap_0.60",
    ]
    for name in preferred:
        if name in names:
            return name
    td3_like = sorted(name for name in names if name.startswith("V"))
    return td3_like[0] if td3_like else ""


def _secondary_td3_name(frame: pd.DataFrame) -> str:
    names = set(frame["strategy_name"].astype(str))
    preferred = [
        "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
        "V4_real_garch_current_cap_0p50",
        "V4_cap_0.50",
    ]
    for name in preferred:
        if name in names:
            return name
    td3_like = sorted(name for name in names if name.startswith("V"))
    return td3_like[1] if len(td3_like) > 1 else (_primary_td3_name(frame) or "")


def _primary_td3_name_from_names(histories: dict[str, pd.Series]) -> str:
    names = pd.DataFrame({"strategy_name": list(histories)})
    return _primary_td3_name(names)



def _annualized_return(cumulative_return: float, n_obs: int) -> float:
    if n_obs <= 0 or cumulative_return <= -1.0:
        return float("nan")
    return float((1.0 + cumulative_return) ** (PERIODS_PER_YEAR / n_obs) - 1.0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def _mandate_style_score(
    sharpe: float,
    sortino: float,
    calmar: float,
    max_drawdown: float,
) -> float:
    if not np.isfinite(max_drawdown):
        return float("nan")
    bucket = assign_drawdown_bucket(max_drawdown)
    if bucket == "not_eligible":
        return 0.0
    score_values = [
        _bounded_score(sharpe, -1.0, 1.5),
        _bounded_score(sortino, -1.0, 2.5),
        _bounded_score(calmar, -1.0, 2.5),
    ]
    valid_scores = [value for value in score_values if np.isfinite(value)]
    if not valid_scores:
        return float("nan")
    quality = float(np.mean(valid_scores))
    if not np.isfinite(quality):
        return float("nan")
    return float(quality * get_drawdown_multiplier(max_drawdown))


def _simple_composite_score(
    sharpe: float,
    cumulative_return: float,
    max_drawdown: float,
    hit_rate: float,
) -> float:
    values = [
        _bounded_score(sharpe, -1.0, 1.5),
        _bounded_score(cumulative_return, -0.5, 0.8),
        _bounded_score(max_drawdown, -0.5, 0.0),
        hit_rate if np.isfinite(hit_rate) else np.nan,
    ]
    score = np.nanmean(values)
    return float(score) if np.isfinite(score) else float("nan")


def _bounded_score(value: float, low: float, high: float) -> float:
    if not np.isfinite(value) or high <= low:
        return float("nan")
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _empty_metrics() -> dict[str, Any]:
    return {
        "n_observations": 0,
        "cumulative_return": np.nan,
        "annualized_return": np.nan,
        "annualized_volatility": np.nan,
        "sharpe": np.nan,
        "sortino": np.nan,
        "max_drawdown": np.nan,
        "calmar": np.nan,
        "average_return": np.nan,
        "hit_rate": np.nan,
        "worst_period_return": np.nan,
        "mandate_bucket": "no_data",
        "mandate_style_score": np.nan,
        "simple_composite_score": np.nan,
    }


def _delta(left: pd.Series, right: pd.Series, metric: str) -> float:
    left_value = pd.to_numeric(pd.Series([left.get(metric)]), errors="coerce").iloc[0]
    right_value = pd.to_numeric(pd.Series([right.get(metric)]), errors="coerce").iloc[0]
    if pd.isna(left_value) or pd.isna(right_value):
        return float("nan")
    return float(left_value - right_value)


def _beats(left: pd.Series, right: pd.Series, metric: str) -> bool:
    delta = _delta(left, right, metric)
    return bool(np.isfinite(delta) and delta > 0.0)


def _best_strategy(frame: pd.DataFrame, metric: str) -> str:
    valid = frame.dropna(subset=[metric])
    if valid.empty:
        return pd.NA
    return str(valid.sort_values(metric, ascending=False).iloc[0]["strategy_name"])


def _rank_for(frame: pd.DataFrame, strategy: str, column: str) -> Any:
    row = frame.loc[frame["strategy_name"] == strategy]
    if row.empty:
        return pd.NA
    return row.iloc[0][column]


def _value_for(frame: pd.DataFrame, strategy: str, column: str) -> float:
    if strategy not in frame.index:
        return float("nan")
    value = pd.to_numeric(pd.Series([frame.loc[strategy, column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def _regime_ids_where_strategy_leads(rankings: pd.DataFrame, strategy: str) -> list[str]:
    rows = rankings[
        (rankings["strategy_name"] == strategy)
        & (rankings["rank_mandate_style_score"] == 1)
        & (rankings["n_observations"] > 0)
    ]
    return rows["regime_id"].astype(str).tolist()


def _pairwise_sentence(
    pairwise: pd.DataFrame,
    left: str,
    right: str,
) -> str:
    rows = pairwise[
        (pairwise["left_strategy"] == left)
        & (pairwise["right_strategy"] == right)
        & (pairwise["comparison_available"] == True)
    ]
    if rows.empty:
        return "no overlapping regime data available"
    sharpe_wins = int(rows["left_beats_right_by_sharpe"].sum())
    drawdown_wins = int(rows["left_beats_right_by_drawdown"].sum())
    mandate_wins = int(rows["left_beats_right_by_mandate_style"].sum())
    n = len(rows)
    return (
        f"wins {mandate_wins}/{n} by mandate-style score, "
        f"{sharpe_wins}/{n} by Sharpe, and {drawdown_wins}/{n} by max drawdown"
    )


def _format_regime_list(regime_ids: list[str]) -> str:
    if not regime_ids:
        return "none"
    return ", ".join(regime_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regime analysis report.")
    parser.add_argument("--final-report-dir", default=DEFAULT_FINAL_REPORT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v3-cap-sensitivity-dir", default=None)
    parser.add_argument("--v3-vintage-cap-sensitivity-dir", default=None)
    parser.add_argument("--v3-clean-no-dxy-cap-sensitivity-dir", default=None)
    parser.add_argument("--v4-cap-sensitivity-dir", default=None)
    parser.add_argument("--v7-cap-sensitivity-dir", default=None)
    parser.add_argument("--v7-clean-no-dxy-garch-cap-sensitivity-dir", default=None)
    parser.add_argument("--v8-cap-sensitivity-dir", default=None)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--asset-specific-only", action="store_true")
    args = parser.parse_args()
    result = build_regime_analysis_report(
        final_report_dir=args.final_report_dir,
        output_dir=args.output_dir,
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
        asset_specific_only=args.asset_specific_only or None,
    )
    print(f"Histories found: {len(result['histories'])}")
    print("\nRegime winners:")
    print(result["winners"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
