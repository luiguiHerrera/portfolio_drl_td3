"""Statistical validation report for final constrained TD3 results.

This module is reporting-only. It reads existing strategy histories, builds
date-averaged TD3 return series when multiple fold/seed histories are present,
and computes block-bootstrap confidence intervals and paired bootstrap deltas.
It does not retrain models or alter production scoring logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SELECTED_FILE = "final_constrained_td3_selected_candidates.csv"
METADATA_FILE = "final_constrained_td3_metadata.json"
ASSET_SPECIFIC_SELECTED_FILE = "asset_specific_cost_selected_candidates.csv"
ASSET_SPECIFIC_METADATA_FILE = "asset_specific_cost_metadata.json"
ASSET_SPECIFIC_COMBINED_RANKING_FILE = "asset_specific_cost_combined_ranking.csv"
FINAL_CORRECTED_COMBINED_PATTERN = "final_corrected_*_combined_ranking.csv"
FINAL_CORRECTED_METADATA_PATTERN = "final_corrected_*_benchmark_comparison_metadata.json"
BENCHMARK_HISTORY_DIR = "benchmarks/histories"

DEFAULT_FINAL_REPORT_DIR = "outputs/tables/final_constrained_td3_report_with_v3_v4_60ep_10seeds"
DEFAULT_OUTPUT_DIR = "outputs/tables/statistical_validation_final_v3_v4"

PRIMARY_CANDIDATES = [
    "V5_no_volatility_block_cap_0p50",
    "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
    "V4_real_garch_current_cap_0p50",
    "V8_ewma_garch_vol_current_cap_0p80",
    "V3_real_macro_vintage_clean_no_dxy_cap_0.50",
    "V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50",
    "V3_real_macro_vintage_cap_0.50",
    "V3_cap_0.60",
    "V4_cap_0.50",
]
DEFAULT_BENCHMARKS = [
    "trend_spy_cash_12p",
    "BuyHold_GLD",
    "Equal_Weight",
]
WEEKLY_PERIODS_PER_YEAR = 52


def build_statistical_validation_report(
    final_report_dir: str = DEFAULT_FINAL_REPORT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    n_bootstrap: int = 1000,
    block_size: int = 12,
    random_seed: int = 123,
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
    """Build statistical validation CSVs and markdown from existing histories."""
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

    metric_ci = build_metric_ci_table(
        histories,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        random_seed=random_seed,
    )
    pairwise = build_pairwise_bootstrap_table(
        histories,
        selected_candidates=selected,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        random_seed=random_seed,
    )
    summary = build_summary_table(metric_ci, pairwise, warnings)
    markdown = build_summary_markdown(metric_ci, pairwise, warnings)

    paths = {
        "metric_ci": out_dir / "statistical_validation_metric_ci.csv",
        "pairwise_bootstrap": out_dir / "statistical_validation_pairwise_bootstrap.csv",
        "summary": out_dir / "statistical_validation_summary.csv",
        "markdown": out_dir / "statistical_validation_summary.md",
    }
    metric_ci.to_csv(paths["metric_ci"], index=False)
    pairwise.to_csv(paths["pairwise_bootstrap"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["markdown"].write_text(markdown, encoding="utf-8")

    return {
        "histories": histories,
        "history_records": history_records,
        "warnings": warnings,
        "report_mode": report_mode,
        "metric_ci": metric_ci,
        "pairwise_bootstrap": pairwise,
        "summary": summary,
        "markdown": markdown,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def locate_strategy_histories(
    final_report_dir: Path,
    selected_candidates: pd.DataFrame,
    metadata: dict[str, Any],
    require_asset_specific: bool = False,
) -> tuple[dict[str, pd.Series], pd.DataFrame, list[str]]:
    """Locate selected TD3 and benchmark histories."""
    histories: dict[str, pd.Series] = {}
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for _, row in selected_candidates.iterrows():
        strategy_name = str(row["strategy_name"])
        base_candidate = str(row["base_candidate"])
        source_dir = _source_dir_for_row(row, metadata)
        cap_label = _row_cap_label(row)
        if source_dir is None:
            warnings.append(f"Missing source directory for {strategy_name}.")
            records.append(_history_record(strategy_name, "td3", False, "", 0, 0))
            continue
        per_candidate = source_dir / "per_candidate" / base_candidate
        pattern = f"*_{base_candidate}_cap_{cap_label}_seed_*/test_policy_history.csv"
        paths = sorted(per_candidate.glob(pattern))
        if not paths:
            warnings.append(f"No TD3 test policy histories found for {strategy_name}.")
            records.append(_history_record(strategy_name, "td3", False, str(per_candidate), 0, 0))
            continue
        series = load_date_averaged_return_series(
            paths,
            require_asset_specific=require_asset_specific,
        )
        histories[strategy_name] = series
        records.append(
            _history_record(
                strategy_name,
                "td3",
                True,
                str(per_candidate),
                len(paths),
                len(series),
            )
        )

    benchmark_dir = _benchmark_history_dir(final_report_dir, metadata)
    for benchmark_name in _benchmark_names_for_report(final_report_dir):
        path = benchmark_dir / f"{benchmark_name}_history.csv"
        if not path.exists():
            warnings.append(f"No benchmark history found for {benchmark_name}.")
            records.append(_history_record(benchmark_name, "benchmark", False, str(path), 0, 0))
            continue
        series = load_return_series(path, require_asset_specific=require_asset_specific)
        histories[benchmark_name] = series
        records.append(
            _history_record(
                benchmark_name,
                "benchmark",
                True,
                str(path),
                1,
                len(series),
            )
        )

    return histories, pd.DataFrame(records), warnings


def load_selected_candidates_and_metadata(final_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Load either legacy final report metadata or asset-specific official metadata."""
    corrected_files = sorted(final_dir.glob(FINAL_CORRECTED_COMBINED_PATTERN))
    if corrected_files:
        ranking = pd.read_csv(corrected_files[0])
        selected = ranking.loc[
            ranking["strategy_type"].astype(str).str.upper() == "TD3"
        ].copy()
        if selected.empty:
            raise ValueError(f"No selected TD3 rows found in {corrected_files[0]}")
        selected["selected_cap"] = selected["cap_label"]
        selected["source"] = "final_corrected_td3_benchmark_comparison"
        metadata_files = sorted(final_dir.glob(FINAL_CORRECTED_METADATA_PATTERN))
        metadata = (
            json.loads(metadata_files[0].read_text(encoding="utf-8"))
            if metadata_files
            else {}
        )
        metadata["runner"] = "src.analysis.final_corrected_td3_benchmark_comparison_report"
        return selected, metadata, "asset_specific"

    asset_selected = final_dir / ASSET_SPECIFIC_SELECTED_FILE
    asset_metadata = final_dir / ASSET_SPECIFIC_METADATA_FILE
    if asset_selected.exists() and asset_metadata.exists():
        selected = pd.read_csv(asset_selected)
        selected = selected.copy()
        selected["strategy_name"] = selected["candidate_name"]
        selected["selected_cap"] = selected["max_weight_cap"]
        selected["source"] = "asset_specific_cost_full_report"
        metadata = json.loads(asset_metadata.read_text(encoding="utf-8"))
        return selected, metadata, "asset_specific"

    selected = pd.read_csv(final_dir / SELECTED_FILE)
    metadata = json.loads((final_dir / METADATA_FILE).read_text(encoding="utf-8"))
    return selected, metadata, "legacy"


def load_date_averaged_return_series(
    paths: list[Path],
    require_asset_specific: bool = False,
) -> pd.Series:
    """Load multiple histories and average duplicate dates."""
    frames = []
    for path in paths:
        series = load_return_series(path, require_asset_specific=require_asset_specific)
        frames.append(series.rename("return").reset_index().rename(columns={"index": "date"}))
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    averaged = combined.groupby("date", sort=True)["return"].mean()
    averaged.name = "return"
    return averaged


def load_return_series(path: Path, require_asset_specific: bool = False) -> pd.Series:
    """Load financial net return, falling back to portfolio return."""
    frame = pd.read_csv(path)
    if require_asset_specific:
        _validate_asset_specific_history(frame, path)
    date_column = "date" if "date" in frame.columns else frame.columns[0]
    if "financial_net_return" in frame.columns:
        return_column = "financial_net_return"
    elif "portfolio_return" in frame.columns:
        return_column = "portfolio_return"
    else:
        raise ValueError(f"No return column found in history: {path}")
    dates = pd.to_datetime(frame[date_column])
    returns = pd.to_numeric(frame[return_column], errors="coerce")
    series = pd.Series(returns.to_numpy(dtype=float), index=dates, name="return")
    series = series.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if series.empty:
        raise ValueError(f"History contains no valid returns: {path}")
    return series


def build_metric_ci_table(
    histories: dict[str, pd.Series],
    n_bootstrap: int = 1000,
    block_size: int = 12,
    random_seed: int = 123,
) -> pd.DataFrame:
    """Compute block-bootstrap metric confidence intervals per strategy."""
    rows = []
    for offset, (strategy_name, returns) in enumerate(sorted(histories.items())):
        estimates = compute_return_metrics(returns)
        samples = block_bootstrap_metrics(
            returns,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            random_seed=random_seed + offset,
        )
        for metric, estimate in estimates.items():
            values = np.array([sample[metric] for sample in samples], dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "strategy_name": strategy_name,
                    "metric": metric,
                    "estimate": estimate,
                    "lower_5pct": float(np.nanpercentile(values, 5)) if len(values) else np.nan,
                    "upper_95pct": float(np.nanpercentile(values, 95)) if len(values) else np.nan,
                    "n_periods": len(returns),
                    "n_bootstrap": n_bootstrap,
                    "block_size": block_size,
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_bootstrap_table(
    histories: dict[str, pd.Series],
    selected_candidates: pd.DataFrame,
    n_bootstrap: int = 1000,
    block_size: int = 12,
    random_seed: int = 123,
) -> pd.DataFrame:
    """Compute paired bootstrap deltas for key TD3 candidates versus benchmarks."""
    selected_names = [
        str(name)
        for name in selected_candidates["strategy_name"].dropna().astype(str)
        if str(name) in histories
    ]
    candidates = list(dict.fromkeys(
        [name for name in PRIMARY_CANDIDATES if name in histories] + selected_names
    ))
    benchmark_names = _pairwise_benchmarks(selected_candidates, histories)
    rows = []
    seed_offset = 0
    for candidate_name in candidates:
        for benchmark_name in benchmark_names:
            if candidate_name == benchmark_name:
                continue
            if benchmark_name not in histories:
                continue
            aligned = align_return_pair(histories[candidate_name], histories[benchmark_name])
            if aligned.empty:
                continue
            samples = paired_block_bootstrap_deltas(
                aligned["candidate"],
                aligned["benchmark"],
                n_bootstrap=n_bootstrap,
                block_size=block_size,
                random_seed=random_seed + seed_offset,
            )
            seed_offset += 1
            candidate_metrics = compute_return_metrics(aligned["candidate"])
            benchmark_metrics = compute_return_metrics(aligned["benchmark"])
            for metric in candidate_metrics:
                deltas = np.array([sample[metric] for sample in samples], dtype=float)
                deltas = deltas[np.isfinite(deltas)]
                rows.append(
                    {
                        "candidate": candidate_name,
                        "benchmark": benchmark_name,
                        "metric": metric,
                        "candidate_estimate": candidate_metrics[metric],
                        "benchmark_estimate": benchmark_metrics[metric],
                        "mean_delta": float(np.nanmean(deltas)) if len(deltas) else np.nan,
                        "lower_5pct_delta": (
                            float(np.nanpercentile(deltas, 5)) if len(deltas) else np.nan
                        ),
                        "upper_95pct_delta": (
                            float(np.nanpercentile(deltas, 95)) if len(deltas) else np.nan
                        ),
                        "probability_candidate_beats": (
                            _probability_beats(deltas, metric) if len(deltas) else np.nan
                        ),
                        "n_aligned_periods": len(aligned),
                        "n_bootstrap": n_bootstrap,
                        "block_size": block_size,
                    }
                )
    columns = [
        "candidate",
        "benchmark",
        "metric",
        "candidate_estimate",
        "benchmark_estimate",
        "mean_delta",
        "lower_5pct_delta",
        "upper_95pct_delta",
        "probability_candidate_beats",
        "n_aligned_periods",
        "n_bootstrap",
        "block_size",
    ]
    return pd.DataFrame(rows, columns=columns)


def compute_return_metrics(returns: pd.Series) -> dict[str, float]:
    """Compute weekly-return performance metrics."""
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "cumulative_return": np.nan,
        }
    cumulative = float((1.0 + clean).prod() - 1.0)
    annualized_return = float((1.0 + cumulative) ** (WEEKLY_PERIODS_PER_YEAR / len(clean)) - 1.0)
    weekly_vol = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    annualized_vol = weekly_vol * np.sqrt(WEEKLY_PERIODS_PER_YEAR) if np.isfinite(weekly_vol) else np.nan
    sharpe = annualized_return / annualized_vol if annualized_vol and annualized_vol > 0 else np.nan
    max_drawdown = calculate_max_drawdown(clean)
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": float(annualized_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "cumulative_return": cumulative,
    }


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate max drawdown from periodic returns."""
    equity = (1.0 + pd.to_numeric(returns, errors="coerce").dropna()).cumprod()
    if equity.empty:
        return np.nan
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def block_bootstrap_metrics(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int = 12,
    random_seed: int = 123,
) -> list[dict[str, float]]:
    """Block bootstrap metric estimates."""
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    samples = []
    for _ in range(n_bootstrap):
        boot = _sample_blocks(values, block_size=block_size, rng=rng)
        samples.append(compute_return_metrics(pd.Series(boot)))
    return samples


def paired_block_bootstrap_deltas(
    candidate_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int = 12,
    random_seed: int = 123,
) -> list[dict[str, float]]:
    """Paired block bootstrap metric deltas candidate minus benchmark."""
    aligned = align_return_pair(candidate_returns, benchmark_returns)
    values = aligned[["candidate", "benchmark"]].to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    deltas = []
    for _ in range(n_bootstrap):
        boot = _sample_blocks(values, block_size=block_size, rng=rng)
        candidate_metrics = compute_return_metrics(pd.Series(boot[:, 0]))
        benchmark_metrics = compute_return_metrics(pd.Series(boot[:, 1]))
        deltas.append(
            {
                metric: candidate_metrics[metric] - benchmark_metrics[metric]
                for metric in candidate_metrics
            }
        )
    return deltas


def align_return_pair(candidate: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Align two return series on common dates."""
    frame = pd.concat(
        [
            candidate.rename("candidate"),
            benchmark.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    return frame.sort_index()


def build_summary_table(
    metric_ci: pd.DataFrame,
    pairwise: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    """Build compact interpretation summary."""
    rows = []
    pairwise_candidates = list(pairwise["candidate"].dropna().astype(str).unique())
    candidates = pairwise_candidates or list(PRIMARY_CANDIDATES)
    for candidate in candidates:
        candidate_pairs = pairwise[pairwise["candidate"] == candidate]
        if candidate_pairs.empty:
            rows.append(
                {
                    "candidate": candidate,
                    "available": False,
                    "buyhold_gld_probability_sharpe_beats": np.nan,
                    "trend_probability_sharpe_beats": np.nan,
                    "interpretation": "missing_history",
                    "warnings": "; ".join(warnings),
                }
            )
            continue
        gld_prob = _pair_probability(candidate_pairs, "BuyHold_GLD", "sharpe")
        trend_prob = _pair_probability(candidate_pairs, "trend_spy_cash_12p", "sharpe")
        interpretation = interpret_pairwise_evidence(gld_prob, trend_prob)
        rows.append(
            {
                "candidate": candidate,
                "available": True,
                "buyhold_gld_probability_sharpe_beats": gld_prob,
                "trend_probability_sharpe_beats": trend_prob,
                "interpretation": interpretation,
                "warnings": "; ".join(warnings),
            }
        )
    rows.append(
        {
            "candidate": "DSR",
            "available": False,
            "buyhold_gld_probability_sharpe_beats": np.nan,
            "trend_probability_sharpe_beats": np.nan,
            "interpretation": (
                "Deflated Sharpe was not recomputed here because this layer uses "
                "date-averaged histories and does not reconstruct the full multiple-"
                "testing universe. Existing robust_score DSR fields remain the DSR source."
            ),
            "warnings": "; ".join(warnings),
        }
    )
    return pd.DataFrame(rows)


def interpret_pairwise_evidence(gld_prob: float, trend_prob: float) -> str:
    """Interpret probabilities conservatively."""
    probs = [p for p in [gld_prob, trend_prob] if pd.notna(p)]
    if not probs:
        return "missing_pairwise_evidence"
    if all(p >= 0.95 for p in probs):
        return "statistically_clear_vs_clean_benchmarks"
    if all(p >= 0.60 for p in probs):
        return "directionally_positive_but_uncertain"
    return "not_statistically_clear"


def build_summary_markdown(
    metric_ci: pd.DataFrame,
    pairwise: pd.DataFrame,
    warnings: list[str],
) -> str:
    """Build Markdown summary."""
    lines = [
        "# Statistical Validation Report",
        "",
        "This report uses existing per-period histories only. TD3 fold/seed histories "
        "are date-averaged before bootstrap so overlapping dates are not treated as "
        "independent observations.",
        "",
        "## Main Pairwise Result",
        "",
    ]
    pairwise_candidates = list(pairwise["candidate"].dropna().astype(str).unique())
    candidates = pairwise_candidates or list(PRIMARY_CANDIDATES)
    for candidate in candidates:
        candidate_pairs = pairwise[
            (pairwise["candidate"] == candidate)
            & (pairwise["metric"] == "sharpe")
            & (pairwise["benchmark"].isin(DEFAULT_BENCHMARKS))
        ]
        if candidate_pairs.empty:
            lines.append(f"- `{candidate}`: missing usable pairwise histories.")
            continue
        for _, row in candidate_pairs.iterrows():
            lines.append(
                f"- `{candidate}` vs `{row['benchmark']}` Sharpe delta "
                f"{_fmt(row['mean_delta'])} "
                f"[{_fmt(row['lower_5pct_delta'])}, {_fmt(row['upper_95pct_delta'])}], "
                f"P(candidate beats)={_fmt(row['probability_candidate_beats'])}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _overall_interpretation(pairwise),
            "",
            "## Caveats",
            "",
            "- Bootstrap confidence intervals are approximate and depend on block length.",
            "- TD3 histories are averaged by date across runs; this is conservative relative to pooled duplicate-date returns.",
            "- Deflated Sharpe is not recomputed in this layer; existing robust_score DSR fields remain the DSR source.",
        ]
    )
    if warnings:
        lines.append("- Warnings: " + "; ".join(warnings))
    return "\n".join(lines) + "\n"


def _overall_interpretation(pairwise: pd.DataFrame) -> str:
    interpretations = []
    pairwise_candidates = list(pairwise["candidate"].dropna().astype(str).unique())
    candidates = pairwise_candidates or list(PRIMARY_CANDIDATES)
    for candidate in candidates:
        pairs = pairwise[
            (pairwise["candidate"] == candidate)
            & (pairwise["metric"] == "sharpe")
            & (pairwise["benchmark"].isin(DEFAULT_BENCHMARKS))
        ]
        probs = list(pd.to_numeric(pairs["probability_candidate_beats"], errors="coerce").dropna())
        if len(probs) < 2:
            interpretations.append(f"`{candidate}` has incomplete benchmark evidence.")
        elif all(prob >= 0.95 for prob in probs):
            interpretations.append(f"`{candidate}` is statistically clear versus BuyHold_GLD and trend_spy_cash by Sharpe bootstrap.")
        elif all(prob >= 0.60 for prob in probs):
            interpretations.append(f"`{candidate}` is directionally positive but statistically uncertain versus clean benchmarks.")
        else:
            interpretations.append(f"`{candidate}` is not statistically clear versus clean benchmarks.")
    return " ".join(interpretations)


def _sample_blocks(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) == 0:
        return values
    block = max(1, min(int(block_size), len(values)))
    pieces = []
    total = 0
    while total < len(values):
        start = int(rng.integers(0, len(values) - block + 1))
        piece = values[start : start + block]
        pieces.append(piece)
        total += len(piece)
    return np.concatenate(pieces, axis=0)[: len(values)]


def _source_dir_for_row(row: pd.Series, metadata: dict[str, Any]) -> Path | None:
    if metadata.get("runner") == "src.analysis.final_corrected_td3_benchmark_comparison_report":
        path = metadata.get("td3_history_dir") or metadata.get("td3_dir")
        return Path(path) if path else None
    if metadata.get("runner") == "src.analysis.asset_specific_cost_final_report":
        source_dirs = metadata.get("source_dirs", {})
        base_candidate = str(row.get("base_candidate", ""))
        if base_candidate == "V7_real_macro_vintage_clean_no_dxy_garch":
            path = source_dirs.get("v7")
        elif base_candidate == "V8_ewma_garch_vol_current":
            path = source_dirs.get("v8")
        else:
            path = source_dirs.get("v2_v6")
        return Path(path) if path else None
    source = str(row.get("source"))
    if source == "seeded_cap_sensitivity":
        path = metadata.get("v3_cap_sensitivity_dir")
    elif source == "v3_vintage_cap_sensitivity":
        path = metadata.get("v3_vintage_cap_sensitivity_dir")
    elif source == "v3_clean_no_dxy_cap_sensitivity":
        path = metadata.get("v3_clean_no_dxy_cap_sensitivity_dir")
    elif source == "v4_cap_sensitivity":
        path = metadata.get("v4_cap_sensitivity_dir")
    elif source == "v7_cap_sensitivity":
        path = metadata.get("v7_cap_sensitivity_dir")
    elif source == "v7_clean_no_dxy_garch_cap_sensitivity":
        path = metadata.get("v7_clean_no_dxy_garch_cap_sensitivity_dir")
    elif source == "v8_cap_sensitivity":
        path = metadata.get("v8_cap_sensitivity_dir")
    else:
        path = metadata.get("cap_sensitivity_dir")
    return Path(path) if path else None


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


def _benchmark_history_dir(final_report_dir: Path, metadata: dict[str, Any]) -> Path:
    benchmark_root = Path(metadata["benchmark_comparison_dir"])
    benchmark_dir = (
        benchmark_root / BENCHMARK_HISTORY_DIR
        if not (benchmark_root / "histories").exists()
        else benchmark_root / "histories"
    )
    if benchmark_dir.exists():
        return benchmark_dir
    return final_report_dir.parent / "capped_td3_protocol_comparison_60ep_10seeds_cap060" / BENCHMARK_HISTORY_DIR


def _benchmark_names_for_report(final_report_dir: Path) -> list[str]:
    corrected_files = sorted(final_report_dir.glob(FINAL_CORRECTED_COMBINED_PATTERN))
    if corrected_files:
        ranking = pd.read_csv(corrected_files[0])
        benchmarks = ranking[ranking["strategy_type"].astype(str) == "benchmark"]
        names = list(benchmarks["strategy_name"].dropna().astype(str).unique())
        key_names = ["trend_spy_cash_12p", "BuyHold_GLD", "Equal_Weight"]
        ordered = [name for name in key_names if name in names]
        ordered.extend([name for name in names if name not in ordered])
        return ordered
    combined_path = final_report_dir.parent / "asset_specific_cost_benchmark_comparison" / ASSET_SPECIFIC_COMBINED_RANKING_FILE
    if combined_path.exists():
        ranking = pd.read_csv(combined_path)
        benchmarks = ranking[ranking["strategy_type"].astype(str) == "benchmark"]
        names = list(benchmarks["strategy_name"].dropna().astype(str).unique())
        key_names = ["trend_spy_cash_12p", "BuyHold_GLD", "Equal_Weight"]
        ordered = [name for name in key_names if name in names]
        ordered.extend([name for name in names if name not in ordered])
        return ordered
    ranking_path = final_report_dir / "final_constrained_td3_mandate_ranking.csv"
    if not ranking_path.exists():
        return list(DEFAULT_BENCHMARKS)
    ranking = pd.read_csv(ranking_path)
    benchmarks = ranking[ranking["strategy_type"].astype(str) == "benchmark"]
    names = list(benchmarks["strategy_name"].dropna().astype(str).unique())
    key_names = [
        "BuyHold_GLD",
        "trend_spy_cash_12p",
        "rolling_markowitz_min_variance_52p",
        "defensive_risk_off_12p",
        "rolling_risk_parity_inverse_vol_12p",
        "60_40_SPY_TLT",
    ]
    ordered = [name for name in key_names if name in names]
    ordered.extend([name for name in names if name not in ordered])
    return ordered


def _pairwise_benchmarks(
    selected_candidates: pd.DataFrame,
    histories: dict[str, pd.Series],
) -> list[str]:
    benchmarks = [name for name in DEFAULT_BENCHMARKS if name in histories]
    if "BuyHold_GLD" not in benchmarks:
        benchmark_candidates = [name for name in histories if name.startswith("BuyHold")]
        benchmarks.extend(benchmark_candidates[:1])
    return list(dict.fromkeys(benchmarks))


def _cap_label(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "uncapped"
    return f"{float(numeric):.2f}".replace(".", "p")


def _row_cap_label(row: pd.Series) -> str:
    candidate_name = str(row.get("strategy_name", ""))
    if "_cap_" in candidate_name and "0p" in candidate_name.rsplit("_cap_", 1)[-1]:
        return candidate_name.rsplit("_cap_", 1)[-1]
    return _cap_label(row.get("selected_cap"))


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


def _history_record(
    strategy_name: str,
    strategy_type: str,
    found: bool,
    source: str,
    n_files: int,
    n_periods: int,
) -> dict[str, Any]:
    return {
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "history_found": found,
        "source": source,
        "n_history_files": n_files,
        "n_periods": n_periods,
    }


def _pair_probability(pairwise: pd.DataFrame, benchmark: str, metric: str) -> float:
    rows = pairwise[
        (pairwise["benchmark"] == benchmark)
        & (pairwise["metric"] == metric)
    ]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0]["probability_candidate_beats"])


def _probability_beats(deltas: np.ndarray, metric: str) -> float:
    if metric == "annualized_volatility":
        return float(np.mean(deltas < 0.0))
    return float(np.mean(deltas > 0.0))


def _fmt(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build statistical validation report from final constrained TD3 outputs.",
    )
    parser.add_argument("--final-report-dir", default=DEFAULT_FINAL_REPORT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=12)
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

    report = build_statistical_validation_report(
        final_report_dir=args.final_report_dir,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        block_size=args.block_size,
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
    print("Histories found:")
    print(report["history_records"].to_string(index=False))
    print("\nPairwise bootstrap:")
    print(report["pairwise_bootstrap"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
