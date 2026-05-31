"""Build a combined report for asset-specific transaction cost retraining.

This reporting layer combines already-generated limited cap-sensitivity outputs.
It does not retrain models or alter scoring logic.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import ParserError
import yaml

from src.analysis.audit_reward_incentives import compute_reward_incentive_flags
from src.analysis.mandate_aware_score import (
    assign_drawdown_bucket,
    calculate_recovery_required,
    get_drawdown_multiplier,
)
from src.analysis.robust_score import (
    compute_composite_robust_score,
    compute_deflated_sharpe_ratio,
    compute_probabilistic_sharpe_ratio,
    normalize_metric_series,
)
from src.backtest.evaluate_policy import (
    annualized_return,
    annualized_volatility,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
)
from src.backtest.performance_metrics import calmar_ratio, sortino_ratio


DEFAULT_V3_DIR = "outputs/tables/asset_specific_cost_v3_clean_no_dxy_60ep_10seeds"
DEFAULT_V7_DIR = "outputs/tables/asset_specific_cost_v7_clean_no_dxy_garch_60ep_10seeds"
DEFAULT_V4_DIR = "outputs/tables/asset_specific_cost_v4_garch_60ep_10seeds"
DEFAULT_OUTPUT_DIR = "outputs/tables/asset_specific_cost_limited_final_report"
DEFAULT_FULL_OUTPUT_DIR = "outputs/tables/asset_specific_cost_full_final_report"
DEFAULT_V2_V6_DIR = "outputs/tables/asset_specific_cost_full_final_candidates_60ep_10seeds"
DEFAULT_FULL_V7_DIR = "outputs/tables/asset_specific_cost_v7_full_grid_60ep_10seeds"
DEFAULT_FULL_V8_DIR = "outputs/tables/asset_specific_cost_v8_full_grid_60ep_10seeds"

CAP_RESULTS_FILE = "cap_sensitivity_all_results.csv"
CAP_METADATA_FILE = "cap_sensitivity_metadata.json"
PERIODS_PER_YEAR = 52
EXPECTED_SEEDS = [7, 21, 42, 84, 101, 123, 202, 303, 404, 505]
EXPECTED_FOLDS = ["F1", "F2", "F3", "F4"]
EXPECTED_HISTORIES_PER_CANDIDATE_CAP = 40
EXPECTED_CAP_LABELS = ["uncapped", "0p50", "0p60", "0p70", "0p80"]
EXPECTED_CAP_VALUES = {
    "uncapped": np.nan,
    "0p50": 0.50,
    "0p60": 0.60,
    "0p70": 0.70,
    "0p80": 0.80,
}
EXPECTED_FULL_CANDIDATES = [
    "V2_reference_full",
    "V3_real_macro_vintage_clean_no_dxy",
    "V4_real_garch_current",
    "V5_no_volatility_block",
    "V6_financial_state",
    "V7_real_macro_vintage_clean_no_dxy_garch",
    "V8_ewma_garch_vol_current",
]
EXPECTED_COST_MODEL = {
    "transaction_cost_mode": "asset_specific",
    "asset_transaction_cost_bps": {
        "SPY": 2.0,
        "TLT": 2.0,
        "GLD": 2.0,
        "BTC-USD": 10.0,
        "CASH": 0.0,
    },
}
REQUIRED_HISTORY_COLUMNS = [
    "transaction_cost_mode",
    "transaction_cost",
    "turnover",
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
    "weight_SPY",
    "weight_TLT",
    "weight_GLD",
    "weight_BTC-USD",
    "weight_CASH",
]
REQUIRED_RUN_FILES = [
    "test_policy_history.csv",
    "test_metrics_table.csv",
    "test_diagnostics.csv",
]
RUN_DIR_PATTERN = re.compile(
    r"^(?P<fold>F\d+)_(?P<candidate>.+)_cap_(?P<cap>uncapped|\d+p\d+)_seed_(?P<seed>\d+)$",
)

REPORT_COLUMNS = [
    "rank_mandate_aware",
    "rank_robust",
    "strategy_name",
    "base_candidate",
    "cap_label",
    "max_weight_cap",
    "transaction_cost_mode",
    "asset_transaction_cost_bps",
    "mean_transaction_cost",
    "average_turnover",
    "average_btc_cost_contribution",
    "average_btc_allocation",
    "average_max_weight",
    "average_effective_number_of_assets",
    "max_drawdown",
    "worst_max_drawdown",
    "sharpe",
    "annualized_return",
    "annualized_volatility",
    "robust_score",
    "mandate_aware_score",
    "decision_label",
    "candidate_output_dir",
    "score_comparability_note",
]

SCORE_COMPARABILITY_NOTE = (
    "Scores are imported from limited cap-sensitivity reports. Because robust "
    "and mandate-aware components may be normalized within each report universe, "
    "cross-candidate score comparisons should be treated as preliminary."
)


def build_asset_specific_cost_final_report(
    v3_dir: str = DEFAULT_V3_DIR,
    v7_dir: str = DEFAULT_V7_DIR,
    v4_dir: str = DEFAULT_V4_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Combine limited asset-specific retraining outputs into one report."""
    input_dirs = {
        "v3_clean_no_dxy": Path(v3_dir),
        "v7_clean_no_dxy_garch": Path(v7_dir),
        "v4_garch": Path(v4_dir),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata_inputs = {}
    warnings = []
    for label, directory in input_dirs.items():
        if not directory.exists():
            warnings.append(f"Missing input directory: {directory}")
            continue
        frame = load_cap_results(directory)
        frame["input_label"] = label
        metadata_inputs[label] = load_metadata(directory)
        rows.append(frame)

    if not rows:
        raise ValueError("No asset-specific cap-sensitivity inputs were found.")

    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = enrich_with_history_diagnostics(combined)
    combined["score_comparability_note"] = SCORE_COMPARABILITY_NOTE
    main_ranking = build_main_ranking(combined)
    selected = build_selected_candidates(main_ranking)
    markdown = build_summary_markdown(main_ranking, selected, warnings)
    metadata = build_metadata(
        input_dirs=input_dirs,
        metadata_inputs=metadata_inputs,
        output_dir=output_path,
        warnings=warnings,
    )

    paths = {
        "selected_candidates": output_path / "asset_specific_cost_selected_candidates.csv",
        "main_ranking": output_path / "asset_specific_cost_main_ranking.csv",
        "markdown_summary": output_path / "asset_specific_cost_summary.md",
        "metadata": output_path / "asset_specific_cost_metadata.json",
    }
    selected.to_csv(paths["selected_candidates"], index=False)
    main_ranking.to_csv(paths["main_ranking"], index=False)
    paths["markdown_summary"].write_text(markdown, encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "selected_candidates": selected,
        "main_ranking": main_ranking,
        "markdown_summary": markdown,
        "metadata": metadata,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def build_official_asset_specific_cost_full_report(
    v2_v6_dir: str = DEFAULT_V2_V6_DIR,
    v7_dir: str = DEFAULT_FULL_V7_DIR,
    v8_dir: str = DEFAULT_FULL_V8_DIR,
    output_dir: str = DEFAULT_FULL_OUTPUT_DIR,
) -> dict[str, Any]:
    """Reconstruct official full-universe asset-specific-cost report from histories."""
    sources = {
        "v2_v6": Path(v2_v6_dir),
        "v7": Path(v7_dir),
        "v8": Path(v8_dir),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    read_log: list[dict[str, Any]] = []
    ignored_dirs: list[str] = []
    run_records = collect_run_records(sources, read_log=read_log, ignored_dirs=ignored_dirs)
    validation = validate_full_coverage(run_records)
    run_metrics, observations = build_run_metrics_and_observations(
        run_records,
        read_log=read_log,
    )
    all_caps = build_all_candidate_caps(run_metrics, observations)
    selected = build_official_selected_candidates(all_caps)
    main_ranking = build_official_main_ranking(selected)
    best_by_metric = build_best_by_metric(all_caps)
    metadata = build_official_metadata(
        sources=sources,
        output_dir=output_path,
        run_records=run_records,
        validation=validation,
        read_log=read_log,
        ignored_dirs=ignored_dirs,
    )
    markdown = build_official_summary_markdown(
        all_caps=all_caps,
        selected=selected,
        main_ranking=main_ranking,
        best_by_metric=best_by_metric,
        metadata=metadata,
    )

    paths = {
        "all_candidate_caps": output_path / "asset_specific_cost_all_candidate_caps.csv",
        "selected_candidates": output_path / "asset_specific_cost_selected_candidates.csv",
        "main_ranking": output_path / "asset_specific_cost_main_ranking.csv",
        "best_by_metric": output_path / "asset_specific_cost_best_by_metric.csv",
        "metadata": output_path / "asset_specific_cost_metadata.json",
        "markdown_summary": output_path / "asset_specific_cost_summary.md",
    }
    all_caps.to_csv(paths["all_candidate_caps"], index=False)
    selected.to_csv(paths["selected_candidates"], index=False)
    main_ranking.to_csv(paths["main_ranking"], index=False)
    best_by_metric.to_csv(paths["best_by_metric"], index=False)
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["markdown_summary"].write_text(markdown, encoding="utf-8")

    return {
        "all_candidate_caps": all_caps,
        "selected_candidates": selected,
        "main_ranking": main_ranking,
        "best_by_metric": best_by_metric,
        "metadata": metadata,
        "markdown_summary": markdown,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def read_csv_with_retry(
    path: str | Path,
    retries: int = 3,
    sleep_seconds: float = 1.0,
    read_log: list[dict[str, Any]] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read CSV with bounded retries for transient I/O failures."""
    csv_path = Path(path)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            frame = pd.read_csv(csv_path, **kwargs)
            if read_log is not None:
                read_log.append(
                    {
                        "path": str(csv_path),
                        "attempts": attempt,
                        "status": "ok",
                    },
                )
            return frame
        except (TimeoutError, OSError, ParserError, pd.errors.EmptyDataError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds)
    if read_log is not None:
        read_log.append(
            {
                "path": str(csv_path),
                "attempts": retries,
                "status": "failed",
                "error": repr(last_error),
            },
        )
    raise RuntimeError(f"Failed to read CSV after {retries} attempts: {csv_path}") from last_error


def parse_run_folder_name(name: str) -> dict[str, Any]:
    """Parse fold, candidate, cap, and seed from a saved TD3 run folder."""
    match = RUN_DIR_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unrecognized run folder name: {name}")
    result = match.groupdict()
    result["seed"] = int(result["seed"])
    result["max_weight_cap"] = EXPECTED_CAP_VALUES[result["cap"]]
    result["cap_label"] = "uncapped" if result["cap"] == "uncapped" else result["cap"].replace("p", ".")
    result["candidate_name"] = (
        f"{result['candidate']}_cap_{result['cap']}"
    )
    return result


def collect_run_records(
    sources: dict[str, Path],
    read_log: list[dict[str, Any]],
    ignored_dirs: list[str],
) -> list[dict[str, Any]]:
    """Collect run folder records from source directories."""
    records = []
    seen = set()
    for source_label, source_dir in sources.items():
        per_candidate = source_dir / "per_candidate"
        if not per_candidate.exists():
            raise FileNotFoundError(f"Missing per_candidate directory: {per_candidate}")
        for candidate_dir in sorted(per_candidate.iterdir()):
            if not candidate_dir.is_dir():
                continue
            if candidate_dir.name == "configs":
                ignored_dirs.append(str(candidate_dir))
                continue
            for child in sorted(candidate_dir.iterdir()):
                if not child.is_dir():
                    continue
                if child.name == "configs":
                    ignored_dirs.append(str(child))
                    continue
                try:
                    parsed = parse_run_folder_name(child.name)
                except ValueError:
                    ignored_dirs.append(str(child))
                    continue
                required = {name: child / name for name in REQUIRED_RUN_FILES}
                missing = [str(path) for path in required.values() if not path.exists()]
                if missing:
                    raise FileNotFoundError(
                        f"Missing required files for {child.name}: {missing}"
                    )
                key = (parsed["candidate"], parsed["cap"], parsed["fold"], parsed["seed"])
                if key in seen:
                    raise ValueError(f"Duplicate run detected: {key}")
                seen.add(key)
                records.append(
                    {
                        **parsed,
                        "base_candidate": parsed["candidate"],
                        "source_label": source_label,
                        "source_dir": str(source_dir),
                        "run_dir": str(child),
                        "history_path": str(required["test_policy_history.csv"]),
                        "metrics_path": str(required["test_metrics_table.csv"]),
                        "diagnostics_path": str(required["test_diagnostics.csv"]),
                    },
                )
    return records


def validate_full_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate official full universe coverage."""
    frame = pd.DataFrame(records)
    expected_total = (
        len(EXPECTED_FULL_CANDIDATES)
        * len(EXPECTED_CAP_LABELS)
        * len(EXPECTED_FOLDS)
        * len(EXPECTED_SEEDS)
    )
    if len(frame) != expected_total:
        raise ValueError(f"Expected {expected_total} histories, found {len(frame)}.")
    candidates = sorted(frame["base_candidate"].unique().tolist())
    missing_candidates = sorted(set(EXPECTED_FULL_CANDIDATES) - set(candidates))
    extra_candidates = sorted(set(candidates) - set(EXPECTED_FULL_CANDIDATES))
    if missing_candidates or extra_candidates:
        raise ValueError(
            f"Candidate coverage mismatch; missing={missing_candidates}, extra={extra_candidates}"
        )
    coverage = (
        frame.groupby(["base_candidate", "cap"], dropna=False)
        .size()
        .rename("n_histories")
        .reset_index()
    )
    bad = coverage.loc[
        coverage["n_histories"] != EXPECTED_HISTORIES_PER_CANDIDATE_CAP
    ]
    if not bad.empty:
        raise ValueError(
            "Expected "
            f"{EXPECTED_HISTORIES_PER_CANDIDATE_CAP} histories per candidate-cap; "
            f"bad rows={bad.to_dict('records')}"
        )
    cap_sets = frame.groupby("base_candidate")["cap"].apply(lambda x: sorted(set(x)))
    bad_caps = {
        candidate: caps
        for candidate, caps in cap_sets.items()
        if caps != sorted(EXPECTED_CAP_LABELS)
    }
    if bad_caps:
        raise ValueError(f"Cap coverage mismatch: {bad_caps}")
    return {
        "expected_histories": expected_total,
        "found_histories": len(frame),
        "expected_candidates": EXPECTED_FULL_CANDIDATES,
        "found_candidates": candidates,
        "expected_caps": EXPECTED_CAP_LABELS,
        "histories_per_candidate_cap": coverage.to_dict(orient="records"),
    }


def build_run_metrics_and_observations(
    records: list[dict[str, Any]],
    read_log: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build run-level metrics and return observations from saved run folders."""
    metric_rows = []
    observation_parts = []
    for record in records:
        history = read_csv_with_retry(record["history_path"], read_log=read_log)
        validate_history_diagnostics(history, record)
        metrics = read_agent_metrics(record["metrics_path"], read_log)
        diagnostics = read_diagnostics(record["diagnostics_path"], read_log)
        returns = pd.to_numeric(history["financial_net_return"], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        dates = pd.to_datetime(history.loc[returns.index, "date"], errors="coerce")
        run_id = Path(record["run_dir"]).name
        observation_parts.append(
            pd.DataFrame(
                {
                    "candidate_name": record["candidate_name"],
                    "base_candidate": record["base_candidate"],
                    "run_id": run_id,
                    "date": dates,
                    "return": returns.to_numpy(),
                },
            ),
        )
        metric_rows.append(
            {
                **{key: record[key] for key in [
                    "candidate_name",
                    "base_candidate",
                    "max_weight_cap",
                    "cap_label",
                    "fold",
                    "seed",
                ]},
                "split": "test",
                "cumulative_return": metrics["cumulative_return"],
                "annualized_return": metrics["annualized_return"],
                "annualized_volatility": metrics["annualized_volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "sortino_ratio": metrics["sortino_ratio"],
                "calmar_ratio": metrics["calmar_ratio"],
                "average_turnover": diagnostics["average_turnover"],
                "average_transaction_cost": diagnostics["average_transaction_cost"],
                "average_effective_number_of_assets": diagnostics[
                    "average_effective_number_of_assets"
                ],
                "average_max_weight": diagnostics["average_max_weight"],
                "cash_weight": diagnostics["average_cash_weight"],
                "cash_above_10_rate": (
                    pd.to_numeric(history["cash_weight"], errors="coerce") > 0.10
                ).mean(),
                "mean_btc_weight": mean_or_na(history, "weight_BTC-USD"),
                "mean_btc_transaction_cost_contribution": mean_or_na(
                    history,
                    "asset_transaction_cost_contribution_BTC-USD",
                ),
            },
        )
    return pd.DataFrame(metric_rows), pd.concat(observation_parts, ignore_index=True)


def validate_history_diagnostics(history: pd.DataFrame, record: dict[str, Any]) -> None:
    """Fail fast if a saved history is missing official asset-specific diagnostics."""
    missing = [column for column in REQUIRED_HISTORY_COLUMNS if column not in history.columns]
    if missing:
        raise ValueError(f"{record['run_dir']} missing required history columns: {missing}")
    modes = set(history["transaction_cost_mode"].dropna().astype(str).unique().tolist())
    if modes != {"asset_specific"}:
        raise ValueError(f"{record['run_dir']} has non asset-specific modes: {sorted(modes)}")


def read_agent_metrics(path: str | Path, read_log: list[dict[str, Any]]) -> dict[str, float]:
    """Read the agent row from saved test metrics."""
    metrics = read_csv_with_retry(path, read_log=read_log)
    label_column = metrics.columns[0]
    agent = metrics.loc[metrics[label_column].astype(str) == "agent"]
    if agent.empty:
        raise ValueError(f"Missing agent metrics row: {path}")
    return agent.iloc[0].to_dict()


def read_diagnostics(path: str | Path, read_log: list[dict[str, Any]]) -> dict[str, float]:
    """Read one-row diagnostics."""
    diagnostics = read_csv_with_retry(path, read_log=read_log)
    if diagnostics.empty:
        raise ValueError(f"Empty diagnostics file: {path}")
    return diagnostics.iloc[0].to_dict()


def build_all_candidate_caps(run_metrics: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run metrics and add official combined-universe scores."""
    aggregate = aggregate_official_metrics(run_metrics)
    scored = add_official_robust_scores(aggregate, observations)
    scored["mandate_bucket"] = scored["max_drawdown"].apply(assign_drawdown_bucket)
    scored["recovery_required"] = scored["max_drawdown"].apply(calculate_recovery_required)
    scored["drawdown_multiplier"] = scored["max_drawdown"].apply(get_drawdown_multiplier)
    scored["mandate_aware_score"] = scored["robust_score"] * scored["drawdown_multiplier"]
    scored.loc[scored["mandate_bucket"] == "not_eligible", "mandate_aware_score"] = 0.0
    flagged = compute_reward_incentive_flags(
        scored.rename(columns={"candidate_name": "strategy"})
    ).rename(columns={"strategy": "candidate_name"})
    return select_official_columns(flagged)


def aggregate_official_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run-level rows to candidate-cap rows."""
    rows = []
    group_cols = ["candidate_name", "base_candidate", "max_weight_cap", "cap_label", "split"]
    for keys, group in run_metrics.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_folds"] = group["fold"].nunique()
        row["n_seeds"] = group["seed"].nunique()
        row["episodes"] = 60
        row["n_histories"] = len(group)
        row["cumulative_return"] = group["cumulative_return"].mean()
        row["annualized_return"] = group["annualized_return"].mean()
        row["annualized_volatility"] = group["annualized_volatility"].mean()
        row["sharpe"] = group["sharpe_ratio"].mean()
        row["std_sharpe"] = group["sharpe_ratio"].std()
        row["sortino"] = group["sortino_ratio"].mean()
        row["calmar"] = group["calmar_ratio"].mean()
        row["max_drawdown"] = group["max_drawdown"].mean()
        row["worst_max_drawdown"] = group["max_drawdown"].min()
        row["average_turnover"] = group["average_turnover"].mean()
        row["mean_transaction_cost"] = group["average_transaction_cost"].mean()
        row["average_effective_number_of_assets"] = group[
            "average_effective_number_of_assets"
        ].mean()
        row["average_max_weight"] = group["average_max_weight"].mean()
        row["mean_cash_weight"] = group["cash_weight"].mean()
        row["cash_above_10_rate"] = group["cash_above_10_rate"].mean()
        row["mean_btc_weight"] = group["mean_btc_weight"].mean()
        row["mean_btc_transaction_cost_contribution"] = group[
            "mean_btc_transaction_cost_contribution"
        ].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def add_official_robust_scores(
    aggregate: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Compute DSR and composite robust_score over the combined 35-row universe."""
    rows = aggregate.copy()
    pooled = []
    mean_run = []
    median_run = []
    min_run = []
    max_run = []
    date_averaged = []
    psr = []
    for candidate_name in rows["candidate_name"]:
        obs = observations.loc[observations["candidate_name"] == candidate_name].copy()
        returns = pd.to_numeric(obs["return"], errors="coerce").dropna()
        psr.append(compute_probabilistic_sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR))
        pooled.append(
            compute_deflated_sharpe_ratio(
                returns,
                n_trials=25,
                periods_per_year=PERIODS_PER_YEAR,
            ),
        )
        run_values = []
        for _, group in obs.groupby("run_id", sort=False):
            run_values.append(
                compute_deflated_sharpe_ratio(
                    group["return"],
                    n_trials=25,
                    periods_per_year=PERIODS_PER_YEAR,
                ),
            )
        run_series = pd.Series(run_values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        mean_run.append(float(run_series.mean()) if not run_series.empty else np.nan)
        median_run.append(float(run_series.median()) if not run_series.empty else np.nan)
        min_run.append(float(run_series.min()) if not run_series.empty else np.nan)
        max_run.append(float(run_series.max()) if not run_series.empty else np.nan)
        dated = obs.copy()
        dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
        date_returns = dated.groupby("date")["return"].mean().dropna()
        date_averaged.append(
            compute_deflated_sharpe_ratio(
                date_returns,
                n_trials=25,
                periods_per_year=PERIODS_PER_YEAR,
            )
            if not date_returns.empty
            else np.nan
        )
    rows["psr_score"] = psr
    rows["pooled_dsr_n25"] = pooled
    rows["dsr_n25"] = pooled
    rows["mean_run_dsr_n25"] = mean_run
    rows["median_run_dsr_n25"] = median_run
    rows["min_run_dsr_n25"] = min_run
    rows["max_run_dsr_n25"] = max_run
    rows["date_averaged_dsr_n25"] = date_averaged
    rows["dsr_score"] = pd.Series(median_run).combine_first(pd.Series(date_averaged)).combine_first(
        pd.Series(pooled)
    )
    fallback = normalize_metric_series(rows["sharpe"])
    rows["dsr_score"] = rows["dsr_score"].combine_first(fallback)
    rows["dsr_available"] = rows["dsr_score"].notna()
    rows["dsr_method"] = "median_run"
    scoring_input = rows.rename(
        columns={
            "candidate_name": "strategy",
            "worst_max_drawdown": "worst_drawdown",
            "average_turnover": "turnover",
            "average_effective_number_of_assets": "effective_assets",
        },
    )
    scoring_input["type"] = "drl"
    scored = compute_composite_robust_score(scoring_input)
    return scored.rename(
        columns={
            "strategy": "candidate_name",
            "worst_drawdown": "worst_max_drawdown",
            "turnover": "average_turnover",
            "effective_assets": "average_effective_number_of_assets",
        },
    )


def select_official_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Select official all-candidate-cap report columns."""
    columns = [
        "candidate_name",
        "base_candidate",
        "max_weight_cap",
        "cap_label",
        "split",
        "n_folds",
        "n_seeds",
        "episodes",
        "n_histories",
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
        "cash_above_10_rate",
        "concentration_classification",
        "suspicious_or_lazy_concentration_candidate",
        "justified_concentration_candidate",
        "dsr_score",
        "median_run_dsr_n25",
        "date_averaged_dsr_n25",
        "dsr_method",
    ]
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, columns].sort_values(
        ["mandate_aware_score", "robust_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_official_selected_candidates(all_caps: pd.DataFrame) -> pd.DataFrame:
    """Select best cap per base candidate by mandate-aware score then robust score."""
    rows = []
    for base_candidate, group in all_caps.groupby("base_candidate", sort=False):
        best = group.sort_values(
            ["mandate_aware_score", "robust_score"],
            ascending=[False, False],
        ).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).sort_values(
        ["mandate_aware_score", "robust_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_official_main_ranking(selected: pd.DataFrame) -> pd.DataFrame:
    """Rank selected candidates by mandate-aware score."""
    ranking = selected.copy()
    ranking["rank_mandate_aware"] = (
        pd.to_numeric(ranking["mandate_aware_score"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    ranking["rank_robust"] = (
        pd.to_numeric(ranking["robust_score"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    return ranking.sort_values(
        ["rank_mandate_aware", "rank_robust"],
    ).reset_index(drop=True)


def build_best_by_metric(all_caps: pd.DataFrame) -> pd.DataFrame:
    """Identify best candidate-cap by each requested metric."""
    specs = [
        ("mandate_aware_score", False),
        ("robust_score", False),
        ("sharpe", False),
        ("max_drawdown", False),
        ("average_turnover", True),
        ("average_effective_number_of_assets", False),
        ("average_max_weight", True),
    ]
    rows = []
    for metric, ascending in specs:
        frame = all_caps.copy()
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        best = frame.dropna(subset=[metric]).sort_values(metric, ascending=ascending).iloc[0]
        rows.append(
            {
                "metric": metric,
                "best_candidate_name": best["candidate_name"],
                "base_candidate": best["base_candidate"],
                "cap_label": best["cap_label"],
                "metric_value": best[metric],
                "mandate_aware_score": best["mandate_aware_score"],
                "robust_score": best["robust_score"],
            },
        )
    return pd.DataFrame(rows)


def build_official_metadata(
    sources: dict[str, Path],
    output_dir: Path,
    run_records: list[dict[str, Any]],
    validation: dict[str, Any],
    read_log: list[dict[str, Any]],
    ignored_dirs: list[str],
) -> dict[str, Any]:
    """Build official report metadata."""
    retries = [entry for entry in read_log if entry.get("attempts", 1) > 1]
    failures = [entry for entry in read_log if entry.get("status") == "failed"]
    return {
        "runner": "src.analysis.asset_specific_cost_final_report",
        "output_dir": str(output_dir),
        "source_dirs": {key: str(path) for key, path in sources.items()},
        "expected_histories": validation["expected_histories"],
        "found_histories": validation["found_histories"],
        "candidates": EXPECTED_FULL_CANDIDATES,
        "caps": EXPECTED_CAP_LABELS,
        "score_scope": "combined_asset_specific_full_universe",
        "cost_model": EXPECTED_COST_MODEL,
        "validation_results": validation,
        "read_retry_count": len(retries),
        "read_retries": retries,
        "read_failures": failures,
        "ignored_dirs": ignored_dirs,
        "v7_v8_official_result_note": (
            "V7/V8 source cap_sensitivity_all_results.csv files were available, "
            "but robust_score and mandate_aware_score were recomputed across the "
            "combined V2-V8 asset-specific full universe."
        ),
        "caveats": [
            "This report is reconstructed from existing histories and does not retrain.",
            "This should not be mixed casually with scalar-cost results.",
            "Benchmark comparison, statistical validation, WRC, regime analysis, and mandate-profile analysis still need regeneration under the same cost model before paper-level claims.",
        ],
    }


def build_official_summary_markdown(
    all_caps: pd.DataFrame,
    selected: pd.DataFrame,
    main_ranking: pd.DataFrame,
    best_by_metric: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    """Build official markdown summary."""
    best_mandate = best_by_metric.set_index("metric").loc["mandate_aware_score"]
    best_robust = best_by_metric.set_index("metric").loc["robust_score"]
    best_sharpe = best_by_metric.set_index("metric").loc["sharpe"]
    capped = all_caps[all_caps["cap_label"] != "uncapped"]
    uncapped = all_caps[all_caps["cap_label"] == "uncapped"].set_index("base_candidate")
    cap_dominance = []
    for _, row in selected.iterrows():
        base = row["base_candidate"]
        if base in uncapped.index:
            cap_dominance.append(
                {
                    "base_candidate": base,
                    "selected_cap": row["cap_label"],
                    "selected_beats_uncapped_mandate": (
                        float(row["mandate_aware_score"])
                        > float(uncapped.loc[base, "mandate_aware_score"])
                    ),
                    "selected_beats_uncapped_robust": (
                        float(row["robust_score"])
                        > float(uncapped.loc[base, "robust_score"])
                    ),
                },
            )
    n_caps_selected = sum(row["selected_cap"] != "uncapped" for row in cap_dominance)
    lines = [
        "# Official Asset-Specific-Cost Full Final Candidate Report",
        "",
        "This is the official full final candidate universe under asset-specific-cost-aware TD3 training.",
        "",
        f"Histories found / expected: {metadata['found_histories']} / {metadata['expected_histories']}.",
        "",
        "Scores were recomputed over `combined_asset_specific_full_universe`, covering all 35 candidate-cap rows.",
        "",
        "## Top Results",
        "",
        f"- Best by mandate-aware score: `{best_mandate['best_candidate_name']}` ({float(best_mandate['metric_value']):.6f}).",
        f"- Best by robust score: `{best_robust['best_candidate_name']}` ({float(best_robust['metric_value']):.6f}).",
        f"- Best by raw Sharpe: `{best_sharpe['best_candidate_name']}` ({float(best_sharpe['metric_value']):.6f}).",
        "",
        f"Selected caps are capped variants for {n_caps_selected} of {len(cap_dominance)} candidates.",
        "",
        "## Selected Candidate Ranking",
        "",
    ]
    for _, row in main_ranking.iterrows():
        lines.append(
            f"{int(row['rank_mandate_aware'])}. `{row['candidate_name']}` "
            f"(mandate-aware {float(row['mandate_aware_score']):.6f}, "
            f"robust {float(row['robust_score']):.6f}, Sharpe {float(row['sharpe']):.4f})."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These asset-specific-cost-aware results should not be mixed casually with scalar-cost results.",
            "If the leading candidate differs from the scalar-cost paper results, that should be treated as evidence that transaction-cost-aware training can affect model selection.",
            "",
            "Benchmark comparison, statistical validation, White Reality Check, regime analysis, and mandate-profile analysis still need regeneration under the same asset-specific cost model before paper-level claims.",
        ],
    )
    return "\n".join(lines) + "\n"


def load_cap_results(directory: Path) -> pd.DataFrame:
    """Load one cap-sensitivity result table."""
    path = directory / CAP_RESULTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing cap sensitivity results: {path}")
    frame = pd.read_csv(path)
    frame = frame[frame.get("split", "test") == "test"].copy()
    if "cap_label" not in frame.columns:
        frame["cap_label"] = frame["max_weight_cap"].map(format_cap_label)
    return frame


def enrich_with_history_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach transaction-cost and BTC diagnostics from saved policy histories."""
    records = []
    for _, row in frame.iterrows():
        record = row.to_dict()
        diagnostics = history_diagnostics_for_candidate(row)
        record.update(diagnostics)
        records.append(record)
    return pd.DataFrame(records)


def history_diagnostics_for_candidate(row: pd.Series) -> dict[str, Any]:
    """Compute diagnostic means from matching per-fold/per-seed policy histories."""
    candidate_name = str(row["candidate_name"])
    output_dir = Path(str(row["candidate_output_dir"]))
    histories = sorted(output_dir.glob(f"F*_{candidate_name}_seed_*/test_policy_history.csv"))
    config_paths = sorted(output_dir.glob(f"configs/F*_{candidate_name}_seed_*.yaml"))
    config_info = read_transaction_cost_config(config_paths[0]) if config_paths else {}

    if not histories:
        return {
            **config_info,
            "average_btc_cost_contribution": pd.NA,
            "average_btc_allocation": pd.NA,
            "history_files_found": 0,
        }

    frames = [pd.read_csv(path) for path in histories]
    history = pd.concat(frames, ignore_index=True, sort=False)
    return {
        **config_info,
        "average_btc_cost_contribution": mean_or_na(
            history,
            "asset_transaction_cost_contribution_BTC-USD",
        ),
        "average_btc_allocation": mean_or_na(history, "weight_BTC-USD"),
        "history_files_found": len(histories),
    }


def read_transaction_cost_config(path: Path) -> dict[str, Any]:
    """Read transaction-cost settings from one generated run config."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    environment = config.get("environment", {})
    return {
        "transaction_cost_mode": environment.get("transaction_cost_mode", "scalar"),
        "asset_transaction_cost_bps": environment.get("asset_transaction_cost_bps"),
    }


def build_main_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Build all-row ranking table."""
    result = frame.copy()
    result["strategy_name"] = result["candidate_name"]
    result["rank_mandate_aware"] = (
        pd.to_numeric(result["mandate_aware_score"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    result["rank_robust"] = (
        pd.to_numeric(result["robust_score"], errors="coerce")
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    for column in REPORT_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result = result[REPORT_COLUMNS + ["history_files_found", "input_label"]]
    return result.sort_values(
        ["rank_mandate_aware", "rank_robust", "strategy_name"],
        na_position="last",
    ).reset_index(drop=True)


def build_selected_candidates(main_ranking: pd.DataFrame) -> pd.DataFrame:
    """Identify best rows by the requested diagnostic lenses."""
    selections = [
        ("best_by_mandate_aware_score", "mandate_aware_score", False),
        ("best_by_robust_score", "robust_score", False),
        ("best_by_drawdown", "max_drawdown", False),
        ("best_by_turnover", "average_turnover", True),
        ("best_by_effective_assets", "average_effective_number_of_assets", False),
    ]
    rows = []
    for selection, metric, ascending in selections:
        valid = main_ranking.copy()
        valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
        valid = valid.dropna(subset=[metric])
        if valid.empty:
            continue
        best = valid.sort_values(metric, ascending=ascending).iloc[0].to_dict()
        best["selection"] = selection
        best["selection_metric"] = metric
        best["selection_metric_value"] = best[metric]
        rows.append(best)
    if not rows:
        return pd.DataFrame()
    columns = [
        "selection",
        "selection_metric",
        "selection_metric_value",
        "strategy_name",
        "base_candidate",
        "cap_label",
        "mandate_aware_score",
        "robust_score",
        "max_drawdown",
        "average_turnover",
        "average_effective_number_of_assets",
        "average_max_weight",
        "average_btc_cost_contribution",
        "average_btc_allocation",
        "transaction_cost_mode",
        "asset_transaction_cost_bps",
    ]
    return pd.DataFrame(rows).loc[:, columns]


def build_summary_markdown(
    main_ranking: pd.DataFrame,
    selected: pd.DataFrame,
    warnings: list[str],
) -> str:
    """Build cautious markdown summary."""
    top_mandate = main_ranking.sort_values("rank_mandate_aware").iloc[0]
    top_robust = main_ranking.sort_values("rank_robust").iloc[0]
    lines = [
        "# Asset-Specific Transaction Cost Limited Final Report",
        "",
        "This report combines limited TD3 retraining runs under asset-specific "
        "transaction costs. It is reporting-only and does not retrain models.",
        "",
        "The comparison is limited to V3 clean no-DXY macro, V7 clean no-DXY "
        "+ GARCH, and V4 real GARCH candidates. It is not the full original "
        "candidate universe.",
        "",
        "Scalar-cost and asset-specific-cost results are not directly "
        "interchangeable. The imported robust and mandate-aware scores may be "
        "normalized within each cap-sensitivity report universe, so cross-candidate "
        "score comparisons should be treated as preliminary.",
        "",
        "## Top Rows",
        "",
        f"- Best by mandate-aware score: `{top_mandate['strategy_name']}` "
        f"({float(top_mandate['mandate_aware_score']):.6f}).",
        f"- Best by robust score: `{top_robust['strategy_name']}` "
        f"({float(top_robust['robust_score']):.6f}).",
        "",
        "## Interpretation",
        "",
        "Preliminary evidence suggests the preferred TD3 candidate may change "
        "under asset-specific-cost-aware training. In this limited set, the "
        "combined report should be used as a prioritization diagnostic, not as "
        "a final superiority claim.",
        "",
        "Do not claim final superiority until benchmark comparisons and "
        "statistical validation are regenerated under the same asset-specific "
        "cost model.",
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(["", "## Selection Table", ""])
    if selected.empty:
        lines.append("No selected candidates could be computed.")
    else:
        for _, row in selected.iterrows():
            lines.append(
                f"- {row['selection']}: `{row['strategy_name']}` "
                f"({row['selection_metric']} = {float(row['selection_metric_value']):.6f})."
            )
    return "\n".join(lines) + "\n"


def build_metadata(
    input_dirs: dict[str, Path],
    metadata_inputs: dict[str, dict[str, Any]],
    output_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Build metadata for the combined report."""
    cost_assumptions = extract_cost_assumptions(metadata_inputs)
    return {
        "runner": "src.analysis.asset_specific_cost_final_report",
        "output_dir": str(output_dir),
        "input_dirs": {key: str(path) for key, path in input_dirs.items()},
        "cost_assumptions": cost_assumptions,
        "warnings": warnings,
        "caveats": [
            "Limited retraining subset only; this is not the full original candidate universe.",
            "Scalar-cost and asset-specific-cost results are not directly interchangeable.",
            "Imported robust and mandate-aware scores may be normalized within each report universe.",
            "Benchmark comparisons and statistical validation must be regenerated under the same cost model before final claims.",
        ],
    }


def load_metadata(directory: Path) -> dict[str, Any]:
    """Load optional cap sensitivity metadata."""
    path = directory / CAP_METADATA_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_cost_assumptions(metadata_inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Collect cost assumptions from input metadata when available."""
    assumptions = {
        "transaction_cost_mode": "asset_specific",
        "asset_transaction_cost_bps": {
            "SPY": 2.0,
            "TLT": 2.0,
            "GLD": 2.0,
            "BTC-USD": 10.0,
            "CASH": 0.0,
        },
    }
    for metadata in metadata_inputs.values():
        if "transaction_cost" in metadata:
            assumptions["legacy_scalar_transaction_cost_field"] = metadata[
                "transaction_cost"
            ]
    return assumptions


def mean_or_na(frame: pd.DataFrame, column: str) -> float | Any:
    """Return column mean or missing value if unavailable."""
    if column not in frame.columns:
        return pd.NA
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return pd.NA
    return float(values.mean())


def format_cap_label(value: Any) -> str:
    """Format cap labels consistently."""
    if pd.isna(value):
        return "uncapped"
    return f"{float(value):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined report for asset-specific transaction cost retraining.",
    )
    parser.add_argument("--v2-v6-dir", default=None)
    parser.add_argument("--v7-dir", default=None)
    parser.add_argument("--v8-dir", default=None)
    parser.add_argument("--v3-dir", default=DEFAULT_V3_DIR)
    parser.add_argument("--v4-dir", default=DEFAULT_V4_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.v2_v6_dir or args.v8_dir:
        report = build_official_asset_specific_cost_full_report(
            v2_v6_dir=args.v2_v6_dir or DEFAULT_V2_V6_DIR,
            v7_dir=args.v7_dir or DEFAULT_FULL_V7_DIR,
            v8_dir=args.v8_dir or DEFAULT_FULL_V8_DIR,
            output_dir=args.output_dir,
        )
        print("Official asset-specific selected candidates:")
        print(report["main_ranking"].to_string(index=False))
    else:
        report = build_asset_specific_cost_final_report(
            v3_dir=args.v3_dir,
            v7_dir=args.v7_dir or DEFAULT_V7_DIR,
            v4_dir=args.v4_dir,
            output_dir=args.output_dir,
        )
        print("Asset-specific cost selected candidates:")
        print(report["selected_candidates"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
