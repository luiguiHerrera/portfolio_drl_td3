"""IBKR-inspired transaction cost sensitivity report.

This reporting-only layer reads existing out-of-sample strategy histories and
recomputes return metrics under broker-inspired transaction-cost assumptions.
It does not retrain TD3, does not overwrite experiment outputs, and does not
change existing rankings.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.mandate_aware_score import (
    assign_drawdown_bucket,
    get_drawdown_multiplier,
)
from src.analysis.statistical_validation_report import (
    BENCHMARK_HISTORY_DIR,
    METADATA_FILE,
    SELECTED_FILE,
    _benchmark_history_dir,
    _benchmark_names_for_report,
    _cap_label,
    _source_dir_for_row,
    _with_cap_sensitivity_overrides,
)


DEFAULT_FINAL_REPORT_DIR = (
    "outputs/tables/"
    "final_constrained_td3_report_with_v3_clean_no_dxy_v7_clean_garch_v4_v7_v8_60ep_10seeds"
)
DEFAULT_OUTPUT_DIR = "outputs/tables/transaction_cost_sensitivity_final"
WEEKLY_PERIODS_PER_YEAR = 52
KEY_V3_CLEAN = "V3_real_macro_vintage_clean_no_dxy_cap_0.50"
KEY_V7_CLEAN_GARCH = "V7_real_macro_vintage_clean_no_dxy_garch_cap_0.50"


@dataclass(frozen=True)
class CostScenario:
    name: str
    etf_stock_bps: float
    btc_bps: float
    blended_bps: float
    use_existing_net_returns: bool = False

    @property
    def etf_stock_rate(self) -> float:
        return self.etf_stock_bps / 10_000.0

    @property
    def btc_rate(self) -> float:
        return self.btc_bps / 10_000.0

    @property
    def blended_rate(self) -> float:
        return self.blended_bps / 10_000.0


SCENARIOS = [
    CostScenario("existing", 0.0, 0.0, 0.0, use_existing_net_returns=True),
    CostScenario("ibkr_proxy", 2.0, 10.0, 5.0),
    CostScenario("stress", 5.0, 30.0, 10.0),
]


def build_transaction_cost_sensitivity_report(
    final_report_dir: str = DEFAULT_FINAL_REPORT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    v3_cap_sensitivity_dir: str | None = None,
    v3_vintage_cap_sensitivity_dir: str | None = None,
    v3_clean_no_dxy_cap_sensitivity_dir: str | None = None,
    v4_cap_sensitivity_dir: str | None = None,
    v7_cap_sensitivity_dir: str | None = None,
    v7_clean_no_dxy_garch_cap_sensitivity_dir: str | None = None,
    v8_cap_sensitivity_dir: str | None = None,
) -> dict[str, Any]:
    """Build transaction-cost sensitivity tables from existing histories."""
    final_dir = Path(final_report_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(final_dir / SELECTED_FILE)
    main_ranking = pd.read_csv(final_dir / "final_constrained_td3_main_ranking.csv")
    final_metrics = main_ranking.set_index("strategy_name").to_dict(orient="index")
    metadata = json.loads((final_dir / METADATA_FILE).read_text(encoding="utf-8"))
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

    histories, history_records, warnings = locate_history_frames(
        final_report_dir=final_dir,
        selected_candidates=selected,
        metadata=metadata,
    )
    rows: list[dict[str, Any]] = []
    method_records: list[dict[str, Any]] = []
    for strategy, payload in histories.items():
        frame = payload["frame"]
        strategy_type = payload["strategy_type"]
        base_metrics = final_metrics.get(strategy, {})
        for scenario in SCENARIOS:
            adjusted = apply_cost_scenario(frame, scenario)
            rows.append(
                build_summary_row(
                    strategy=strategy,
                    strategy_type=strategy_type,
                    scenario=scenario,
                    adjusted=adjusted,
                    base_metrics=base_metrics,
                )
            )
            method_records.append(
                {
                    "strategy": strategy,
                    "scenario": scenario.name,
                    "cost_method": adjusted["cost_method"],
                    "asset_level_turnover_available": adjusted["asset_level_turnover_available"],
                    "blended_proxy_used": adjusted["blended_proxy_used"],
                }
            )

    summary = pd.DataFrame(rows)
    summary["rank_within_scenario"] = (
        summary.groupby("scenario")["mandate_score_or_available_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    summary = summary.sort_values(["scenario", "rank_within_scenario", "strategy"]).reset_index(drop=True)
    winners = build_winners_table(summary)
    markdown = build_summary_markdown(winners, method_records)

    paths = {
        "summary": out_dir / "transaction_cost_sensitivity_summary.csv",
        "winners": out_dir / "transaction_cost_sensitivity_winners.csv",
        "metadata": out_dir / "transaction_cost_sensitivity_metadata.json",
        "markdown": out_dir / "transaction_cost_sensitivity_summary.md",
    }
    summary.to_csv(paths["summary"], index=False)
    winners.to_csv(paths["winners"], index=False)
    paths["markdown"].write_text(markdown, encoding="utf-8")

    methods = pd.DataFrame(method_records)
    metadata_out = {
        "final_report_dir": str(final_dir),
        "history_dirs": history_records,
        "cost_assumptions": [asdict(scenario) for scenario in SCENARIOS],
        "asset_level_turnover_available_any": bool(methods["asset_level_turnover_available"].any()) if not methods.empty else False,
        "blended_proxy_used_any": bool(methods["blended_proxy_used"].any()) if not methods.empty else False,
        "cost_methods_by_strategy": method_records,
        "source_notes": [
            "US stocks/ETFs: IBKR Pro tiered pricing starts at 0.0035 USD/share for <=300,000 monthly shares with a 0.35 USD minimum per order.",
            "US stocks/ETFs: IBKR fixed pricing is 0.005 USD/share with a 1.00 USD minimum per order and 1% of trade value maximum.",
            "Crypto through IBKR/Paxos/Zero Hash has published commissions of 0.18%, 0.15%, or 0.12% of trade value depending on monthly volume, with a 1.75 USD minimum per order capped at 1% of trade value.",
            "CASH is assigned zero transaction cost.",
        ],
        "caveats": [
            "This is a reporting-only approximation, not an exact routed-order IBKR execution simulation.",
            "Weekly portfolio histories do not contain order sizes, share counts, venue, spreads, or minimum-order effects.",
            "When asset-level turnover cannot be inferred from weight columns, total turnover is multiplied by a blended proxy cost.",
            "Existing experiment outputs are not overwritten.",
        ],
        "warnings": warnings,
    }
    paths["metadata"].write_text(json.dumps(_json_safe(metadata_out), indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "winners": winners,
        "metadata": metadata_out,
        "warnings": warnings,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def locate_history_frames(
    final_report_dir: Path,
    selected_candidates: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Locate and date-average TD3 frames plus benchmark history frames."""
    histories: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for _, row in selected_candidates.iterrows():
        strategy_name = str(row["strategy_name"])
        base_candidate = str(row["base_candidate"])
        source_dir = _source_dir_for_row(row, metadata)
        cap_label = _cap_label(row.get("selected_cap"))
        if source_dir is None:
            warnings.append(f"Missing source directory for {strategy_name}.")
            continue
        per_candidate = source_dir / "per_candidate" / base_candidate
        pattern = f"*_{base_candidate}_cap_{cap_label}_seed_*/test_policy_history.csv"
        paths = sorted(per_candidate.glob(pattern))
        if not paths:
            warnings.append(f"No TD3 histories found for {strategy_name}.")
            continue
        frame = load_date_averaged_history_frame(paths)
        histories[strategy_name] = {"frame": frame, "strategy_type": "td3"}
        records.append(
            {
                "strategy": strategy_name,
                "strategy_type": "td3",
                "source": str(per_candidate),
                "n_history_files": len(paths),
                "n_periods": len(frame),
            }
        )

    benchmark_dir = _benchmark_history_dir(final_report_dir, metadata)
    for benchmark_name in _benchmark_names_for_report(final_report_dir):
        path = benchmark_dir / f"{benchmark_name}_history.csv"
        if not path.exists():
            warnings.append(f"No benchmark history found for {benchmark_name}.")
            continue
        frame = load_history_frame(path)
        histories[benchmark_name] = {"frame": frame, "strategy_type": "benchmark"}
        records.append(
            {
                "strategy": benchmark_name,
                "strategy_type": "benchmark",
                "source": str(path),
                "n_history_files": 1,
                "n_periods": len(frame),
            }
        )
    return histories, records, warnings


def load_history_frame(path: Path) -> pd.DataFrame:
    """Load one policy/benchmark history with normalized date ordering."""
    frame = pd.read_csv(path)
    date_column = "date" if "date" in frame.columns else frame.columns[0]
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame[date_column])
    frame = frame.replace([np.inf, -np.inf], np.nan).sort_values("date")
    return frame.dropna(subset=["date"]).reset_index(drop=True)


def load_date_averaged_history_frame(paths: list[Path]) -> pd.DataFrame:
    """Date-average duplicate TD3 fold/seed rows for numeric columns."""
    frames = [load_history_frame(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        col
        for col in combined.columns
        if col != "date" and pd.api.types.is_numeric_dtype(combined[col])
    ]
    return combined.groupby("date", sort=True)[numeric_columns].mean().reset_index()


def apply_cost_scenario(frame: pd.DataFrame, scenario: CostScenario) -> dict[str, Any]:
    """Return adjusted returns and cost diagnostics for a scenario."""
    returns = base_return_series(frame, scenario)
    if scenario.use_existing_net_returns:
        cost_drag = pd.Series(0.0, index=returns.index)
        return {
            "returns": returns,
            "cost_drag": cost_drag,
            "cost_method": "existing_net_returns_as_is",
            "asset_level_turnover_available": has_asset_level_weights(frame),
            "blended_proxy_used": False,
            "frame": frame,
        }

    cost_drag, method, asset_level_available, blended_used = estimate_cost_drag(frame, scenario)
    adjusted = returns - cost_drag.reindex(returns.index).fillna(0.0)
    return {
        "returns": adjusted,
        "cost_drag": cost_drag.reindex(returns.index).fillna(0.0),
        "cost_method": method,
        "asset_level_turnover_available": asset_level_available,
        "blended_proxy_used": blended_used,
        "frame": frame,
    }


def base_return_series(frame: pd.DataFrame, scenario: CostScenario) -> pd.Series:
    """Return existing net or gross-like returns depending on scenario."""
    if scenario.use_existing_net_returns:
        column = "financial_net_return" if "financial_net_return" in frame.columns else "portfolio_return"
    else:
        if "gross_return" in frame.columns:
            column = "gross_return"
        elif "portfolio_return" in frame.columns:
            column = "portfolio_return"
        elif "financial_net_return" in frame.columns:
            column = "financial_net_return"
        else:
            raise ValueError("History frame has no valid return column.")
    return pd.Series(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float), index=frame.index)


def estimate_cost_drag(
    frame: pd.DataFrame,
    scenario: CostScenario,
) -> tuple[pd.Series, str, bool, bool]:
    """Estimate one-way cost drag from asset-level weights or blended turnover."""
    if has_asset_level_weights(frame):
        asset_turnover = infer_asset_level_turnover(frame)
        drag = pd.Series(0.0, index=frame.index, dtype=float)
        for column in asset_turnover.columns:
            asset = column.removeprefix("weight_")
            if asset == "CASH":
                rate = 0.0
            elif asset == "BTC-USD":
                rate = scenario.btc_rate
            else:
                rate = scenario.etf_stock_rate
            drag = drag + asset_turnover[column] * rate
        return drag, "asset_weight_turnover_proxy", True, False

    turnover = pd.to_numeric(frame.get("turnover", pd.Series(0.0, index=frame.index)), errors="coerce")
    drag = turnover.fillna(0.0).astype(float) * scenario.blended_rate
    return drag.reset_index(drop=True), "blended_total_turnover_proxy", False, True


def has_asset_level_weights(frame: pd.DataFrame) -> bool:
    weight_columns = [col for col in frame.columns if str(col).startswith("weight_")]
    return bool(weight_columns)


def infer_asset_level_turnover(frame: pd.DataFrame) -> pd.DataFrame:
    """Infer absolute per-asset weight changes from saved weights."""
    weight_columns = [col for col in frame.columns if str(col).startswith("weight_")]
    weights = frame[weight_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    changes = weights.diff().abs()
    if len(weights):
        total_turnover = pd.to_numeric(frame.get("turnover", pd.Series(np.nan, index=frame.index)), errors="coerce")
        first_turnover = total_turnover.iloc[0] if len(total_turnover) else np.nan
        if pd.notna(first_turnover) and float(first_turnover) > 0:
            weight_sum = float(weights.iloc[0].abs().sum())
            if weight_sum > 0:
                changes.iloc[0] = weights.iloc[0].abs() / weight_sum * float(first_turnover)
            else:
                changes.iloc[0] = 0.0
        else:
            changes.iloc[0] = weights.iloc[0].abs()
    return changes.fillna(0.0)


def build_summary_row(
    strategy: str,
    strategy_type: str,
    scenario: CostScenario,
    adjusted: dict[str, Any],
    base_metrics: dict[str, Any],
) -> dict[str, Any]:
    returns = pd.to_numeric(adjusted["returns"], errors="coerce").dropna()
    metrics = compute_metrics(returns)
    frame = adjusted["frame"]
    avg_turnover = _mean_column(frame, "turnover")
    avg_effective = _metric_from_base_or_nan(base_metrics, "average_effective_number_of_assets")
    avg_max_weight = _mean_column(frame, "max_weight")
    if pd.isna(avg_max_weight):
        avg_max_weight = _metric_from_base_or_nan(base_metrics, "average_max_weight")
    robust_score = _metric_from_base_or_nan(base_metrics, "robust_score")
    mandate_score = approximate_mandate_score(
        robust_score=robust_score,
        max_drawdown=metrics["max_drawdown"],
        sharpe=metrics["sharpe"],
    )
    cost_drag = pd.to_numeric(adjusted["cost_drag"], errors="coerce").fillna(0.0)
    return {
        "scenario": scenario.name,
        "strategy": strategy,
        "strategy_type": strategy_type,
        "n_periods": len(returns),
        "cumulative_return": metrics["cumulative_return"],
        "annualized_return": metrics["annualized_return"],
        "annualized_volatility": metrics["annualized_volatility"],
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "max_drawdown": metrics["max_drawdown"],
        "average_turnover": avg_turnover,
        "average_effective_number_of_assets": avg_effective,
        "average_max_weight": avg_max_weight,
        "average_cost_drag_weekly": float(cost_drag.mean()) if len(cost_drag) else 0.0,
        "annualized_cost_drag": float(cost_drag.mean() * WEEKLY_PERIODS_PER_YEAR) if len(cost_drag) else 0.0,
        "mandate_score_or_available_score": mandate_score,
        "cost_method": adjusted["cost_method"],
        "asset_level_turnover_available": adjusted["asset_level_turnover_available"],
        "blended_proxy_used": adjusted["blended_proxy_used"],
    }


def compute_metrics(returns: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
        }
    cumulative = float((1.0 + clean).prod() - 1.0)
    annualized_return = float((1.0 + cumulative) ** (WEEKLY_PERIODS_PER_YEAR / len(clean)) - 1.0)
    weekly_vol = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    annualized_vol = weekly_vol * np.sqrt(WEEKLY_PERIODS_PER_YEAR) if np.isfinite(weekly_vol) else np.nan
    sharpe = annualized_return / annualized_vol if annualized_vol and annualized_vol > 0 else np.nan
    downside = clean[clean < 0.0]
    downside_vol = float(downside.std(ddof=1)) * np.sqrt(WEEKLY_PERIODS_PER_YEAR) if len(downside) > 1 else np.nan
    sortino = annualized_return / downside_vol if downside_vol and downside_vol > 0 else np.nan
    equity = (1.0 + clean).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_volatility": float(annualized_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
    }


def approximate_mandate_score(
    robust_score: float,
    max_drawdown: float,
    sharpe: float,
) -> float:
    """Approximate scenario score using existing robust_score when available."""
    if pd.notna(robust_score):
        bucket = assign_drawdown_bucket(max_drawdown)
        if bucket == "not_eligible":
            return 0.0
        return float(robust_score) * get_drawdown_multiplier(max_drawdown)
    if pd.isna(sharpe):
        return np.nan
    bucket = assign_drawdown_bucket(max_drawdown)
    if bucket == "not_eligible":
        return 0.0
    return max(float(sharpe), 0.0) * get_drawdown_multiplier(max_drawdown)


def build_winners_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in [scenario.name for scenario in SCENARIOS]:
        group = summary[summary["scenario"] == scenario].sort_values("rank_within_scenario")
        if group.empty:
            continue
        best_overall = group.iloc[0]
        td3 = group[group["strategy_type"] == "td3"]
        benchmark = group[group["strategy_type"] == "benchmark"]
        best_td3 = td3.iloc[0] if not td3.empty else None
        best_benchmark = benchmark.iloc[0] if not benchmark.empty else None
        rows.append(
            {
                "scenario": scenario,
                "best_overall": best_overall["strategy"],
                "best_td3": _row_value(best_td3, "strategy"),
                "best_benchmark": _row_value(best_benchmark, "strategy"),
                "leading_v3_clean_rank": _strategy_rank(group, KEY_V3_CLEAN),
                "leading_v3_clean_score": _strategy_score(group, KEY_V3_CLEAN),
                "v7_clean_garch_rank": _strategy_rank(group, KEY_V7_CLEAN_GARCH),
                "v7_clean_garch_score": _strategy_score(group, KEY_V7_CLEAN_GARCH),
                "interpretation": interpret_scenario(group),
            }
        )
    return pd.DataFrame(rows)


def interpret_scenario(group: pd.DataFrame) -> str:
    leader = str(group.iloc[0]["strategy"])
    v3_rank = _strategy_rank(group, KEY_V3_CLEAN)
    if leader == KEY_V3_CLEAN:
        return "leading_v3_clean_remains_top"
    if pd.notna(v3_rank) and v3_rank <= 3:
        return "main_conclusion_survives_with_v3_near_top"
    return "ranking_sensitive_to_transaction_cost_assumption"


def build_summary_markdown(
    winners: pd.DataFrame,
    method_records: list[dict[str, Any]],
) -> str:
    methods = pd.DataFrame(method_records)
    blended_used = bool(methods["blended_proxy_used"].any()) if not methods.empty else False
    asset_level_used = bool(methods["asset_level_turnover_available"].any()) if not methods.empty else False
    lines = [
        "# Transaction Cost Sensitivity Report",
        "",
        "This is a reporting-only transaction cost realism layer. It does not retrain TD3 and does not overwrite existing experiment outputs.",
        "",
        "The scenarios are inspired by Interactive Brokers pricing, but they are approximations rather than exact routed-order execution costs.",
        "",
        "ETF/stock and crypto costs differ materially. CASH is assigned zero transaction cost.",
        "",
        f"Asset-level weight turnover proxy used: `{asset_level_used}`.",
        f"Blended total-turnover proxy used: `{blended_used}`.",
        "",
        "The aim is to test whether the mandate-aware result survives higher transaction-cost assumptions, not to claim exact IBKR backtest execution.",
        "",
        "## Winners",
        "",
    ]
    for _, row in winners.iterrows():
        lines.append(
            f"- `{row['scenario']}`: best overall `{row['best_overall']}`, "
            f"best TD3 `{row['best_td3']}`, best benchmark `{row['best_benchmark']}`; "
            f"interpretation `{row['interpretation']}`."
        )
    return "\n".join(lines) + "\n"


def _mean_column(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _metric_from_base_or_nan(base_metrics: dict[str, Any], key: str) -> float:
    value = pd.to_numeric(pd.Series([base_metrics.get(key)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _row_value(row: pd.Series | None, key: str) -> str | float:
    if row is None:
        return np.nan
    return row[key]


def _strategy_rank(group: pd.DataFrame, strategy: str) -> float:
    rows = group[group["strategy"] == strategy]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0]["rank_within_scenario"])


def _strategy_score(group: pd.DataFrame, strategy: str) -> float:
    rows = group[group["strategy"] == strategy]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0]["mandate_score_or_available_score"])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if not isinstance(value, (str, bool, dict, list)) and pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IBKR-inspired transaction cost sensitivity report.",
    )
    parser.add_argument("--final-report-dir", default=DEFAULT_FINAL_REPORT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v3-cap-sensitivity-dir", default=None)
    parser.add_argument("--v3-vintage-cap-sensitivity-dir", default=None)
    parser.add_argument("--v3-clean-no-dxy-cap-sensitivity-dir", default=None)
    parser.add_argument("--v4-cap-sensitivity-dir", default=None)
    parser.add_argument("--v7-cap-sensitivity-dir", default=None)
    parser.add_argument("--v7-clean-no-dxy-garch-cap-sensitivity-dir", default=None)
    parser.add_argument("--v8-cap-sensitivity-dir", default=None)
    args = parser.parse_args()

    result = build_transaction_cost_sensitivity_report(
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
    )
    print("Transaction cost sensitivity winners:")
    print(result["winners"].to_string(index=False))
    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
