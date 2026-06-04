"""Reporting-only execution spread robustness for final corrected histories.

This module recomputes already-realized strategy histories under additional
bid-ask spread assumptions. It does not retrain models, alter the portfolio
environment, or select new final winners.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.evaluate_policy import (
    annualized_return,
    annualized_volatility,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
)
from src.costs.spread_costs import build_proxy_weekly_spreads


PERIODS_PER_YEAR = 52
ASSETS = ["SPY", "TLT", "GLD", "BTC-USD", "CASH"]
DEFAULT_ZERO_TD3_DIR = "outputs/tables/final_corrected_limited_td3_60ep_10seeds"
DEFAULT_ZERO_BENCHMARK_DIR = (
    "~/Projects/portfolio_drl_outputs/final_corrected_zero_cash_benchmark_comparison/benchmarks"
)
DEFAULT_BIL_TD3_DIR = (
    "~/Projects/portfolio_drl_outputs/final_corrected_limited_td3_cash_bil_proxy_60ep_10seeds"
)
DEFAULT_BIL_BENCHMARK_DIR = (
    "~/Projects/portfolio_drl_outputs/final_corrected_bil_cash_benchmark_comparison/benchmarks"
)
DEFAULT_OUTPUT_DIR = "~/Projects/portfolio_drl_outputs/final_corrected_execution_spread_robustness"


@dataclass(frozen=True)
class SpreadScenario:
    name: str
    base_half_spreads_bps: dict[str, float]
    beta: float = 0.5

    @property
    def base_half_spreads_decimal(self) -> dict[str, float]:
        return {asset: bps / 10_000.0 for asset, bps in self.base_half_spreads_bps.items()}


SPREAD_SCENARIOS = [
    SpreadScenario(
        "base_no_extra_spread",
        {"SPY": 0.0, "TLT": 0.0, "GLD": 0.0, "BTC-USD": 0.0, "CASH": 0.0},
        beta=0.0,
    ),
    SpreadScenario(
        "institutional_clean_spread",
        {"SPY": 0.25, "TLT": 0.50, "GLD": 0.50, "BTC-USD": 3.0, "CASH": 0.0},
    ),
    SpreadScenario(
        "conservative_spread",
        {"SPY": 1.0, "TLT": 2.0, "GLD": 2.0, "BTC-USD": 15.0, "CASH": 0.0},
    ),
    SpreadScenario(
        "stress_spread",
        {"SPY": 3.0, "TLT": 5.0, "GLD": 5.0, "BTC-USD": 50.0, "CASH": 0.0},
    ),
]


def build_execution_spread_robustness_report(
    zero_td3_dir: str = DEFAULT_ZERO_TD3_DIR,
    zero_benchmark_dir: str = DEFAULT_ZERO_BENCHMARK_DIR,
    bil_td3_dir: str = DEFAULT_BIL_TD3_DIR,
    bil_benchmark_dir: str = DEFAULT_BIL_BENCHMARK_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    include_references: bool = True,
) -> dict[str, Any]:
    """Build execution-spread robustness outputs from existing histories."""
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "cash_assumption": "zero_cash",
            "strategy_name": "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
            "strategy_type": "TD3",
            "history": load_td3_history(
                Path(zero_td3_dir).expanduser(),
                "V3_real_macro_vintage_clean_no_dxy",
                "0p70",
            ),
        },
        {
            "cash_assumption": "bil_cash",
            "strategy_name": "V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80",
            "strategy_type": "TD3",
            "history": load_td3_history(
                Path(bil_td3_dir).expanduser(),
                "V7_real_macro_vintage_clean_no_dxy_garch",
                "0p80",
            ),
        },
    ]
    benchmark_names = ["trend_spy_cash_12p"]
    if include_references:
        benchmark_names.extend(["Equal_Weight", "BuyHold_GLD"])
    for cash_assumption, benchmark_dir in [
        ("zero_cash", Path(zero_benchmark_dir).expanduser()),
        ("bil_cash", Path(bil_benchmark_dir).expanduser()),
    ]:
        for benchmark in benchmark_names:
            specs.append(
                {
                    "cash_assumption": cash_assumption,
                    "strategy_name": benchmark,
                    "strategy_type": "benchmark",
                    "history": load_benchmark_history(benchmark_dir, benchmark),
                }
            )

    warnings_list: list[str] = []
    rows = []
    for spec in specs:
        history = spec["history"]
        for scenario in SPREAD_SCENARIOS:
            scenario_frame, scenario_warnings = apply_spread_scenario(history, scenario)
            warnings_list.extend(
                f"{spec['cash_assumption']}/{spec['strategy_name']}/{scenario.name}: {warning}"
                for warning in scenario_warnings
            )
            rows.append(
                {
                    "cash_assumption": spec["cash_assumption"],
                    "strategy_name": spec["strategy_name"],
                    "strategy_type": spec["strategy_type"],
                    "scenario": scenario.name,
                    **compute_strategy_metrics(scenario_frame),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics = add_base_deltas(metrics)
    degradation = build_degradation_summary(metrics)
    metadata = build_metadata(
        output_dir=str(output_path),
        zero_td3_dir=zero_td3_dir,
        zero_benchmark_dir=zero_benchmark_dir,
        bil_td3_dir=bil_td3_dir,
        bil_benchmark_dir=bil_benchmark_dir,
        warnings=warnings_list,
    )
    summary = build_summary_markdown(metrics, degradation, metadata)

    paths = {
        "metrics": output_path / "execution_spread_strategy_metrics.csv",
        "degradation": output_path / "execution_spread_degradation_summary.csv",
        "summary": output_path / "execution_spread_summary.md",
        "metadata": output_path / "execution_spread_metadata.json",
    }
    metrics.to_csv(paths["metrics"], index=False)
    degradation.to_csv(paths["degradation"], index=False)
    paths["summary"].write_text(summary, encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "degradation": degradation,
        "metadata": metadata,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def load_td3_history(experiment_dir: Path, base_candidate: str, cap_label: str) -> pd.DataFrame:
    """Load and date-average TD3 test histories for one candidate/cap."""
    candidate_dir = experiment_dir / "per_candidate" / base_candidate
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Missing TD3 candidate directory: {candidate_dir}")
    pattern = f"*_{base_candidate}_cap_{cap_label}_seed_*/test_policy_history.csv"
    paths = sorted(candidate_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No TD3 histories found for {base_candidate} cap {cap_label} in {candidate_dir}")
    frames = [_load_history(path) for path in paths]
    return date_average_histories(frames)


def load_benchmark_history(benchmark_dir: Path, benchmark_name: str) -> pd.DataFrame:
    path = benchmark_dir / "histories" / f"{benchmark_name}_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark history: {path}")
    return _load_history(path)


def date_average_histories(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"])
    numeric = combined.select_dtypes(include=[np.number]).columns.tolist()
    averaged = combined.groupby("date", as_index=False)[numeric].mean()
    if "transaction_cost_mode" in combined.columns:
        modes = combined.groupby("date")["transaction_cost_mode"].first().reset_index()
        averaged = averaged.merge(modes, on="date", how="left")
    return averaged.sort_values("date").reset_index(drop=True)


def apply_spread_scenario(history: pd.DataFrame, scenario: SpreadScenario) -> tuple[pd.DataFrame, list[str]]:
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    validate_history_columns(frame)
    spreads, warnings_list = build_proxy_weekly_spreads(
        frame["date"],
        ASSETS,
        scenario.base_half_spreads_decimal,
        rolling_vol=estimate_history_rolling_vol(frame),
        beta=scenario.beta,
    )
    spreads = spreads.reindex(pd.DatetimeIndex(frame["date"]))
    spread_cost = pd.Series(0.0, index=frame.index, dtype=float)
    for asset in ASSETS:
        turnover_col = f"asset_turnover_{asset}"
        if turnover_col not in frame.columns:
            warnings_list.append(f"Missing {turnover_col}; using total turnover proxy for spread cost.")
            spread_cost += pd.to_numeric(frame["turnover"], errors="coerce").fillna(0.0) * float(
                spreads[asset].fillna(0.0).mean()
            )
        else:
            spread_cost += (
                pd.to_numeric(frame[turnover_col], errors="coerce").fillna(0.0).to_numpy()
                * spreads[asset].fillna(0.0).to_numpy()
            )
    frame["spread_cost"] = spread_cost
    frame["execution_net_return"] = pd.to_numeric(frame["financial_net_return"], errors="coerce") - frame[
        "spread_cost"
    ]
    frame["total_execution_cost"] = pd.to_numeric(frame["transaction_cost"], errors="coerce").fillna(0.0) + frame[
        "spread_cost"
    ]
    return frame, warnings_list


def estimate_history_rolling_vol(history: pd.DataFrame) -> pd.Series:
    returns = pd.to_numeric(history["financial_net_return"], errors="coerce").fillna(0.0)
    return returns.rolling(12, min_periods=3).std().fillna(returns.std(ddof=0) if len(returns) else 0.0)


def compute_strategy_metrics(history: pd.DataFrame) -> dict[str, float | int]:
    returns = pd.Series(pd.to_numeric(history["execution_net_return"], errors="coerce").fillna(0.0))
    weight_cash = pd.to_numeric(history.get("weight_CASH", 0.0), errors="coerce")
    weight_btc = pd.to_numeric(history.get("weight_BTC-USD", 0.0), errors="coerce")
    return {
        "n_periods": int(len(returns)),
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(returns, PERIODS_PER_YEAR),
        "annualized_volatility": annualized_volatility(returns, PERIODS_PER_YEAR),
        "sharpe": sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR),
        "max_drawdown": max_drawdown(returns),
        "average_turnover": float(pd.to_numeric(history["turnover"], errors="coerce").mean()),
        "total_commission_cost": float(pd.to_numeric(history["transaction_cost"], errors="coerce").sum()),
        "total_spread_cost": float(pd.to_numeric(history["spread_cost"], errors="coerce").sum()),
        "total_execution_cost": float(pd.to_numeric(history["total_execution_cost"], errors="coerce").sum()),
        "mean_cash_weight": float(weight_cash.mean()),
        "mean_btc_weight": float(weight_btc.mean()),
    }


def add_base_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    base = result[result["scenario"] == "base_no_extra_spread"][
        ["cash_assumption", "strategy_name", "annualized_return", "sharpe"]
    ].rename(columns={"annualized_return": "base_annualized_return", "sharpe": "base_sharpe"})
    result = result.merge(base, on=["cash_assumption", "strategy_name"], how="left")
    result["delta_return_vs_base"] = result["annualized_return"] - result["base_annualized_return"]
    result["delta_sharpe_vs_base"] = result["sharpe"] - result["base_sharpe"]
    return result.drop(columns=["base_annualized_return", "base_sharpe"])


def build_degradation_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    non_base = metrics[metrics["scenario"] != "base_no_extra_spread"].copy()
    return non_base.sort_values(
        ["cash_assumption", "scenario", "delta_sharpe_vs_base", "delta_return_vs_base"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_metadata(
    output_dir: str,
    zero_td3_dir: str,
    zero_benchmark_dir: str,
    bil_td3_dir: str,
    bil_benchmark_dir: str,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "report_type": "execution_spread_robustness",
        "reporting_only": True,
        "retrained": False,
        "creates_new_final_winners": False,
        "output_dir": output_dir,
        "history_sources": {
            "zero_td3_dir": zero_td3_dir,
            "zero_benchmark_dir": zero_benchmark_dir,
            "bil_td3_dir": bil_td3_dir,
            "bil_benchmark_dir": bil_benchmark_dir,
        },
        "strategies": {
            "zero_cash_td3": "V3_real_macro_vintage_clean_no_dxy_cap_0p70",
            "bil_cash_td3": "V7_real_macro_vintage_clean_no_dxy_garch_cap_0p80",
            "primary_benchmark": "trend_spy_cash_12p",
            "reference_benchmarks": ["Equal_Weight", "BuyHold_GLD"],
        },
        "scenarios": [
            {
                "name": scenario.name,
                "base_half_spreads_bps": scenario.base_half_spreads_bps,
                "beta": scenario.beta,
            }
            for scenario in SPREAD_SCENARIOS
        ],
        "methodology": {
            "spread_cost_formula": "sum(abs(target_weight - drifted_weight) * asset_half_spread)",
            "history_implementation": "uses asset_turnover_* columns already stored in histories",
            "cash_spread": "always zero",
            "proxy_caveat": "Proxy spreads are scenario assumptions, not calibrated execution estimates.",
            "not_market_impact": True,
        },
        "warnings": sorted(set(warnings)),
    }


def build_summary_markdown(metrics: pd.DataFrame, degradation: pd.DataFrame, metadata: dict[str, Any]) -> str:
    lines = [
        "# Execution Spread Robustness",
        "",
        "This is a reporting-only execution-friction robustness layer. It does not retrain TD3 and does not create new final model-selection winners.",
        "",
        "The report applies top-of-book half-spread assumptions to existing realized histories using stored asset turnover columns.",
        "",
        "## Scenarios",
        "",
    ]
    for scenario in metadata["scenarios"]:
        lines.append(f"- `{scenario['name']}`: {scenario['base_half_spreads_bps']} half-spread bps")
    lines.extend(["", "## Largest Sharpe Degradation", ""])
    if degradation.empty:
        lines.append("No degradation rows were produced.")
    else:
        top = degradation.nsmallest(8, "delta_sharpe_vs_base")
        for _, row in top.iterrows():
            lines.append(
                f"- {row['cash_assumption']} / {row['strategy_name']} / {row['scenario']}: "
                f"delta Sharpe {row['delta_sharpe_vs_base']:.4f}, "
                f"delta annualized return {row['delta_return_vs_base']:.4f}, "
                f"total spread cost {row['total_spread_cost']:.4f}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Additional spread assumptions reduce realized performance in proportion to asset turnover and scenario severity.",
            "This layer strengthens execution realism without changing the final corrected model-selection claims.",
            "The proxy scenarios are not calibrated market-impact or exact broker execution models.",
        ]
    )
    if metadata["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in metadata["warnings"][:20]:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def validate_history_columns(frame: pd.DataFrame) -> None:
    required = ["date", "financial_net_return", "turnover", "transaction_cost"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"History missing required columns: {missing}")


def _load_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_history_columns(frame)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build execution spread robustness report.")
    parser.add_argument("--zero-td3-dir", default=DEFAULT_ZERO_TD3_DIR)
    parser.add_argument("--zero-benchmark-dir", default=DEFAULT_ZERO_BENCHMARK_DIR)
    parser.add_argument("--bil-td3-dir", default=DEFAULT_BIL_TD3_DIR)
    parser.add_argument("--bil-benchmark-dir", default=DEFAULT_BIL_BENCHMARK_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-references", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_execution_spread_robustness_report(
        zero_td3_dir=args.zero_td3_dir,
        zero_benchmark_dir=args.zero_benchmark_dir,
        bil_td3_dir=args.bil_td3_dir,
        bil_benchmark_dir=args.bil_benchmark_dir,
        output_dir=args.output_dir,
        include_references=not args.no_references,
    )
    print("Execution spread robustness report written:")
    for path in result["paths"].values():
        print(path)


if __name__ == "__main__":
    main()
