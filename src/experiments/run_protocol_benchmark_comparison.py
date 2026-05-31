"""Unified benchmark-only comparison under the common experimental protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.backtest.allocation_diagnostics import allocation_diagnostics
from src.backtest.dynamic_allocation_benchmarks import (
    build_defensive_risk_off_weights,
    build_momentum_winner_weights,
    build_risk_adjusted_momentum_winner_weights,
    build_rolling_markowitz_weights,
    build_rolling_risk_parity_weights,
    build_trend_following_spy_cash_weights,
    evaluate_weight_strategy,
)
from src.utils.config import load_config


STANDARD_ASSETS = ("SPY", "TLT", "GLD", "BTC-USD", "CASH")
RISKY_ASSETS = ("SPY", "TLT", "GLD", "BTC-USD")
DEFAULT_OUTPUT_DIR = "outputs/tables/protocol_benchmark_comparison"
BROKER_COST_CAVEAT = (
    "Broker/exchange-style trading-cost proxy only; does not model fiat ramps, "
    "withdrawals, custody frictions, taxes, market impact, or delays."
)


def run_protocol_benchmark_comparison(
    returns_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    transaction_cost: float = 0.001,
    transaction_cost_mode: str = "scalar",
    asset_transaction_cost_bps: dict | None = None,
    initial_value: float = 100000.0,
    date_column: str = "date",
    base_config_path: str | None = None,
) -> dict:
    """Evaluate protocol-comparable benchmarks and write summary outputs."""
    returns = load_protocol_returns(returns_path, date_column=date_column)
    benchmark_weights = build_protocol_benchmark_weights(returns)
    evaluations = evaluate_protocol_benchmark_weights(
        returns=returns,
        benchmark_weights=benchmark_weights,
        transaction_cost=transaction_cost,
        transaction_cost_mode=transaction_cost_mode,
        asset_transaction_cost_bps=asset_transaction_cost_bps,
        initial_value=initial_value,
    )

    output_path = Path(output_dir)
    histories_path = output_path / "histories"
    histories_path.mkdir(parents=True, exist_ok=True)

    metrics_table = build_benchmark_metrics_table(evaluations)
    diagnostics = build_benchmark_diagnostics(evaluations)
    comparison_summary = build_benchmark_comparison_summary(metrics_table)

    metrics_path = output_path / "benchmark_metrics_table.csv"
    summary_path = output_path / "benchmark_comparison_summary.csv"
    diagnostics_path = output_path / "benchmark_diagnostics.csv"
    metadata_path = output_path / "benchmark_metadata.json"
    metrics_table.to_csv(metrics_path, index=False)
    comparison_summary.to_csv(summary_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    metadata = build_benchmark_metadata(
        returns_path=returns_path,
        output_dir=str(output_path),
        transaction_cost=transaction_cost,
        transaction_cost_mode=transaction_cost_mode,
        asset_transaction_cost_bps=asset_transaction_cost_bps,
        initial_value=initial_value,
        date_column=date_column,
        base_config_path=base_config_path,
        benchmark_names=list(evaluations),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_paths = {}
    for benchmark_name, evaluation in evaluations.items():
        history_path = histories_path / f"{_safe_filename(benchmark_name)}_history.csv"
        evaluation["history"].to_csv(history_path, index=False)
        history_paths[benchmark_name] = str(history_path)

    return {
        "returns": returns,
        "evaluations": evaluations,
        "metrics_table": metrics_table,
        "comparison_summary": comparison_summary,
        "diagnostics": diagnostics,
        "paths": {
            "output_dir": str(output_path),
            "metrics_table": str(metrics_path),
            "comparison_summary": str(summary_path),
            "diagnostics": str(diagnostics_path),
            "metadata": str(metadata_path),
            "histories_dir": str(histories_path),
            "histories": history_paths,
        },
    }


def load_protocol_returns(
    returns_path: str,
    date_column: str = "date",
) -> pd.DataFrame:
    """Load weekly returns and align them to the standard protocol asset set."""
    path = Path(returns_path)
    if not path.exists():
        raise FileNotFoundError(f"returns_path does not exist: {returns_path}")

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("returns file must not be empty.")

    resolved_date_column = date_column if date_column in raw.columns else raw.columns[0]
    raw[resolved_date_column] = pd.to_datetime(raw[resolved_date_column])
    raw = raw.set_index(resolved_date_column).sort_index()

    missing_risky_assets = [asset for asset in RISKY_ASSETS if asset not in raw.columns]
    if missing_risky_assets:
        raise ValueError(f"returns file is missing required assets: {missing_risky_assets}")
    if "CASH" not in raw.columns:
        raw["CASH"] = 0.0

    returns = raw.loc[:, [asset for asset in STANDARD_ASSETS if asset in raw.columns]]
    returns = returns.astype(float)
    if returns.isna().any().any():
        raise ValueError("returns file contains missing values after alignment.")

    return returns


def build_protocol_benchmark_weights(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all benchmark weight matrices used by the common protocol."""
    _validate_protocol_returns(returns)
    return {
        "BuyHold_SPY": _constant_single_asset_weights(returns, "SPY"),
        "BuyHold_TLT": _constant_single_asset_weights(returns, "TLT"),
        "BuyHold_GLD": _constant_single_asset_weights(returns, "GLD"),
        "BuyHold_BTC-USD": _constant_single_asset_weights(returns, "BTC-USD"),
        "Equal_Weight": _constant_equal_weight(returns, list(returns.columns)),
        "Equal_Weight_Risky": _constant_equal_weight(returns, list(RISKY_ASSETS)),
        "60_40_SPY_TLT": _constant_allocation_weights(
            returns,
            {"SPY": 0.60, "TLT": 0.40},
        ),
        "momentum_winner_12p": build_momentum_winner_weights(
            returns,
            window=12,
        ),
        "risk_adjusted_momentum_winner_12p_12p": (
            build_risk_adjusted_momentum_winner_weights(
                returns,
                momentum_window=12,
                volatility_window=12,
            )
        ),
        "trend_spy_cash_12p": build_trend_following_spy_cash_weights(
            returns,
            window=12,
        ),
        "defensive_risk_off_12p": build_defensive_risk_off_weights(
            returns,
            window=12,
        ),
        "rolling_risk_parity_inverse_vol_12p": build_rolling_risk_parity_weights(
            returns,
            window=12,
            include_cash=False,
        ),
        "rolling_markowitz_long_only_52p": build_rolling_markowitz_weights(
            returns,
            window=52,
            include_cash=False,
            use_mean_returns=True,
        ),
        "rolling_markowitz_min_variance_52p": build_rolling_markowitz_weights(
            returns,
            window=52,
            include_cash=False,
            use_mean_returns=False,
        ),
    }


def evaluate_protocol_benchmark_weights(
    returns: pd.DataFrame,
    benchmark_weights: dict[str, pd.DataFrame],
    transaction_cost: float = 0.001,
    transaction_cost_mode: str = "scalar",
    asset_transaction_cost_bps: dict | None = None,
    initial_value: float = 100000.0,
) -> dict[str, dict]:
    """Evaluate benchmark weights through the shared protocol evaluator."""
    evaluations = {}
    for benchmark_name, weights in benchmark_weights.items():
        evaluations[benchmark_name] = evaluate_weight_strategy(
            returns=returns,
            weights=weights,
            transaction_cost=transaction_cost,
            transaction_cost_mode=transaction_cost_mode,
            asset_transaction_cost_bps=asset_transaction_cost_bps,
            initial_value=initial_value,
        )
    return evaluations


def build_benchmark_metrics_table(evaluations: dict[str, dict]) -> pd.DataFrame:
    """Create the main metrics table from evaluated benchmarks."""
    rows = []
    for benchmark_name, evaluation in evaluations.items():
        history = evaluation["history"]
        weights = _extract_weights(history)
        diagnostics = allocation_diagnostics(
            weights,
            turnover=history["turnover"],
            transaction_costs=history["transaction_cost"],
        )
        row = {
            "benchmark_name": benchmark_name,
            "cumulative_return": evaluation["cumulative_return"],
            "annualized_return": evaluation["annualized_return"],
            "annualized_volatility": evaluation["annualized_volatility"],
            "sharpe": evaluation["sharpe_ratio"],
            "sortino": evaluation["sortino_ratio"],
            "calmar": evaluation["calmar_ratio"],
            "max_drawdown": evaluation["max_drawdown"],
            "average_turnover": evaluation["average_turnover"],
            "total_transaction_cost": float(history["transaction_cost"].sum()),
            "average_transaction_cost": float(history["transaction_cost"].mean()),
            "average_max_weight": diagnostics["average_max_weight"],
            "average_effective_number_of_assets": diagnostics[
                "average_effective_number_of_assets"
            ],
        }
        if "weight_CASH" in history.columns:
            cash_weight = history["weight_CASH"]
            row["mean_cash_weight"] = float(cash_weight.mean())
            row["cash_above_10pct"] = float((cash_weight > 0.10).mean())
        rows.append(row)

    return pd.DataFrame(rows)


def build_benchmark_diagnostics(evaluations: dict[str, dict]) -> pd.DataFrame:
    """Create allocation and cost diagnostics for each benchmark."""
    rows = []
    for benchmark_name, evaluation in evaluations.items():
        history = evaluation["history"]
        weights = _extract_weights(history)
        diagnostics = allocation_diagnostics(
            weights,
            turnover=history["turnover"],
            transaction_costs=history["transaction_cost"],
        )
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "total_transaction_cost": float(history["transaction_cost"].sum()),
                "average_transaction_cost": float(history["transaction_cost"].mean()),
                "transaction_cost_mode": (
                    str(history["transaction_cost_mode"].dropna().iloc[0])
                    if "transaction_cost_mode" in history
                    and not history["transaction_cost_mode"].dropna().empty
                    else "scalar"
                ),
                "cash_above_10pct": (
                    float((history["weight_CASH"] > 0.10).mean())
                    if "weight_CASH" in history.columns
                    else 0.0
                ),
                **diagnostics,
            }
        )

    return pd.DataFrame(rows)


def build_benchmark_comparison_summary(metrics_table: pd.DataFrame) -> pd.DataFrame:
    """Create a concise sorted benchmark comparison table."""
    summary = metrics_table.copy()
    return summary.sort_values(
        ["sharpe", "cumulative_return"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _constant_single_asset_weights(
    returns: pd.DataFrame,
    asset: str,
) -> pd.DataFrame:
    return _constant_allocation_weights(returns, {asset: 1.0})


def _constant_equal_weight(
    returns: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    _validate_available_assets(returns, assets)
    return _constant_allocation_weights(
        returns,
        {asset: 1.0 / len(assets) for asset in assets},
    )


def _constant_allocation_weights(
    returns: pd.DataFrame,
    allocation: dict[str, float],
) -> pd.DataFrame:
    _validate_available_assets(returns, list(allocation))
    if any(weight < 0.0 for weight in allocation.values()):
        raise ValueError("allocation weights must be non-negative.")
    total_weight = sum(allocation.values())
    if abs(total_weight - 1.0) > 1e-12:
        raise ValueError("allocation weights must sum to 1.")

    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for asset, weight in allocation.items():
        weights[asset] = float(weight)
    return weights


def _extract_weights(history: pd.DataFrame) -> pd.DataFrame:
    weight_columns = [column for column in history.columns if column.startswith("weight_")]
    weights = history[weight_columns].copy()
    weights.columns = [column.replace("weight_", "", 1) for column in weight_columns]
    return weights


def _validate_protocol_returns(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex.")
    _validate_available_assets(returns, list(RISKY_ASSETS))
    if "CASH" not in returns.columns:
        raise ValueError("protocol returns must include CASH; load_protocol_returns adds it.")


def _validate_available_assets(returns: pd.DataFrame, assets: list[str]) -> None:
    missing_assets = [asset for asset in assets if asset not in returns.columns]
    if missing_assets:
        raise ValueError(f"returns are missing required assets: {missing_assets}")


def _safe_filename(value: str) -> str:
    return value.replace("/", "_")


def build_benchmark_metadata(
    returns_path: str,
    output_dir: str,
    transaction_cost: float,
    transaction_cost_mode: str,
    asset_transaction_cost_bps: dict | None,
    initial_value: float,
    date_column: str,
    base_config_path: str | None,
    benchmark_names: list[str],
) -> dict:
    """Build benchmark reproducibility metadata."""
    return {
        "runner": "src.experiments.run_protocol_benchmark_comparison",
        "returns_path": returns_path,
        "output_dir": output_dir,
        "transaction_cost": transaction_cost,
        "transaction_cost_mode": transaction_cost_mode,
        "asset_transaction_cost_bps": asset_transaction_cost_bps,
        "initial_value": initial_value,
        "date_column": date_column,
        "base_config_path": base_config_path,
        "benchmark_names": benchmark_names,
        "cost_caveat": BROKER_COST_CAVEAT,
        "reporting_only_note": (
            "Deterministic benchmarks are regenerated from returns and benchmark "
            "weight rules. No TD3 retraining is performed."
        ),
    }


def parse_asset_transaction_cost_bps(value: str | None) -> dict | None:
    """Parse CLI mapping like SPY:2.0,TLT:2.0."""
    if value is None:
        return None
    result = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError("asset transaction cost entries must use ASSET:BPS format.")
        asset, bps = item.split(":", 1)
        asset = asset.strip()
        if not asset:
            raise ValueError("asset transaction cost asset name must be non-empty.")
        result[asset] = float(bps)
    return result


def resolve_transaction_cost_settings(
    base_config_path: str | None,
    transaction_cost: float | None,
    transaction_cost_mode: str | None,
    asset_transaction_cost_bps: dict | None,
) -> dict:
    """Resolve benchmark transaction-cost settings from config plus CLI overrides."""
    resolved = {
        "transaction_cost": 0.001,
        "transaction_cost_mode": "scalar",
        "asset_transaction_cost_bps": None,
    }
    if base_config_path:
        config = load_config(base_config_path)
        environment = config["environment"]
        resolved["transaction_cost"] = environment.get("transaction_cost", 0.001)
        resolved["transaction_cost_mode"] = environment.get("transaction_cost_mode", "scalar")
        resolved["asset_transaction_cost_bps"] = environment.get("asset_transaction_cost_bps")
    if transaction_cost is not None:
        resolved["transaction_cost"] = transaction_cost
    if transaction_cost_mode is not None:
        resolved["transaction_cost_mode"] = transaction_cost_mode
    if asset_transaction_cost_bps is not None:
        resolved["asset_transaction_cost_bps"] = asset_transaction_cost_bps
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run protocol-comparable benchmark-only evaluation.",
    )
    parser.add_argument("--returns-path", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--transaction-cost", type=float, default=None)
    parser.add_argument("--transaction-cost-mode", choices=["scalar", "asset_specific"])
    parser.add_argument("--asset-transaction-cost-bps")
    parser.add_argument("--base-config-path")
    parser.add_argument("--initial-value", type=float, default=100000.0)
    parser.add_argument("--date-column", default="date")
    args = parser.parse_args()
    cost_settings = resolve_transaction_cost_settings(
        base_config_path=args.base_config_path,
        transaction_cost=args.transaction_cost,
        transaction_cost_mode=args.transaction_cost_mode,
        asset_transaction_cost_bps=parse_asset_transaction_cost_bps(
            args.asset_transaction_cost_bps,
        ),
    )

    result = run_protocol_benchmark_comparison(
        returns_path=args.returns_path,
        output_dir=args.output_dir,
        transaction_cost=cost_settings["transaction_cost"],
        transaction_cost_mode=cost_settings["transaction_cost_mode"],
        asset_transaction_cost_bps=cost_settings["asset_transaction_cost_bps"],
        initial_value=args.initial_value,
        date_column=args.date_column,
        base_config_path=args.base_config_path,
    )
    print(result["comparison_summary"].to_string(index=False))
    print(f"Outputs written to {result['paths']['output_dir']}")


if __name__ == "__main__":
    main()
