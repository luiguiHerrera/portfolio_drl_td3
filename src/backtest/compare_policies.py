"""Policy comparison utilities for TD3 and basic benchmarks.

This module compares an already-created agent against simple benchmark return
series in memory. The equal-weight gross benchmark is a simple gross-return
reference, the equal-weight rebalanced net benchmark includes drift-based
rebalancing transaction costs, and buy-and-hold remains a gross-return
reference for now. This module does not save files, produce plots, or implement
advanced benchmarks such as Markowitz or risk parity.
"""

import pandas as pd

from src.backtest.benchmarks import (
    buy_and_hold_returns,
    equal_weight_rebalanced_benchmark,
    equal_weight_returns,
    individual_buy_and_hold_returns,
)
from src.backtest.evaluate_agent import evaluate_agent
from src.backtest.evaluate_policy import summary_metrics
from src.backtest.performance_metrics import extended_summary_metrics


def compare_agent_to_basic_benchmarks(
    agent,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    periods_per_year: int = 52,
    risk_free_rate: float = 0.0,
    initial_cash: float = 100000.0,
    transaction_cost: float = 0.001,
    reward_config: dict | None = None,
) -> dict:
    """Compare an agent policy against basic in-memory portfolio benchmarks."""
    agent_evaluation = evaluate_agent(
        agent=agent,
        returns=returns,
        features=features,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        initial_cash=initial_cash,
        transaction_cost=transaction_cost,
        reward_config=reward_config,
    )
    agent_returns = agent_evaluation["episode"]["financial_net_returns"]
    aligned_returns = returns.loc[agent_returns.index]
    equal_weight_gross_series = equal_weight_returns(aligned_returns)
    equal_weight_rebalanced = equal_weight_rebalanced_benchmark(
        aligned_returns,
        transaction_cost=transaction_cost,
    )
    equal_weight_rebalanced_net_series = equal_weight_rebalanced["net_returns"]
    buy_and_hold_series = buy_and_hold_returns(aligned_returns)
    individual_buy_hold_series = individual_buy_and_hold_returns(aligned_returns)
    market_returns = aligned_returns["SPY"] if "SPY" in aligned_returns.columns else None

    policy_return_series = {
        "agent": agent_returns,
        "equal_weight_gross": equal_weight_gross_series,
        "equal_weight_rebalanced_net": equal_weight_rebalanced_net_series,
        "buy_and_hold": buy_and_hold_series,
        **individual_buy_hold_series,
    }
    policy_metrics = {
        policy_name: _policy_summary_metrics(
            policy_returns=policy_returns,
            benchmark_returns=equal_weight_rebalanced_net_series,
            market_returns=market_returns,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )
        for policy_name, policy_returns in policy_return_series.items()
    }
    metrics_table = pd.DataFrame(policy_metrics).T

    individual_buy_hold_benchmarks = {
        policy_name: {
            "returns": policy_returns,
            "metrics": policy_metrics[policy_name],
        }
        for policy_name, policy_returns in individual_buy_hold_series.items()
    }

    return {
        "agent": agent_evaluation,
        "benchmarks": {
            "equal_weight_gross": {
                "returns": equal_weight_gross_series,
                "metrics": policy_metrics["equal_weight_gross"],
            },
            "equal_weight_rebalanced_net": {
                "returns": equal_weight_rebalanced_net_series,
                "metrics": policy_metrics["equal_weight_rebalanced_net"],
                "diagnostics": {
                    "turnover": equal_weight_rebalanced["turnover"],
                    "transaction_costs": equal_weight_rebalanced["transaction_costs"],
                    "weights": equal_weight_rebalanced["weights"],
                },
            },
            "buy_and_hold": {
                "returns": buy_and_hold_series,
                "metrics": policy_metrics["buy_and_hold"],
            },
            **individual_buy_hold_benchmarks,
        },
        "metrics_table": metrics_table,
    }


def _policy_summary_metrics(
    policy_returns: pd.Series,
    benchmark_returns: pd.Series,
    market_returns: pd.Series | None,
    periods_per_year: int,
    risk_free_rate: float,
) -> dict:
    base_metrics = summary_metrics(
        policy_returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    extended_metrics = extended_summary_metrics(
        policy_returns,
        benchmark_returns=benchmark_returns,
        market_returns=market_returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )

    renamed_extended_metrics = {
        "sortino_ratio": extended_metrics["sortino_ratio"],
        "calmar_ratio": extended_metrics["calmar_ratio"],
        "tracking_error_vs_equal_weight_rebalanced_net": extended_metrics["tracking_error"],
        "information_ratio_vs_equal_weight_rebalanced_net": extended_metrics[
            "information_ratio"
        ],
    }
    if "capm_beta" in extended_metrics:
        renamed_extended_metrics["capm_beta_vs_SPY"] = extended_metrics["capm_beta"]
        renamed_extended_metrics["capm_alpha_vs_SPY"] = extended_metrics["capm_alpha"]

    return {
        **base_metrics,
        **renamed_extended_metrics,
    }
