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
)
from src.backtest.evaluate_agent import evaluate_agent
from src.backtest.evaluate_policy import summary_metrics


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

    agent_metrics = agent_evaluation["metrics"]
    equal_weight_gross_metrics = summary_metrics(
        equal_weight_gross_series,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    equal_weight_rebalanced_net_metrics = summary_metrics(
        equal_weight_rebalanced_net_series,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    buy_and_hold_metrics = summary_metrics(
        buy_and_hold_series,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    metrics_table = pd.DataFrame(
        {
            "agent": agent_metrics,
            "equal_weight_gross": equal_weight_gross_metrics,
            "equal_weight_rebalanced_net": equal_weight_rebalanced_net_metrics,
            "buy_and_hold": buy_and_hold_metrics,
        }
    ).T

    return {
        "agent": agent_evaluation,
        "benchmarks": {
            "equal_weight_gross": {
                "returns": equal_weight_gross_series,
                "metrics": equal_weight_gross_metrics,
            },
            "equal_weight_rebalanced_net": {
                "returns": equal_weight_rebalanced_net_series,
                "metrics": equal_weight_rebalanced_net_metrics,
                "diagnostics": {
                    "turnover": equal_weight_rebalanced["turnover"],
                    "transaction_costs": equal_weight_rebalanced["transaction_costs"],
                    "weights": equal_weight_rebalanced["weights"],
                },
            },
            "buy_and_hold": {
                "returns": buy_and_hold_series,
                "metrics": buy_and_hold_metrics,
            },
        },
        "metrics_table": metrics_table,
    }
