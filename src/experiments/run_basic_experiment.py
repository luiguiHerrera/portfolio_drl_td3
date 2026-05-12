"""Minimal in-memory experiment runner.

This module organizes the output of the minimal TD3 training pipeline into a
compact experiment result. It does not save artifacts, print reports, or create
plots.
"""

import pandas as pd

from src.train.train_td3 import train_td3


REQUIRED_METRIC_COLUMNS = {
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
}
REQUIRED_POLICY_ROWS = {
    "agent",
    "equal_weight_gross",
    "equal_weight_rebalanced_net",
    "buy_and_hold",
}


def summarize_metrics_table(metrics_table: pd.DataFrame) -> dict:
    """Summarize an in-memory policy comparison metrics table."""
    missing_columns = REQUIRED_METRIC_COLUMNS.difference(metrics_table.columns)
    if missing_columns:
        raise KeyError(f"metrics_table is missing required columns: {sorted(missing_columns)}")

    missing_rows = REQUIRED_POLICY_ROWS.difference(metrics_table.index)
    if missing_rows:
        raise KeyError(f"metrics_table is missing required rows: {sorted(missing_rows)}")

    sharpe_ranking = metrics_table["sharpe_ratio"].sort_values(ascending=False)
    best_policy_by_sharpe = sharpe_ranking.index[0]
    agent_rank_by_sharpe = int(sharpe_ranking.index.get_loc("agent") + 1)
    agent_sharpe_ratio = float(metrics_table.loc["agent", "sharpe_ratio"])
    agent_cumulative_return = float(metrics_table.loc["agent", "cumulative_return"])
    individual_buyhold_rows = [
        policy_name for policy_name in metrics_table.index if policy_name.startswith("buy_hold_")
    ]
    if not individual_buyhold_rows:
        raise KeyError("metrics_table must include individual buy-hold rows.")

    individual_buyhold_metrics = metrics_table.loc[individual_buyhold_rows]
    individual_buyhold_sharpe_ranking = individual_buyhold_metrics["sharpe_ratio"].sort_values(
        ascending=False
    )
    best_individual_buyhold_by_sharpe = individual_buyhold_sharpe_ranking.index[0]
    best_individual_buyhold_sharpe_ratio = float(individual_buyhold_sharpe_ranking.iloc[0])
    best_individual_buyhold_cumulative_return = float(
        metrics_table.loc[best_individual_buyhold_by_sharpe, "cumulative_return"]
    )

    return {
        "best_policy_by_sharpe": best_policy_by_sharpe,
        "best_sharpe_ratio": float(sharpe_ranking.iloc[0]),
        "agent_rank_by_sharpe": agent_rank_by_sharpe,
        "agent_sharpe_ratio": agent_sharpe_ratio,
        "agent_cumulative_return": agent_cumulative_return,
        "agent_max_drawdown": float(metrics_table.loc["agent", "max_drawdown"]),
        "agent_vs_equal_weight_rebalanced_net_sharpe_diff": agent_sharpe_ratio
        - float(metrics_table.loc["equal_weight_rebalanced_net", "sharpe_ratio"]),
        "agent_vs_buy_and_hold_sharpe_diff": agent_sharpe_ratio
        - float(metrics_table.loc["buy_and_hold", "sharpe_ratio"]),
        "best_individual_buyhold_by_sharpe": best_individual_buyhold_by_sharpe,
        "best_individual_buyhold_sharpe_ratio": best_individual_buyhold_sharpe_ratio,
        "best_individual_buyhold_cumulative_return": best_individual_buyhold_cumulative_return,
        "agent_vs_best_individual_buyhold_sharpe_diff": agent_sharpe_ratio
        - best_individual_buyhold_sharpe_ratio,
        "agent_vs_equal_weight_rebalanced_net_cumulative_return_diff": agent_cumulative_return
        - float(metrics_table.loc["equal_weight_rebalanced_net", "cumulative_return"]),
        "agent_vs_buy_and_hold_cumulative_return_diff": agent_cumulative_return
        - float(metrics_table.loc["buy_and_hold", "cumulative_return"]),
        "agent_vs_best_individual_buyhold_cumulative_return_diff": agent_cumulative_return
        - best_individual_buyhold_cumulative_return,
    }


def run_basic_experiment(config_path: str) -> dict:
    """Run the basic TD3 pipeline and return organized in-memory results."""
    result = train_td3(config_path)
    episode_logs = result["episode_logs"]
    final_episode = episode_logs[-1]
    training_summary = {
        "total_episodes": len(episode_logs),
        "final_episode": final_episode["episode"],
        "final_portfolio_value": final_episode["final_portfolio_value"],
        "final_total_reward": final_episode["total_reward"],
        "final_average_turnover": final_episode["average_turnover"],
        "final_average_transaction_cost": final_episode["average_transaction_cost"],
        "final_max_weight": final_episode["max_weight"],
        "final_cash_weight": final_episode["cash_weight"],
    }
    validation_metrics_table = result["validation_comparison"]["metrics_table"]
    test_metrics_table = result["test_comparison"]["metrics_table"]

    return {
        "training_summary": training_summary,
        "validation_metrics_table": validation_metrics_table,
        "test_metrics_table": test_metrics_table,
        "validation_comparison_summary": summarize_metrics_table(validation_metrics_table),
        "test_comparison_summary": summarize_metrics_table(test_metrics_table),
        "validation_diagnostics": result["validation_evaluation"]["diagnostics"],
        "test_diagnostics": result["test_evaluation"]["diagnostics"],
        "raw_result": result,
    }
