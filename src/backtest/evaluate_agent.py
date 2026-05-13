"""TD3 agent evaluation utilities.

This module evaluates an already-created policy-like agent on aligned returns
and feature observations. It does not train agents, save files, or generate
plots.
"""

import pandas as pd

from src.backtest.allocation_diagnostics import allocation_diagnostics
from src.backtest.evaluate_policy import summary_metrics
from src.env.portfolio_env import PortfolioEnv


def run_policy_episode(
    agent,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    initial_cash: float = 100000.0,
    transaction_cost: float = 0.001,
    reward_config: dict | None = None,
) -> dict:
    """Run one policy episode and collect realized portfolio diagnostics."""
    env = PortfolioEnv(
        returns=returns,
        features=features,
        initial_cash=initial_cash,
        transaction_cost=transaction_cost,
        reward_config=reward_config,
    )
    state = env.reset()
    done = False
    rewards = []
    policy_returns = []
    financial_net_returns = []
    portfolio_values = []
    turnover = []
    transaction_costs = []
    drawdown = []
    concentration = []
    weights = []

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        rewards.append(reward)
        policy_returns.append(info["portfolio_return"])
        financial_net_returns.append(info["financial_net_return"])
        portfolio_values.append(info["portfolio_value"])
        turnover.append(info["turnover"])
        transaction_costs.append(info["transaction_cost"])
        drawdown.append(info["drawdown"])
        concentration.append(info["concentration"])
        weights.append(info["weights"])
        state = next_state

    episode_index = env.returns.index

    return {
        "policy_returns": pd.Series(policy_returns, index=episode_index, name="policy_returns"),
        "financial_net_returns": pd.Series(
            financial_net_returns,
            index=episode_index,
            name="financial_net_returns",
        ),
        "rewards": pd.Series(rewards, index=episode_index, name="rewards"),
        "portfolio_values": pd.Series(
            portfolio_values,
            index=episode_index,
            name="portfolio_values",
        ),
        "turnover": pd.Series(turnover, index=episode_index, name="turnover"),
        "transaction_costs": pd.Series(
            transaction_costs,
            index=episode_index,
            name="transaction_costs",
        ),
        "drawdown": pd.Series(drawdown, index=episode_index, name="drawdown"),
        "concentration": pd.Series(
            concentration,
            index=episode_index,
            name="concentration",
        ),
        "weights": pd.DataFrame(weights, index=episode_index, columns=env.asset_names),
        "final_portfolio_value": portfolio_values[-1],
    }


def evaluate_agent(
    agent,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    periods_per_year: int = 52,
    risk_free_rate: float = 0.0,
    initial_cash: float = 100000.0,
    transaction_cost: float = 0.001,
    reward_config: dict | None = None,
) -> dict:
    """Evaluate an agent episode and compute return-based summary metrics."""
    episode = run_policy_episode(
        agent=agent,
        returns=returns,
        features=features,
        initial_cash=initial_cash,
        transaction_cost=transaction_cost,
        reward_config=reward_config,
    )

    return {
        "episode": episode,
        "metrics": summary_metrics(
            episode["financial_net_returns"],
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
        "diagnostics": summarize_episode_diagnostics(episode),
        "policy_history": build_policy_history(episode),
    }


def build_policy_history(episode: dict) -> pd.DataFrame:
    """Build per-period portfolio behavior history from an evaluated episode."""
    history = pd.DataFrame(
        {
            "date": episode["financial_net_returns"].index,
            "portfolio_return": episode["policy_returns"].to_numpy(),
            "financial_net_return": episode["financial_net_returns"].to_numpy(),
            "portfolio_value": episode["portfolio_values"].to_numpy(),
            "drawdown": episode["drawdown"].to_numpy(),
            "turnover": episode["turnover"].to_numpy(),
            "transaction_cost": episode["transaction_costs"].to_numpy(),
        },
        index=episode["financial_net_returns"].index,
    )
    weights = episode["weights"].add_prefix("weight_")
    history["max_weight"] = weights.max(axis=1).to_numpy()
    history["cash_weight"] = (
        weights["weight_CASH"].to_numpy() if "weight_CASH" in weights else 0.0
    )

    return pd.concat([history, weights], axis=1)


def summarize_episode_diagnostics(episode: dict) -> dict:
    """Summarize episode-level allocation and transaction diagnostics."""
    final_weights = {
        asset_name: float(weight)
        for asset_name, weight in episode["weights"].iloc[-1].items()
    }
    allocation_summary = allocation_diagnostics(
        episode["weights"],
        turnover=episode["turnover"],
        transaction_costs=episode["transaction_costs"],
    )

    return {
        "final_portfolio_value": float(episode["final_portfolio_value"]),
        **allocation_summary,
        "final_weights": final_weights,
        "max_weight": allocation_summary["final_max_weight"],
        "cash_weight": allocation_summary["final_cash_weight"],
    }
