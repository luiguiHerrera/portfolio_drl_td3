"""TD3 agent evaluation utilities.

This module evaluates an already-created policy-like agent on aligned returns
and feature observations. It does not train agents, save files, or generate
plots.
"""

import pandas as pd

from src.backtest.evaluate_policy import summary_metrics
from src.env.portfolio_env import PortfolioEnv


def run_policy_episode(
    agent,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    initial_cash: float = 100000.0,
    transaction_cost: float = 0.001,
) -> dict:
    """Run one policy episode and collect realized portfolio diagnostics."""
    env = PortfolioEnv(
        returns=returns,
        features=features,
        initial_cash=initial_cash,
        transaction_cost=transaction_cost,
    )
    state = env.reset()
    done = False
    rewards = []
    policy_returns = []
    portfolio_values = []
    turnover = []
    transaction_costs = []
    weights = []

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        rewards.append(reward)
        policy_returns.append(info["portfolio_return"])
        portfolio_values.append(info["portfolio_value"])
        turnover.append(info["turnover"])
        transaction_costs.append(info["transaction_cost"])
        weights.append(info["weights"])
        state = next_state

    episode_index = env.returns.index

    return {
        "policy_returns": pd.Series(policy_returns, index=episode_index, name="policy_returns"),
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
) -> dict:
    """Evaluate an agent episode and compute return-based summary metrics."""
    episode = run_policy_episode(
        agent=agent,
        returns=returns,
        features=features,
        initial_cash=initial_cash,
        transaction_cost=transaction_cost,
    )

    return {
        "episode": episode,
        "metrics": summary_metrics(
            episode["policy_returns"],
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
    }
