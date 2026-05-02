"""Initial reward functions for portfolio allocation.

This module starts with a minimal net-return reward baseline. More advanced
components such as dynamic Sharpe, drawdown penalties, transaction diagnostics,
and turnover penalties should be added only after the environment behavior is
validated.
"""


def compute_net_return_reward(portfolio_return: float, transaction_cost: float) -> float:
    """Return portfolio return net of realized transaction cost."""
    return portfolio_return - transaction_cost
