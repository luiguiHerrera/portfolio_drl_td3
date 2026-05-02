"""Policy evaluation scaffold for portfolio backtests.

This module will run trained or deterministic policies through the portfolio
environment and collect equity curves, weights, turnover, drawdowns, and reward
components. Evaluation logic should wait until the environment API is stable.
"""


def evaluate_policy(*args, **kwargs):
    """Evaluate a policy on a fixed dataset or environment split."""
    raise NotImplementedError("Policy evaluation has not been implemented yet.")
