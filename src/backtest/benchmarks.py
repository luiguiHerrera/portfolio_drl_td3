"""Benchmark policy scaffold for portfolio backtesting.

This module will contain simple reference policies such as equal-weight,
buy-and-hold, and possibly cash-aware baselines. These benchmarks should be
implemented before evaluating TD3 so the learned policy has defensible
comparators.
"""


def equal_weight_policy(*args, **kwargs):
    """Return equal portfolio weights for the configured asset universe."""
    raise NotImplementedError("Equal-weight benchmark has not been implemented yet.")


def buy_and_hold_policy(*args, **kwargs):
    """Return buy-and-hold allocations for a backtest period."""
    raise NotImplementedError("Buy-and-hold benchmark has not been implemented yet.")
