"""Minimal portfolio allocation environment.

This module defines a lightweight environment for validating portfolio mechanics
before adding Gymnasium integration or TD3 training. The environment consumes a
precomputed returns DataFrame and enforces long-only, fully invested portfolio
weights through clipping and normalization.
"""

import numpy as np
import pandas as pd

from src.rewards.reward import compute_net_return_reward


class PortfolioEnv:
    """Minimal long-only, fully invested portfolio environment."""

    def __init__(
        self,
        returns: pd.DataFrame,
        initial_cash: float = 100000.0,
        transaction_cost: float = 0.001,
    ):
        if returns.empty:
            raise ValueError("returns must not be empty.")
        if transaction_cost < 0:
            raise ValueError("transaction_cost must be non-negative.")

        self.returns = returns
        self.n_assets = len(returns.columns)
        self.asset_names = list(returns.columns)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost

        self.current_step = 0
        self.portfolio_value = initial_cash
        self.previous_weights = self._equal_weights()

    def reset(self) -> np.ndarray:
        """Reset environment state and return the initial observation."""
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.previous_weights = self._equal_weights()

        return self._get_observation()

    def step(self, action: np.ndarray):
        """Advance one period using the current weights, then rebalance."""
        weights = self._normalize_action(action)
        period_returns = self.returns.iloc[self.current_step].to_numpy(dtype=float)

        turnover = float(np.sum(np.abs(weights - self.previous_weights)))
        realized_transaction_cost = float(self.transaction_cost * turnover)
        portfolio_return = float(np.dot(self.previous_weights, period_returns))
        reward = compute_net_return_reward(portfolio_return, realized_transaction_cost)

        self.portfolio_value *= 1.0 + reward
        self.previous_weights = weights
        self.current_step += 1

        done = self.current_step >= len(self.returns)
        observation = np.zeros(self.n_assets, dtype=float) if done else self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "transaction_cost": realized_transaction_cost,
            "turnover": turnover,
            "weights": weights,
        }

        return observation, reward, done, info

    def _get_observation(self) -> np.ndarray:
        return self.returns.iloc[self.current_step].to_numpy(dtype=float)

    def _equal_weights(self) -> np.ndarray:
        return np.full(self.n_assets, 1.0 / self.n_assets, dtype=float)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (self.n_assets,):
            raise ValueError(f"action must have shape ({self.n_assets},).")

        clipped_action = np.clip(action_array, a_min=0.0, a_max=None)
        action_sum = float(clipped_action.sum())

        if action_sum == 0.0:
            return self._equal_weights()

        return clipped_action / action_sum
