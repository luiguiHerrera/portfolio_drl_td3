"""Minimal portfolio allocation environment.

This module defines a lightweight environment for validating portfolio mechanics
before adding Gymnasium integration or TD3 training. The environment consumes a
precomputed returns DataFrame for realized portfolio returns and an optional
features DataFrame for agent observations. It enforces long-only, fully invested
portfolio weights through clipping and normalization.
"""

import numpy as np
import pandas as pd

from src.rewards.reward import (
    compute_risk_aware_reward,
    concentration_penalty,
    drawdown_penalty,
)


DEFAULT_REWARD_CONFIG = {
    "lambda_return": 1.0,
    "lambda_transaction_cost": 1.0,
    "lambda_turnover": 0.0,
    "lambda_concentration": 0.0,
    "lambda_drawdown": 0.0,
}


class PortfolioEnv:
    """Minimal long-only, fully invested portfolio environment."""

    def __init__(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame | None = None,
        initial_cash: float = 100000.0,
        transaction_cost: float = 0.001,
        reward_config: dict | None = None,
    ):
        if returns.empty:
            raise ValueError("returns must not be empty.")
        if transaction_cost < 0:
            raise ValueError("transaction_cost must be non-negative.")

        self.returns, self.features = self._align_returns_and_features(returns, features)
        self.n_assets = len(self.returns.columns)
        self.observation_dim = len(self.features.columns)
        self.asset_names = list(self.returns.columns)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.reward_config = (
            DEFAULT_REWARD_CONFIG.copy() if reward_config is None else reward_config.copy()
        )

        self.current_step = 0
        self.portfolio_value = initial_cash
        self.peak_portfolio_value = initial_cash
        self.previous_weights = self._equal_weights()

    def reset(self) -> np.ndarray:
        """Reset environment state and return the initial observation."""
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.previous_weights = self._equal_weights()

        return self._get_observation()

    def step(self, action: np.ndarray):
        """Advance one period using weights selected for this period."""
        if self.current_step >= len(self.returns):
            raise RuntimeError("Cannot call step() after the environment is done. Call reset().")

        weights = self._normalize_action(action)
        period_returns = self.returns.iloc[self.current_step].to_numpy(dtype=float)

        turnover = float(np.sum(np.abs(weights - self.previous_weights)))
        realized_transaction_cost = float(self.transaction_cost * turnover)
        portfolio_return = float(np.dot(weights, period_returns))
        financial_net_return = portfolio_return - realized_transaction_cost
        new_portfolio_value = self.portfolio_value * (1.0 + financial_net_return)
        updated_peak_portfolio_value = max(self.peak_portfolio_value, new_portfolio_value)
        drawdown = drawdown_penalty(new_portfolio_value, updated_peak_portfolio_value)
        concentration = concentration_penalty(weights)
        reward = compute_risk_aware_reward(
            portfolio_return=portfolio_return,
            transaction_cost=realized_transaction_cost,
            turnover=turnover,
            weights=weights,
            portfolio_value=new_portfolio_value,
            peak_portfolio_value=updated_peak_portfolio_value,
            reward_config=self.reward_config,
        )

        self.portfolio_value = new_portfolio_value
        self.peak_portfolio_value = updated_peak_portfolio_value
        self.previous_weights = weights
        self.current_step += 1

        done = self.current_step >= len(self.returns)
        observation = (
            np.zeros(self.observation_dim, dtype=float) if done else self._get_observation()
        )
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "financial_net_return": financial_net_return,
            "transaction_cost": realized_transaction_cost,
            "turnover": turnover,
            "weights": weights,
            "drawdown": drawdown,
            "concentration": concentration,
            "reward": reward,
            "peak_portfolio_value": self.peak_portfolio_value,
        }

        return observation, reward, done, info

    def _get_observation(self) -> np.ndarray:
        return self.features.iloc[self.current_step].to_numpy(dtype=float)

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

    def _align_returns_and_features(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if features is None:
            return returns, returns
        if features.empty:
            raise ValueError("features must not be empty.")

        shared_index = returns.index[returns.index.isin(features.index)]
        if shared_index.empty:
            raise ValueError("returns and features must have at least one shared index.")

        aligned_returns = returns.loc[shared_index]
        aligned_features = features.loc[shared_index]

        if aligned_returns.empty or aligned_features.empty:
            raise ValueError("aligned returns and features must not be empty.")

        return aligned_returns, aligned_features
