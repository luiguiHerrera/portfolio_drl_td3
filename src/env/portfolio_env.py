"""Minimal portfolio allocation environment.

This module defines a lightweight environment for validating portfolio mechanics
before adding Gymnasium integration or TD3 training. The environment consumes a
precomputed returns DataFrame for realized portfolio returns and an optional
features DataFrame for agent observations. It enforces long-only, fully invested
portfolio weights through clipping and normalization.
"""

import numpy as np
import pandas as pd

from src.risk.mandate_penalties import compute_cash_breach, compute_mandate_penalty
from src.risk.mandate_profiles import get_mandate_limits
from src.rewards.reward import (
    compute_turnover_penalty,
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
        self.financial_net_return_history = []

    def reset(self) -> np.ndarray:
        """Reset environment state and return the initial observation."""
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.previous_weights = self._equal_weights()
        self.financial_net_return_history = []

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
        self.financial_net_return_history.append(financial_net_return)
        reward = compute_risk_aware_reward(
            portfolio_return=portfolio_return,
            transaction_cost=realized_transaction_cost,
            turnover=turnover,
            weights=weights,
            portfolio_value=new_portfolio_value,
            peak_portfolio_value=updated_peak_portfolio_value,
            reward_config=self.reward_config,
        )
        turnover_penalty_result = self._compute_turnover_penalty(turnover)
        mandate_result = None
        if self.reward_config.get("use_mandate_penalty", False):
            mandate_result = self._compute_mandate_penalty(
                drawdown=drawdown,
                turnover=turnover,
                weights=weights,
                concentration=concentration,
            )
            lambda_mandate = self._reward_config_number("lambda_mandate", 0.0)
            reward -= lambda_mandate * mandate_result["penalty"]
        cash_penalty_result = None
        if self.reward_config.get("use_cash_risk_off_penalty", False):
            cash_penalty_result = self._compute_cash_penalty(weights)
            reward -= cash_penalty_result["cash_penalty"]

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
            "turnover_penalty": turnover_penalty_result["turnover_penalty"],
            "turnover_penalty_mode": turnover_penalty_result[
                "turnover_penalty_mode"
            ],
            "turnover_free_band": turnover_penalty_result["turnover_free_band"],
            "turnover_excess": turnover_penalty_result["turnover_excess"],
            "weights": weights,
            "drawdown": drawdown,
            "concentration": concentration,
            "reward": reward,
            "peak_portfolio_value": self.peak_portfolio_value,
        }
        if mandate_result is not None:
            breaches = mandate_result["breaches"]
            info.update(
                {
                    "mandate_penalty": mandate_result["penalty"],
                    "mandate_drawdown_breach": breaches["drawdown_breach"],
                    "mandate_volatility_breach": breaches["volatility_breach"],
                    "mandate_max_weight_breach": breaches["max_weight_breach"],
                    "mandate_effective_assets_breach": breaches[
                        "effective_assets_breach"
                    ],
                    "mandate_turnover_breach": breaches["turnover_breach"],
                }
            )
        if cash_penalty_result is not None:
            info.update(cash_penalty_result)

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

    def _compute_mandate_penalty(
        self,
        drawdown: float,
        turnover: float,
        weights: np.ndarray,
        concentration: float,
    ) -> dict:
        mandate_limits = get_mandate_limits(
            self.reward_config.get("mandate_profile", "moderate")
        )
        return compute_mandate_penalty(
            current_drawdown=-drawdown,
            current_volatility=self._current_trailing_volatility(),
            max_weight=float(weights.max()),
            effective_assets=self._effective_assets(concentration),
            turnover=turnover,
            mandate_limits=mandate_limits,
            penalty_weights=self.reward_config.get("mandate_penalty_weights"),
        )

    def _compute_cash_penalty(self, weights: np.ndarray) -> dict:
        normal_cash_max = self._reward_config_number("normal_cash_max", 0.10)
        cash_penalty_weight = self._reward_config_number("cash_penalty_weight", 1.0)
        cash_risk_off_state = self._reward_config_bool("cash_risk_off_state", False)
        cash_weight = self._current_cash_weight(weights)
        cash_breach = compute_cash_breach(
            cash_weight=cash_weight,
            normal_cash_max=normal_cash_max,
            risk_off_state=cash_risk_off_state,
        )
        cash_penalty = cash_penalty_weight * cash_breach

        return {
            "cash_penalty": cash_penalty,
            "cash_breach": cash_breach,
            "normal_cash_max": normal_cash_max,
            "cash_risk_off_state": cash_risk_off_state,
        }

    def _compute_turnover_penalty(self, turnover: float) -> dict:
        lambda_turnover = self._reward_config_number("lambda_turnover", 0.0)
        mode = self.reward_config.get("turnover_penalty_mode", "linear")
        free_band = self._reward_config_number("turnover_free_band", 0.0)
        quadratic_weight = self._reward_config_number(
            "turnover_quadratic_weight",
            0.0,
        )
        turnover_penalty = compute_turnover_penalty(
            turnover=turnover,
            lambda_turnover=lambda_turnover,
            mode=mode,
            free_band=free_band,
            quadratic_weight=quadratic_weight,
        )
        if mode == "linear":
            turnover_excess = turnover
        elif mode == "none":
            turnover_excess = 0.0
        else:
            turnover_excess = max(turnover - free_band, 0.0)

        return {
            "turnover_penalty": turnover_penalty,
            "turnover_penalty_mode": mode,
            "turnover_free_band": free_band,
            "turnover_excess": turnover_excess,
        }

    def _current_cash_weight(self, weights: np.ndarray) -> float:
        if "CASH" not in self.asset_names:
            return 0.0

        return float(weights[self.asset_names.index("CASH")])

    def _current_trailing_volatility(self) -> float:
        window = int(self.reward_config.get("mandate_volatility_window", 12))
        if len(self.financial_net_return_history) < window:
            return 0.0

        recent_returns = np.asarray(self.financial_net_return_history[-window:], dtype=float)
        return float(np.std(recent_returns, ddof=1) * np.sqrt(52))

    def _effective_assets(self, concentration: float) -> float:
        if concentration <= 0.0:
            return float("inf")

        return float(1.0 / concentration)

    def _reward_config_number(self, field_name: str, default: float) -> float:
        value = self.reward_config.get(field_name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Reward config field {field_name} must be numeric.")
        if value < 0.0:
            raise ValueError(f"Reward config field {field_name} must be non-negative.")

        return float(value)

    def _reward_config_bool(self, field_name: str, default: bool) -> bool:
        value = self.reward_config.get(field_name, default)
        if not isinstance(value, bool):
            raise ValueError(f"Reward config field {field_name} must be bool.")

        return value

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
