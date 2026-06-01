"""Training-only behavior-policy exploration helpers."""

from __future__ import annotations

import numpy as np

from src.utils.action_projection import (
    project_portfolio_action,
    project_to_capped_simplex,
)


def apply_behavior_exploration_noise(
    action,
    *,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    noise_clip: float | None = None,
    max_weight: float | None = None,
) -> np.ndarray:
    """Add Gaussian behavior noise and project back to the portfolio simplex.

    This helper is intended only for experience collection during training.
    Evaluation/backtesting should continue using deterministic actor actions.
    """
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    if noise_clip is not None and noise_clip < 0.0:
        raise ValueError("noise_clip must be non-negative.")

    action_array = np.asarray(action, dtype=float)
    if action_array.ndim != 1:
        raise ValueError("action must be one-dimensional.")
    if not np.isfinite(action_array).all():
        raise ValueError("action must contain only finite values.")

    noisy_action = action_array.copy()
    if noise_std > 0.0:
        selected_rng = np.random.default_rng() if rng is None else rng
        noise = selected_rng.normal(loc=0.0, scale=noise_std, size=action_array.shape)
        if noise_clip is not None:
            noise = np.clip(noise, -noise_clip, noise_clip)
        noisy_action = noisy_action + noise

    return project_to_simplex(noisy_action, max_weight=max_weight)


def project_to_simplex(action, max_weight: float | None = None) -> np.ndarray:
    """Project a one-dimensional action to long-only fully-invested weights."""
    return project_portfolio_action(action, max_weight=max_weight)
