"""Training-only behavior-policy exploration helpers."""

from __future__ import annotations

import numpy as np


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
    weights = np.asarray(action, dtype=float)
    if weights.ndim != 1:
        raise ValueError("action must be one-dimensional.")
    if weights.size == 0:
        raise ValueError("action must not be empty.")
    if not np.isfinite(weights).all():
        raise ValueError("action must contain only finite values.")

    if max_weight is None:
        return _normalize_non_negative(weights)
    return project_to_capped_simplex(weights, max_weight=max_weight)


def project_to_capped_simplex(
    action,
    *,
    max_weight: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Project weights to a long-only simplex with per-asset upper bound."""
    weights = np.asarray(action, dtype=float)
    if weights.ndim != 1:
        raise ValueError("action must be one-dimensional.")
    if weights.size == 0:
        raise ValueError("action must not be empty.")
    if not np.isfinite(weights).all():
        raise ValueError("action must contain only finite values.")

    cap = float(max_weight)
    if cap <= 0.0 or cap > 1.0:
        raise ValueError("max_weight must be in (0, 1].")
    if cap * weights.size < 1.0 - tolerance:
        raise ValueError("max_weight cap is infeasible for the number of assets.")

    projected = _normalize_non_negative(weights)
    capped = np.minimum(projected, cap)

    for _ in range(weights.size * 2):
        deficit = 1.0 - float(capped.sum())
        if abs(deficit) <= tolerance:
            break
        room = np.maximum(cap - capped, 0.0)
        room_total = float(room.sum())
        if room_total <= tolerance:
            break
        capped += room / room_total * deficit
        capped = np.minimum(capped, cap)

    capped = np.clip(capped, 0.0, cap)
    total = float(capped.sum())
    if abs(total - 1.0) > 1e-9:
        room = np.maximum(cap - capped, 0.0)
        room_total = float(room.sum())
        if room_total <= tolerance:
            capped = capped / total
        else:
            capped += room / room_total * (1.0 - total)
    return capped / capped.sum()


def _normalize_non_negative(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(weights, a_min=0.0, a_max=None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(clipped.shape, 1.0 / clipped.size, dtype=float)
    return clipped / total
