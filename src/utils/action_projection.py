"""Shared portfolio action projection utilities."""

from __future__ import annotations

import numpy as np
import torch


def project_portfolio_action(action, max_weight: float | None = None) -> np.ndarray:
    """Project one action to long-only fully-invested portfolio weights."""
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
    """Project weights to a long-only simplex with a per-asset upper bound."""
    weights = np.asarray(action, dtype=float)
    if weights.ndim != 1:
        raise ValueError("action must be one-dimensional.")
    if weights.size == 0:
        raise ValueError("action must not be empty.")
    if not np.isfinite(weights).all():
        raise ValueError("action must contain only finite values.")

    cap = _validate_max_weight(max_weight, weights.size, tolerance=tolerance)
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


def project_torch_portfolio_actions(
    actions: torch.Tensor,
    max_weight: float | None = None,
    tolerance: float = 1e-12,
) -> torch.Tensor:
    """Project one or more torch actions to the feasible portfolio simplex."""
    if actions.dim() not in {1, 2}:
        raise ValueError("actions must be a one- or two-dimensional tensor.")
    single_action = actions.dim() == 1
    projected = actions.unsqueeze(0) if single_action else actions
    if projected.shape[-1] == 0:
        raise ValueError("actions must not be empty.")

    normalized = _normalize_torch_non_negative(projected)
    if max_weight is None:
        return normalized.squeeze(0) if single_action else normalized

    cap = _validate_max_weight(max_weight, projected.shape[-1], tolerance=tolerance)
    capped = torch.minimum(normalized, torch.full_like(normalized, cap))

    for _ in range(projected.shape[-1] * 2):
        deficit = 1.0 - capped.sum(dim=-1, keepdim=True)
        if bool(torch.all(torch.abs(deficit) <= tolerance)):
            break
        room = torch.clamp(cap - capped, min=0.0)
        room_total = room.sum(dim=-1, keepdim=True)
        adjustment = torch.where(
            room_total > tolerance,
            room / room_total.clamp(min=tolerance) * deficit,
            torch.zeros_like(room),
        )
        capped = torch.minimum(capped + adjustment, torch.full_like(capped, cap))

    capped = torch.clamp(capped, min=0.0, max=cap)
    total = capped.sum(dim=-1, keepdim=True)
    room = torch.clamp(cap - capped, min=0.0)
    room_total = room.sum(dim=-1, keepdim=True)
    redistribute = room / room_total.clamp(min=tolerance) * (1.0 - total)
    capped = torch.where(room_total > tolerance, capped + redistribute, capped)
    capped = torch.clamp(capped, min=0.0, max=cap)

    total = capped.sum(dim=-1, keepdim=True)
    normalized_capped = torch.where(
        total > tolerance,
        capped / total.clamp(min=tolerance),
        torch.full_like(capped, 1.0 / projected.shape[-1]),
    )
    return normalized_capped.squeeze(0) if single_action else normalized_capped


def _normalize_non_negative(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(weights, a_min=0.0, a_max=None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(clipped.shape, 1.0 / clipped.size, dtype=float)
    return clipped / total


def _normalize_torch_non_negative(actions: torch.Tensor) -> torch.Tensor:
    clipped = actions.clamp(min=0.0)
    row_sums = clipped.sum(dim=-1, keepdim=True)
    equal_weights = torch.full_like(clipped, 1.0 / actions.shape[-1])
    return torch.where(row_sums > 0.0, clipped / row_sums.clamp(min=1e-12), equal_weights)


def _validate_max_weight(
    max_weight: float,
    n_assets: int,
    *,
    tolerance: float = 1e-12,
) -> float:
    cap = float(max_weight)
    if cap <= 0.0 or cap > 1.0:
        raise ValueError("max_weight must be in (0, 1].")
    if cap * n_assets < 1.0 - tolerance:
        raise ValueError("max_weight cap is infeasible for the number of assets.")
    return cap
