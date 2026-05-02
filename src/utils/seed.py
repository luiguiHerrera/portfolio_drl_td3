"""Reproducibility utilities.

This module provides a single entrypoint for setting random seeds across the
libraries used by the project. PyTorch seeding is applied only when PyTorch is
available in the current environment.
"""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and optionally PyTorch."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
