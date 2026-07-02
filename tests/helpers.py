"""Shared, non-fixture test helpers.

Not a conftest.py: these are plain functions imported directly by test
modules (`from helpers import equicorr_matrix`), not pytest fixtures.
"""

import numpy as np


def equicorr_matrix(N: int, rho: float) -> np.ndarray:
    """N x N equicorrelation matrix with off-diagonal rho."""
    return (1 - rho) * np.eye(N) + rho * np.ones((N, N))
