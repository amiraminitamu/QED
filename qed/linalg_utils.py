"""
Linear algebra helper utilities for QED routines.
"""

from __future__ import annotations

import numpy as np


def sym(A: np.ndarray) -> np.ndarray:
    """Hermitize a matrix for numerical stability."""
    return 0.5 * (A + A.conj().T)


def s_inv_sqrt(S: np.ndarray, thresh: float = 1e-12) -> np.ndarray:
    """
    Build ``S^{-1/2}`` via eigendecomposition for numerical stability.
    """
    S = sym(S)
    e, U = np.linalg.eigh(S)
    keep = e > thresh
    if not np.all(keep):
        raise RuntimeError(f"S has small/negative eigenvalues: min={e.min():.3e}")
    X = U @ np.diag(1.0 / np.sqrt(e)) @ U.conj().T
    return sym(X)
