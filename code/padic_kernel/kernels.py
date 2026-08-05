"""Fidelity kernels, and simple noise and finite-shot models."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["depolarise", "fidelity_gram", "overlap_gram", "sample_kernel"]


def overlap_gram(psi: np.ndarray) -> np.ndarray:
    """``G[x, y] = <psi_x | psi_y>`` for unit rows ``psi``."""
    return psi.conj() @ psi.T


def fidelity_gram(psi: np.ndarray) -> np.ndarray:
    """``K[x, y] = |<psi_x | psi_y>|^2``, clipped into ``[0, 1]``."""
    k = np.abs(overlap_gram(psi)) ** 2
    k = 0.5 * (k + k.T)
    return np.clip(k, 0.0, 1.0)


def depolarise(kernel: np.ndarray, rate: float, dim: int) -> np.ndarray:
    """Global depolarising model on the compute-uncompute kernel estimator.

    The estimator reads off the probability of the all-zeros outcome. Replacing the
    state by the maximally mixed state with probability ``rate`` sends that
    probability to ``1 / dim``, so ``K -> (1 - rate) K + rate / dim``. Because the
    fidelity moves towards ``1 / dim`` rather than towards zero, a large ``rate``
    flattens the kernel, which is exactly what destroys the hierarchy.

    The diagonal is pinned to 1: ``K(x, x)`` is never actually measured in a
    kernel-matrix protocol, it is filled in by definition.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("depolarising rate must lie in [0, 1]")
    if dim < 1:
        raise ValueError("dim must be positive")
    out = (1.0 - rate) * kernel + rate / dim
    np.fill_diagonal(out, 1.0)
    return out


def sample_kernel(kernel: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Binomial finite-shot estimate of a fidelity kernel."""
    if shots < 1:
        raise ValueError("shots must be positive")
    m = kernel.shape[0]
    iu = np.triu_indices(m, k=1)
    draws = rng.binomial(shots, np.clip(kernel[iu], 0.0, 1.0)) / shots
    out = np.eye(m)
    out[iu] = draws
    out[(iu[1], iu[0])] = draws
    return out
