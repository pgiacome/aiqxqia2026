"""Diagnostics: how far a kernel is from being ultrametric, and how expressive it is."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "dimension_lower_bound",
    "gromov_delta",
    "kernel_target_alignment",
    "level_constancy",
    "profile_residual",
    "qubit_lower_bound",
    "regular_tree_dimension_bound",
    "regular_tree_qubit_bound",
    "resolution_depth",
    "strong_triangle_violations",
    "violation_rate",
]

MAX_EXACT_SCAN = 512


class _Callable(Protocol):
    def __call__(self, v: int | np.ndarray) -> np.ndarray: ...


def _excess(dist: np.ndarray) -> np.ndarray:
    """``E[x, y, z] = d(x, z) - max(d(x, y), d(y, z))``."""
    if dist.shape[0] > MAX_EXACT_SCAN:
        raise ValueError(
            f"exact triple scan is limited to {MAX_EXACT_SCAN} points; subsample first"
        )
    d_xy = dist[:, :, None]
    d_yz = dist[None, :, :]
    d_xz = dist[:, None, :]
    return d_xz - np.maximum(d_xy, d_yz)


def strong_triangle_violations(dist: np.ndarray, tol: float = 1e-9) -> tuple[int, float]:
    """Count triples breaking ``d(x, z) <= max(d(x, y), d(y, z))``, and the worst excess."""
    exc = _excess(dist)
    return int(np.count_nonzero(exc > tol)), float(exc.max())


def violation_rate(dist: np.ndarray, tol: float = 1e-9) -> float:
    """Violating triples as a fraction of all ordered triples."""
    count, _ = strong_triangle_violations(dist, tol=tol)
    m = dist.shape[0]
    return count / float(m**3)


def gromov_delta(dist: np.ndarray, base: int = 0) -> float:
    """Gromov hyperbolicity via the Gromov product with base point ``base``.

    ``delta = max over (x, y, z) of min(gp(x, y), gp(y, z)) - gp(x, z)`` where
    ``gp(x, y) = (d(x, w) + d(y, w) - d(x, y)) / 2``. An ultrametric space has
    ``delta = 0``.
    """
    w = base
    gp = 0.5 * (dist[:, w][:, None] + dist[:, w][None, :] - dist)
    inner = np.minimum(gp[:, :, None], gp[None, :, :])
    return float(np.max(inner - gp[:, None, :]))


def kernel_target_alignment(kernel: np.ndarray, labels: np.ndarray) -> float:
    """Frobenius alignment between ``K`` and the label kernel ``y y^T``."""
    target = (labels[:, None] == labels[None, :]).astype(float)
    num = float(np.sum(kernel * target))
    den = float(np.linalg.norm(kernel) * np.linalg.norm(target))
    return num / den if den > 0 else 0.0


def dimension_lower_bound(kernel: np.ndarray) -> float:
    """Theorem B: ``D >= L^2 / sum_{x,y} K(x, y)``.

    The Gram matrix ``G`` of the feature vectors is PSD with unit diagonal, so
    ``trace(G) = L`` and ``||G||_F^2 = sum K``. For a PSD matrix of rank ``r``,
    ``||G||_F^2 >= trace(G)^2 / r``, which rearranges to the stated bound.
    """
    total = float(kernel.sum())
    if total <= 0:
        raise ValueError("kernel sum must be positive")
    return float(kernel.shape[0] ** 2) / total


def qubit_lower_bound(kernel: np.ndarray) -> float:
    """The Theorem B bound expressed in qubits."""
    return float(np.log2(dimension_lower_bound(kernel)))


def regular_tree_dimension_bound(profile: _Callable, p: int, n: int) -> float:
    """The Theorem B bound on the *full* regular p-ary tree, in closed form.

    Evaluating :func:`dimension_lower_bound` on a subsample of ``L'`` leaves is still a
    valid lower bound -- restricting the point set can only make the kernel easier to
    realise -- but it is capped at ``L'``, so a subsampled estimate understates the
    bound and saturates at ``log2(L')`` qubits. This computes it on all ``p**n`` leaves
    without materialising them:

    ``D >= p**n / S`` where ``S = f(n) + sum_{v<n} (p**(n-v) - p**(n-v-1)) f(v)``

    is the number of leaves at each LCA depth from a fixed leaf, weighted by ``f``.
    """
    if p < 2 or n < 1:
        raise ValueError("require p >= 2 and n >= 1")
    total = float(profile(n))
    for v in range(n):
        siblings = p ** (n - v) - p ** (n - v - 1)
        total += siblings * float(profile(v))
    return float(p**n) / total


def regular_tree_qubit_bound(profile: _Callable, p: int, n: int) -> float:
    """:func:`regular_tree_dimension_bound` expressed in qubits."""
    return float(np.log2(regular_tree_dimension_bound(profile, p, n)))


def profile_residual(kernel: np.ndarray, lca: np.ndarray, profile: _Callable) -> float:
    """``max |K(x, y) - f(lambda(x, y))|``."""
    return float(np.max(np.abs(kernel - profile(lca))))


def level_constancy(kernel: np.ndarray, lca: np.ndarray) -> float:
    """The largest spread of ``K`` within a single LCA level.

    Zero exactly when ``K(x, y) = f(lambda(x, y))`` for some profile ``f``, i.e. when
    the kernel is ultrametric *with a profile*. This is the property Theorem A rules
    out for product maps, and it is strictly stronger than having no strong-triangle
    violations: the delta kernel of a basis encoding has zero violations but resolves
    nothing, so violations alone cannot separate the encodings.
    """
    spreads = [float(np.ptp(kernel[lca == v])) for v in np.unique(lca)]
    return max(spreads) if spreads else 0.0


def resolution_depth(kernel: np.ndarray, lca: np.ndarray, tol: float = 1e-6) -> int:
    """How many distinct kernel values the encoding assigns across LCA levels.

    Theorem A says an ultrametric product map resolves the tree only to depth
    ``v* + 1``, giving at most three distinct values. The exact construction gives
    ``height + 1``. A basis encoding gives 2.
    """
    means = np.array([float(kernel[lca == v].mean()) for v in np.unique(lca)])
    keep = [means[0]]
    for value in means[1:]:
        if abs(value - keep[-1]) > tol:
            keep.append(value)
    return len(keep)
