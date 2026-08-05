"""Kernel profiles ``f`` and the path-state amplitudes they induce."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "Profile",
    "delta_profile",
    "geometric_profile",
    "linear_profile",
    "uniform_profile",
]

_TOL = 1e-12


@dataclass(frozen=True)
class Profile:
    """A map ``f : {0, ..., n} -> [0, 1]`` with ``f(n) = 1``."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) < 1:
            raise ValueError("profile needs at least one value")
        if abs(self.values[-1] - 1.0) > _TOL:
            raise ValueError("a kernel profile must satisfy f(n) = 1")
        if min(self.values) < -_TOL or max(self.values) > 1.0 + _TOL:
            raise ValueError("profile values must lie in [0, 1]")

    @property
    def height(self) -> int:
        return len(self.values) - 1

    @property
    def is_strictly_monotone(self) -> bool:
        v = np.asarray(self.values)
        return bool(np.all(np.diff(v) > _TOL) and v[0] > _TOL)

    def __call__(self, v: int | np.ndarray) -> np.ndarray:
        return np.asarray(self.values)[np.asarray(v)]

    def amplitudes(self) -> np.ndarray:
        """``a[i] = sqrt(sqrt(f(i)) - sqrt(f(i-1)))``, with ``sqrt(f(-1)) := 0``."""
        if not self.is_strictly_monotone:
            raise ValueError("path-state amplitudes require a strictly monotone profile")
        root = np.sqrt(np.asarray(self.values))
        squared = np.diff(root, prepend=0.0)
        return np.sqrt(squared)


def geometric_profile(n: int, p: int, s: float = 1.0) -> Profile:
    """``f(v) = p ** (-(n - v) * s)`` -- the p-adic absolute value raised to ``s``."""
    if s <= 0:
        raise ValueError("require s > 0")
    return Profile(tuple(float(p ** (-(n - v) * s)) for v in range(n + 1)))


def linear_profile(n: int) -> Profile:
    """``f(v) = (v + 1) / (n + 1)`` -- the slowest strictly monotone profile we use."""
    return Profile(tuple((v + 1) / (n + 1) for v in range(n + 1)))


def uniform_profile(n: int) -> Profile:
    """``f == 1``: the degenerate constant kernel. Ablation baseline only."""
    return Profile(tuple(1.0 for _ in range(n + 1)))


def delta_profile(n: int) -> Profile:
    """``f = 1[v = n]``: the kernel induced by computational-basis encoding."""
    return Profile(tuple(1.0 if v == n else 0.0 for v in range(n + 1)))
