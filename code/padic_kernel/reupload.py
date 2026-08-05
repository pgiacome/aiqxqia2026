"""Data re-uploading on top of the exact path-state encoding.

The exact construction sits at ``gamma = 0, entangle = 0``; increasing either knob
trades ultrametricity for expressivity. This is the tunable family whose Pareto front
E4 measures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from padic_kernel.encoding import Encoding
from padic_kernel.tree import RootedTree, ancestor_matrix

logger = logging.getLogger(__name__)

__all__ = ["ReuploadEncoding", "haar_unitary"]


def haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-random unitary of size ``dim`` via QR of a complex Ginibre matrix."""
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    return q * (np.diag(r) / np.abs(np.diag(r)))[None, :]


@dataclass
class ReuploadEncoding(Encoding):
    """``Phi(x) = W_L D(x) ... W_1 D(x) Phi_base(x)``.

    ``D(x)`` applies phase ``gamma`` to every node on the root-to-leaf path of ``x``,
    which is the natural data-dependent diagonal on the path-state Hilbert space.
    ``entangle`` interpolates each ``W_l`` between the identity and a Haar unitary.
    """

    base: Encoding
    layers: int = 1
    gamma: float = 0.0
    entangle: float = 0.0
    seed: int = 0
    name: str = field(default="reupload", init=False)

    def _mixers(self, dim: int) -> list[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        mixers = []
        for _ in range(self.layers):
            u = haar_unitary(dim, rng)
            m = (1.0 - self.entangle) * np.eye(dim) + self.entangle * u
            # Re-unitarise the interpolant so the map stays norm-preserving.
            q, r = np.linalg.qr(m)
            mixers.append(q * (np.diag(r) / np.abs(np.diag(r)))[None, :])
        return mixers

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        psi = self.base.states(tree, leaves)
        dim = psi.shape[1]
        anc = ancestor_matrix(tree, leaves)
        on_path = np.zeros((psi.shape[0], dim), dtype=bool)
        rows = np.arange(psi.shape[0])[:, None]
        on_path[rows, anc] = True
        phases = np.exp(1j * self.gamma * on_path)
        for mixer in self._mixers(dim):
            psi = (psi * phases) @ mixer.T
        return psi / np.linalg.norm(psi, axis=1, keepdims=True)
