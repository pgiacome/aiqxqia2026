"""Quantum feature maps as explicit statevectors.

Every encoding returns an ``(m, dim)`` complex array whose rows are unit vectors.
Fidelity kernels are formed downstream in :mod:`padic_kernel.kernels`; keeping the
statevectors explicit lets the same code serve exact, noisy, and finite-shot paths.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from padic_kernel.profiles import Profile
from padic_kernel.tree import RootedTree, ancestor_matrix, digit_matrix

logger = logging.getLogger(__name__)

__all__ = [
    "AngleEncoding",
    "BasisEncoding",
    "BlockProductEncoding",
    "Encoding",
    "EncodingFactory",
    "PathStateEncoding",
    "RandomProductEncoding",
    "ZZFeatureMapEncoding",
    "register_encoding",
]

ENCODING_REGISTRY: dict[str, type[Encoding]] = {}
E = TypeVar("E", bound="Encoding")


def register_encoding(name: str) -> Callable[[type[E]], type[E]]:
    def decorator(cls: type[E]) -> type[E]:
        ENCODING_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def EncodingFactory(name: str, **kwargs: Any) -> Encoding:
    """Instantiate a registered encoding by name."""
    if name not in ENCODING_REGISTRY:
        raise KeyError(f"unknown encoding {name!r}; known: {sorted(ENCODING_REGISTRY)}")
    return ENCODING_REGISTRY[name](**kwargs)


class Encoding(ABC):
    """A feature map from tree leaves to unit vectors."""

    name: str = "encoding"

    @abstractmethod
    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        """Return the ``(m, dim)`` complex statevector array for ``leaves``."""


def _kron_rows(factors: list[np.ndarray]) -> np.ndarray:
    """Row-wise Kronecker product of ``(m, d_i)`` arrays."""
    out = factors[0]
    for nxt in factors[1:]:
        out = (out[:, :, None] * nxt[:, None, :]).reshape(out.shape[0], -1)
    return out


@register_encoding("path_state")
@dataclass
class PathStateEncoding(Encoding):
    """``Phi_f(x) = sum_i a_i |anc_i(x)>`` -- the exact ultrametric construction."""

    profile: Profile

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        if self.profile.height != tree.height:
            raise ValueError(f"profile height {self.profile.height} != tree height {tree.height}")
        anc = ancestor_matrix(tree, leaves)
        amps = self.profile.amplitudes()
        psi = np.zeros((anc.shape[0], tree.num_nodes), dtype=np.complex128)
        rows = np.arange(anc.shape[0])[:, None]
        np.add.at(psi, (rows, anc), amps[None, :].astype(np.complex128))
        return psi


@register_encoding("angle")
@dataclass
class AngleEncoding(Encoding):
    """One qubit per level: ``|psi_i(a)> = cos(theta_a/2)|0> + sin(theta_a/2)|1>``."""

    radix: int

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        dig = digit_matrix(tree, leaves)
        theta = np.pi * dig / self.radix
        factors = [
            np.stack([np.cos(theta[:, i] / 2), np.sin(theta[:, i] / 2)], axis=1).astype(
                np.complex128
            )
            for i in range(dig.shape[1])
        ]
        return _kron_rows(factors)


@register_encoding("basis")
@dataclass
class BasisEncoding(Encoding):
    """One qudit per level in the computational basis."""

    radix: int

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        dig = digit_matrix(tree, leaves)
        factors = []
        for i in range(dig.shape[1]):
            block = np.zeros((dig.shape[0], self.radix), dtype=np.complex128)
            block[np.arange(dig.shape[0]), dig[:, i]] = 1.0
            factors.append(block)
        return _kron_rows(factors)


@register_encoding("random_product")
@dataclass
class RandomProductEncoding(Encoding):
    """A random product map: independent random unit vectors per (level, digit)."""

    radix: int
    local_dim: int
    seed: int = 0

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        dig = digit_matrix(tree, leaves)
        rng = np.random.default_rng(self.seed)
        factors = []
        for i in range(dig.shape[1]):
            table = rng.normal(size=(self.radix, self.local_dim)) + 1j * rng.normal(
                size=(self.radix, self.local_dim)
            )
            table /= np.linalg.norm(table, axis=1, keepdims=True)
            factors.append(table[dig[:, i]])
        return _kron_rows(factors)


@register_encoding("block_product")
@dataclass
class BlockProductEncoding(Encoding):
    """A product map over consecutive blocks of ``block_size`` levels.

    Each block carries a random unit vector per joint digit assignment, so the block
    is maximally expressive internally while the map remains a tensor product across
    blocks. This is the object Theorem A's block generalisation constrains.
    """

    radix: int
    block_size: int
    seed: int = 0

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        dig = digit_matrix(tree, leaves)
        rng = np.random.default_rng(self.seed)
        factors = []
        for start in range(0, dig.shape[1], self.block_size):
            block = dig[:, start : start + self.block_size]
            width = block.shape[1]
            codes = np.zeros(block.shape[0], dtype=np.int64)
            for j in range(width):
                codes = codes * self.radix + block[:, j]
            size = self.radix**width
            table = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
            table /= np.linalg.norm(table, axis=1, keepdims=True)
            factors.append(table[codes])
        return _kron_rows(factors)


@register_encoding("zz")
@dataclass
class ZZFeatureMapEncoding(Encoding):
    """The IQP-style ZZ feature map: ``(U_phi H^{ot n})^{reps} |0>``.

    ``phi_i(x) = x_i`` and ``phi_{ij}(x) = (pi - x_i)(pi - x_j)``, the standard
    second-order Pauli-Z choice. The all-pairs ZZ terms make this map entangling, so
    it is *not* a product map and Theorem A does not apply to it; Theorem B does.
    """

    radix: int
    reps: int = 2

    def states(self, tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
        dig = digit_matrix(tree, leaves)
        m, n = dig.shape
        angles = 2.0 * np.pi * dig / self.radix
        basis = np.arange(2**n)
        # z[k, i] = +1/-1 eigenvalue of Z_i on basis state k
        bits = ((basis[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        z = 1.0 - 2.0 * bits
        phase = angles @ z.T  # (m, 2**n) single-qubit part
        for i in range(n):
            for j in range(i + 1, n):
                coupling = (np.pi - angles[:, i]) * (np.pi - angles[:, j])
                phase = phase + 2.0 * coupling[:, None] * (z[:, i] * z[:, j])[None, :]
        psi = np.full((m, 2**n), 2.0 ** (-n / 2), dtype=np.complex128)
        for _ in range(self.reps):
            psi = psi * np.exp(1j * phase)
            psi = _hadamard_all(psi, n)
        return psi / np.linalg.norm(psi, axis=1, keepdims=True)


def _hadamard_all(psi: np.ndarray, n: int) -> np.ndarray:
    """Apply ``H^{ot n}`` to each row of ``psi`` by the fast Walsh-Hadamard transform."""
    out = psi.reshape(psi.shape[0], *([2] * n)).copy()
    for axis in range(1, n + 1):
        a = np.take(out, 0, axis=axis)
        b = np.take(out, 1, axis=axis)
        out = np.stack([(a + b), (a - b)], axis=axis)
    return out.reshape(psi.shape[0], 2**n) / np.sqrt(2.0) ** n
