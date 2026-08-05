"""Rooted trees, ancestor paths, and lowest-common-ancestor depth.

Level convention. For a leaf ``x`` at depth ``n``, ``anc_i(x)`` is its depth-``i``
ancestor (``anc_0(x)`` is the root, ``anc_n(x) = x``). Digit ``i`` (1-indexed) is
the index of ``anc_i(x)`` among the children of ``anc_{i-1}(x)``. On the regular
p-ary tree built by :func:`build_padic_tree` this makes digit ``i`` the ``i``-th
least-significant base-p digit of the leaf index, so the LCA depth of two leaves
equals the p-adic valuation ``v_p(x - y)`` capped at ``n``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "RootedTree",
    "ancestor_matrix",
    "build_padic_tree",
    "digit_matrix",
    "from_parent_map",
    "lca_depth_matrix",
    "pad_to_uniform_depth",
]


@dataclass(frozen=True)
class RootedTree:
    """A finite rooted tree over node indices ``0 .. num_nodes - 1``."""

    parent: np.ndarray
    depth: np.ndarray
    labels: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.parent.ndim != 1 or self.depth.shape != self.parent.shape:
            raise ValueError("parent and depth must be 1-D arrays of equal length")
        if int((self.parent < 0).sum()) != 1:
            raise ValueError("a rooted tree needs exactly one root")

    @property
    def num_nodes(self) -> int:
        return int(self.parent.shape[0])

    @property
    def height(self) -> int:
        return int(self.depth.max())

    @property
    def root(self) -> int:
        return int(np.flatnonzero(self.parent < 0)[0])

    @property
    def leaves(self) -> np.ndarray:
        has_child = np.zeros(self.num_nodes, dtype=bool)
        inner = self.parent[self.parent >= 0]
        has_child[inner] = True
        return np.flatnonzero(~has_child)

    def children(self, node: int) -> np.ndarray:
        return np.flatnonzero(self.parent == node)


def build_padic_tree(p: int, n: int) -> RootedTree:
    """The regular p-ary tree of depth ``n``; leaf ``x`` is the integer ``x`` in Z/p^n.

    Nodes are laid out level by level. The depth-``i`` level occupies indices
    ``off(i) .. off(i) + p**i - 1`` with ``off(i) = (p**i - 1) // (p - 1)``, and the
    node at position ``r`` of level ``i`` represents the residue class ``r mod p**i``.
    Hence ``anc_i(x) = off(i) + (x mod p**i)``.
    """
    if p < 2 or n < 1:
        raise ValueError("require p >= 2 and n >= 1")
    offsets = [(p**i - 1) // (p - 1) for i in range(n + 2)]
    num_nodes = offsets[n + 1]
    parent = np.empty(num_nodes, dtype=np.int64)
    depth = np.empty(num_nodes, dtype=np.int64)
    parent[0] = -1
    depth[0] = 0
    for i in range(1, n + 1):
        residues = np.arange(p**i, dtype=np.int64)
        idx = offsets[i] + residues
        parent[idx] = offsets[i - 1] + (residues % p ** (i - 1))
        depth[idx] = i
    logger.info("Built p-adic tree p=%d n=%d with %d nodes", p, n, num_nodes)
    return RootedTree(parent=parent, depth=depth)


def from_parent_map(parent_of: dict[str, str | None]) -> RootedTree:
    """Build a tree from a ``child -> parent`` label map (root maps to ``None``)."""
    names = sorted(parent_of)
    index = {name: i for i, name in enumerate(names)}
    parent = np.full(len(names), -1, dtype=np.int64)
    for name, par in parent_of.items():
        if par is not None:
            if par not in index:
                raise ValueError(f"parent {par!r} of {name!r} is not a node")
            parent[index[name]] = index[par]
    if int((parent < 0).sum()) != 1:
        raise ValueError("a rooted tree needs exactly one root")
    depth = _depths_from_parent(parent)
    return RootedTree(parent=parent, depth=depth, labels=tuple(names))


def _depths_from_parent(parent: np.ndarray) -> np.ndarray:
    depth = np.full(parent.shape[0], -1, dtype=np.int64)
    root = int(np.flatnonzero(parent < 0)[0])
    depth[root] = 0
    frontier = np.array([root], dtype=np.int64)
    level = 0
    while frontier.size:
        level += 1
        frontier = np.flatnonzero(np.isin(parent, frontier))
        depth[frontier] = level
    if int((depth < 0).sum()):
        raise ValueError("graph is not connected: some nodes are unreachable from the root")
    return depth


def pad_to_uniform_depth(tree: RootedTree) -> RootedTree:
    """Extend every shallow leaf by a chain of unary nodes down to ``tree.height``.

    Padding does not change LCA depths between the original leaves, because a unary
    chain adds no branching; it only makes every leaf sit at the same depth so that
    a single amplitude profile applies uniformly.
    """
    height = tree.height
    parents = list(map(int, tree.parent))
    depths = list(map(int, tree.depth))
    labels = list(tree.labels) if tree.labels else [str(i) for i in range(tree.num_nodes)]
    for leaf in map(int, tree.leaves):
        current, d = leaf, int(tree.depth[leaf])
        while d < height:
            d += 1
            parents.append(current)
            depths.append(d)
            labels.append(f"{labels[leaf]}#pad{d}")
            current = len(parents) - 1
    return RootedTree(
        parent=np.asarray(parents, dtype=np.int64),
        depth=np.asarray(depths, dtype=np.int64),
        labels=tuple(labels),
    )


def ancestor_matrix(tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
    """``anc[j, i]`` is the depth-``i`` ancestor of ``leaves[j]``."""
    leaves = np.asarray(leaves, dtype=np.int64)
    height = tree.height
    if not np.all(tree.depth[leaves] == height):
        raise ValueError("all leaves must sit at the tree height; call pad_to_uniform_depth")
    anc = np.empty((leaves.shape[0], height + 1), dtype=np.int64)
    anc[:, height] = leaves
    for i in range(height - 1, -1, -1):
        anc[:, i] = tree.parent[anc[:, i + 1]]
    return anc


def lca_depth_matrix(anc: np.ndarray) -> np.ndarray:
    """Pairwise LCA depth from an ancestor matrix."""
    agree = anc[:, None, :] == anc[None, :, :]
    return agree.sum(axis=2).astype(np.int64) - 1


def digit_matrix(tree: RootedTree, leaves: np.ndarray) -> np.ndarray:
    """``dig[j, i - 1]`` is the child-index of ``anc_i`` among the children of ``anc_{i-1}``."""
    anc = ancestor_matrix(tree, leaves)
    order = np.zeros(tree.num_nodes, dtype=np.int64)
    sort_by_parent = np.argsort(tree.parent, kind="stable")
    parents_sorted = tree.parent[sort_by_parent]
    starts = np.searchsorted(parents_sorted, parents_sorted, side="left")
    order[sort_by_parent] = np.arange(tree.num_nodes, dtype=np.int64) - starts
    return order[anc[:, 1:]]
