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
    "max_branching",
    "pad_to_uniform_depth",
    "restrict_to_leaves",
    "truncate_at_depth",
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


def pad_to_uniform_depth(tree: RootedTree) -> tuple[RootedTree, np.ndarray]:
    """Extend every shallow leaf by a chain of unary nodes down to ``tree.height``.

    Padding does not change LCA depths between the original leaves, because a unary
    chain adds no branching; it only makes every leaf sit at the same depth so that a
    single amplitude profile applies uniformly.

    Returns the padded tree and a ``tip`` array: ``tip[u]`` is ``u`` itself for every
    node that was already deep enough, and the bottom of ``u``'s padding chain for a
    shallow leaf. Callers holding leaf indices **must** push them through ``tip``,
    because a padded leaf is no longer a leaf of the padded tree.
    """
    height = tree.height
    parents = list(map(int, tree.parent))
    depths = list(map(int, tree.depth))
    labels = list(tree.labels) if tree.labels else [str(i) for i in range(tree.num_nodes)]
    tip = np.arange(tree.num_nodes, dtype=np.int64)
    for leaf in map(int, tree.leaves):
        current, d = leaf, int(tree.depth[leaf])
        while d < height:
            d += 1
            parents.append(current)
            depths.append(d)
            labels.append(f"{labels[leaf]}#pad{d}")
            current = len(parents) - 1
        tip[leaf] = current
    padded = RootedTree(
        parent=np.asarray(parents, dtype=np.int64),
        depth=np.asarray(depths, dtype=np.int64),
        labels=tuple(labels),
    )
    return padded, tip


def truncate_at_depth(tree: RootedTree, max_depth: int) -> RootedTree:
    """Drop every node below ``max_depth``, making depth-``max_depth`` nodes leaves.

    Real hierarchies are deep and irregular; WordNet's noun tree reaches depth 19.
    Truncation bounds the height so that the ``2**height``-dimensional product
    baselines stay computable and every encoding is compared on the same tree. It is a
    stated preprocessing choice, not a silent one.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    keep = np.flatnonzero(tree.depth <= max_depth)
    sub, _ = _induced_subtree(tree, keep)
    return sub


def restrict_to_leaves(tree: RootedTree, leaves: np.ndarray) -> tuple[RootedTree, np.ndarray]:
    """Restrict ``tree`` to the ancestor closure of ``leaves``; return it and the remap.

    LCA depths among ``leaves`` are unchanged, because every ancestor of a kept leaf is
    kept. Only branches leading to discarded leaves disappear. Without this the
    path-state dimension is the size of the whole hierarchy -- 74k nodes for WordNet --
    when only the sampled leaves' paths can ever carry amplitude.
    """
    # Walk parents rather than calling ancestor_matrix: leaves may sit at mixed depths
    # at this point, since padding happens after restriction.
    frontier = np.unique(np.asarray(leaves, dtype=np.int64))
    keep = frontier
    while frontier.size:
        frontier = np.unique(tree.parent[frontier])
        frontier = frontier[frontier >= 0]
        frontier = np.setdiff1d(frontier, keep, assume_unique=False)
        keep = np.union1d(keep, frontier)
    sub, index_of = _induced_subtree(tree, keep)
    return sub, index_of[np.asarray(leaves, dtype=np.int64)]


def _induced_subtree(tree: RootedTree, keep: np.ndarray) -> tuple[RootedTree, np.ndarray]:
    """The subtree on ``keep``, which must be closed under taking parents.

    Returns the subtree and an ``old index -> new index`` map (``-1`` where dropped).
    """
    keep = np.unique(np.asarray(keep, dtype=np.int64))
    index_of = np.full(tree.num_nodes, -1, dtype=np.int64)
    index_of[keep] = np.arange(keep.shape[0], dtype=np.int64)
    old_parent = tree.parent[keep]
    parent = np.where(old_parent < 0, -1, index_of[np.clip(old_parent, 0, None)])
    if int((parent < 0).sum()) != 1:
        raise ValueError("kept node set is not closed under taking parents")
    labels = tuple(tree.labels[i] for i in keep) if tree.labels else ()
    sub = RootedTree(parent=parent, depth=tree.depth[keep].copy(), labels=labels)
    return sub, index_of


def max_branching(tree: RootedTree) -> int:
    """The largest number of children of any node -- the effective radix."""
    inner = tree.parent[tree.parent >= 0]
    if inner.size == 0:
        return 1
    return int(np.bincount(inner).max())


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
