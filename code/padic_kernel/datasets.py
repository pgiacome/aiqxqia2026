"""Hierarchical datasets reduced to rooted trees.

Every source here is a DAG in the wild. We reduce each to a tree by keeping the
longest root-ward path per node, breaking ties lexicographically. This is stated in
the paper as a named preprocessing choice, not hidden.

The pipeline is fixed and applies to every source:

1. build the full DAG and reduce it to a tree by longest path;
2. truncate at ``spec.max_depth`` so the ``2**height`` product baselines stay
   computable and every encoding is compared on the same tree;
3. subsample ``spec.max_leaves`` leaves under ``spec.seed``;
4. restrict to the sampled leaves' ancestor closure;
5. pad the remaining shallow leaves to uniform depth.

Steps 2-5 leave the LCA depths among the sampled leaves unchanged below the
truncation cap, which is all the kernel depends on.
"""

from __future__ import annotations

import logging
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from padic_kernel.tree import (
    RootedTree,
    ancestor_matrix,
    build_padic_tree,
    from_parent_map,
    max_branching,
    pad_to_uniform_depth,
    restrict_to_leaves,
    truncate_at_depth,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetFactory",
    "DatasetSpec",
    "LabelledTree",
    "finalise",
    "load_go",
    "load_ncbi",
    "load_synthetic",
    "load_wordnet",
    "register_dataset",
]

DATASET_REGISTRY: dict[str, Callable[..., LabelledTree]] = {}


@dataclass(frozen=True)
class DatasetSpec:
    """How much of a dataset to use, how deep to cut it, and with what seed."""

    name: str
    max_leaves: int = 256
    max_depth: int = 8
    seed: int = 42
    label_depth: int = 2


@dataclass(frozen=True)
class LabelledTree:
    """A padded tree, a leaf sample, and a class label per sampled leaf."""

    tree: RootedTree
    leaves: np.ndarray
    labels: np.ndarray
    radix: int
    source_nodes: int
    source_height: int
    label_names: tuple[str, ...] = ()

    @property
    def height(self) -> int:
        return self.tree.height


def register_dataset(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        DATASET_REGISTRY[name] = fn
        return fn

    return decorator


def DatasetFactory(name: str, **kwargs: Any) -> Callable[[DatasetSpec], LabelledTree]:
    """Return a loader closure for the named dataset."""
    if name not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASET_REGISTRY)}")
    builder = DATASET_REGISTRY[name]

    def loader(spec: DatasetSpec) -> LabelledTree:
        return builder(spec, **kwargs)

    return loader


def finalise(tree: RootedTree, spec: DatasetSpec) -> LabelledTree:
    """Apply steps 2-5 of the pipeline and attach labels."""
    source_nodes, source_height = tree.num_nodes, tree.height
    truncated = truncate_at_depth(tree, spec.max_depth)
    leaves = _subsample(truncated, spec)
    restricted, remapped = restrict_to_leaves(truncated, leaves)
    padded, tip = pad_to_uniform_depth(restricted)
    final_leaves = tip[remapped]
    labels = _labels_at_depth(padded, final_leaves, min(spec.label_depth, padded.height))
    logger.info(
        "%s: %d source nodes (height %d) -> %d sampled leaves, %d nodes, height %d, "
        "%d classes, radix %d",
        spec.name,
        source_nodes,
        source_height,
        final_leaves.shape[0],
        padded.num_nodes,
        padded.height,
        len(np.unique(labels)),
        max_branching(padded),
    )
    return LabelledTree(
        tree=padded,
        leaves=final_leaves,
        labels=labels,
        radix=max_branching(padded),
        source_nodes=source_nodes,
        source_height=source_height,
    )


def _subsample(tree: RootedTree, spec: DatasetSpec) -> np.ndarray:
    leaves = tree.leaves
    if leaves.shape[0] <= spec.max_leaves:
        return leaves
    rng = np.random.default_rng(spec.seed)
    return np.sort(rng.choice(leaves, size=spec.max_leaves, replace=False))


def _labels_at_depth(tree: RootedTree, leaves: np.ndarray, depth: int) -> np.ndarray:
    """Class label = the ancestor at ``depth``, remapped to 0-based ids."""
    anc = ancestor_matrix(tree, leaves)[:, depth]
    _, labels = np.unique(anc, return_inverse=True)
    return labels.astype(np.int64)


@register_dataset("synthetic")
def load_synthetic(spec: DatasetSpec, p: int = 2, n: int = 6) -> LabelledTree:
    """A regular p-ary tree; labels are the depth-``label_depth`` ancestor."""
    return finalise(build_padic_tree(p, n), spec)


def _longest_path_parents(children_of: dict[str, list[str]], root: str) -> dict[str, str | None]:
    """Keep, for each node, the parent that maximises its depth; ties broken by name.

    A Bellman-Ford style relaxation over levels rather than a single BFS, because the
    longest path to a node is not found on its first visit in a DAG.
    """
    depth: dict[str, int] = {root: 0}
    parent: dict[str, str | None] = {root: None}
    frontier = [root]
    while frontier:
        nxt: set[str] = set()
        for u in frontier:
            for v in children_of.get(u, ()):
                cand = depth[u] + 1
                if v not in depth or (cand, u) > (depth[v], parent[v] or ""):
                    depth[v] = cand
                    parent[v] = u
                    nxt.add(v)
        frontier = sorted(nxt)
    return parent


@register_dataset("wordnet")
def load_wordnet(spec: DatasetSpec, root_name: str = "entity.n.01") -> LabelledTree:
    """WordNet nouns via NLTK, reduced to a tree by the longest hypernym path."""
    import nltk
    from nltk.corpus import wordnet as wn

    try:
        wn.synset(root_name)
    except LookupError:  # pragma: no cover - one-time download path
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    children_of: dict[str, list[str]] = {}
    stack = [wn.synset(root_name)]
    seen = {root_name}
    while stack:
        syn = stack.pop()
        kids = syn.hyponyms()
        children_of[syn.name()] = sorted(k.name() for k in kids)
        for k in kids:
            if k.name() not in seen:
                seen.add(k.name())
                stack.append(k)
    return finalise(from_parent_map(_longest_path_parents(children_of, root_name)), spec)


@register_dataset("go")
def load_go(
    spec: DatasetSpec,
    obo_path: Path = Path("data/go-basic.obo"),
    namespace: str = "molecular_function",
) -> LabelledTree:
    """Gene Ontology from a ``go-basic.obo`` file, restricted to one namespace."""
    if not obo_path.exists():
        raise FileNotFoundError(
            f"{obo_path} not found; fetch it from https://purl.obolibrary.org/obo/go/go-basic.obo"
        )
    children_of: dict[str, list[str]] = {}
    roots: list[str] = []
    for block in obo_path.read_text(encoding="utf-8").split("\n\n"):
        if not block.startswith("[Term]"):
            continue
        term: dict[str, Any] = {"is_a": []}
        for line in block.splitlines()[1:]:
            key, _, val = line.partition(": ")
            if key == "id":
                term["id"] = val.strip()
            elif key == "namespace":
                term["namespace"] = val.strip()
            elif key == "is_a":
                term["is_a"].append(val.split("!")[0].strip())
            elif key == "is_obsolete":
                term["obsolete"] = True
        if term.get("namespace") != namespace or term.get("obsolete"):
            continue
        node = term["id"]
        children_of.setdefault(node, [])
        if not term["is_a"]:
            roots.append(node)
        for par in term["is_a"]:
            children_of.setdefault(par, []).append(node)
    if len(roots) != 1:
        raise ValueError(f"expected one namespace root, found {roots}")
    # Drop cross-namespace parents that were never declared as terms here.
    children_of = {k: sorted(v) for k, v in children_of.items()}
    parent = _longest_path_parents(children_of, roots[0])
    parent = {k: v for k, v in parent.items() if k in children_of}
    return finalise(from_parent_map(parent), spec)


@register_dataset("ncbi")
def load_ncbi(
    spec: DatasetSpec,
    taxdump_path: Path = Path("data/taxdump.tar.gz"),
    clade_taxid: int = 40674,
) -> LabelledTree:
    """NCBI taxonomy from ``taxdump.tar.gz``, restricted to a clade (default Mammalia)."""
    if not taxdump_path.exists():
        raise FileNotFoundError(
            f"{taxdump_path} not found; fetch it from "
            "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
        )
    parent_of_all: dict[int, int] = {}
    with tarfile.open(taxdump_path) as tar:
        member = tar.extractfile("nodes.dmp")
        if member is None:
            raise ValueError("nodes.dmp missing from the taxdump archive")
        for raw in member:
            fields = raw.decode("utf-8").split("\t|\t")
            parent_of_all[int(fields[0])] = int(fields[1])
    children_of_all: dict[int, list[int]] = {}
    for child, par in parent_of_all.items():
        if child != par:
            children_of_all.setdefault(par, []).append(child)
    children_of: dict[str, list[str]] = {}
    stack = [clade_taxid]
    while stack:
        u = stack.pop()
        kids = sorted(children_of_all.get(u, []))
        children_of[str(u)] = [str(k) for k in kids]
        stack.extend(kids)
    return finalise(from_parent_map(_longest_path_parents(children_of, str(clade_taxid))), spec)
