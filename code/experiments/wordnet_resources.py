"""Resource counts for the path-state encoding on the full WordNet noun hierarchy.

The paper's headline resource figure is the qubit count for untruncated WordNet, and
getting it right requires care that an earlier draft did not take. The path state of
Theorem C lives in ``C^V`` for a tree whose leaves *all sit at depth n*, so the relevant
node count is the **padded** one. WordNet's noun tree has 74,374 nodes but reaches depth
19 with most leaves far shallower, and padding every shallow leaf to depth 19 adds an
order of magnitude more nodes than the tree started with.

Padding chains are private to a leaf and carry no branching, so all of a leaf's padding
nodes can be merged into a single basis vector holding their summed amplitude
``sqrt(1 - sqrt(f(d)))`` for original depth ``d``. Overlaps with every other leaf are
unchanged, because those nodes were shared with no one. This is verified exactly in
``test_theorem_c_padding_collapse_preserves_the_kernel``.

This script reports all three counts so the paper can quote the honest pair.
"""

from __future__ import annotations

import logging

import numpy as np

from padic_kernel.datasets import _longest_path_parents
from padic_kernel.tree import from_parent_map
from padic_kernel.utils import make_output_dir, write_json

logger = logging.getLogger(__name__)

__all__ = ["wordnet_tree"]


def wordnet_tree(root_name: str = "entity.n.01"):
    """The WordNet noun hierarchy reduced to a tree by the longest hypernym path."""
    import nltk
    from nltk.corpus import wordnet as wn

    try:
        wn.synset(root_name)
    except LookupError:  # pragma: no cover - one-time download path
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    children: dict[str, list[str]] = {}
    stack = [wn.synset(root_name)]
    seen = {root_name}
    while stack:
        syn = stack.pop()
        kids = syn.hyponyms()
        children[syn.name()] = sorted(k.name() for k in kids)
        for k in kids:
            if k.name() not in seen:
                seen.add(k.name())
                stack.append(k)
    return from_parent_map(_longest_path_parents(children, root_name))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tree = wordnet_tree()
    leaves = tree.leaves
    height = tree.height

    unpadded = tree.num_nodes
    pad_nodes = int((height - tree.depth[leaves]).sum())
    padded = unpadded + pad_nodes
    shallow = int((tree.depth[leaves] < height).sum())
    collapsed = unpadded + shallow

    def qubits(d: int) -> int:
        return int(np.ceil(np.log2(d)))

    results = {
        "source_nodes": unpadded,
        "height": height,
        "leaves": int(leaves.shape[0]),
        "shallow_leaves": shallow,
        "padding_nodes": pad_nodes,
        "padded_nodes": padded,
        "collapsed_nodes": collapsed,
        "qubits_unpadded": qubits(unpadded),
        "qubits_padded": qubits(padded),
        "qubits_collapsed": qubits(collapsed),
        "amplitudes": height + 1,
    }
    for key, value in results.items():
        logger.info("  %-18s %s", key, value)

    out = make_output_dir("wordnet_resources")
    write_json(out / "results.json", results)


if __name__ == "__main__":
    main()
