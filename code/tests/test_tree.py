import numpy as np
import pytest

from padic_kernel.tree import (
    ancestor_matrix,
    build_padic_tree,
    digit_matrix,
    from_parent_map,
    lca_depth_matrix,
    max_branching,
    pad_to_uniform_depth,
    restrict_to_leaves,
    truncate_at_depth,
)


def test_padic_tree_shape():
    t = build_padic_tree(p=3, n=2)
    # 1 root + 3 + 9 nodes
    assert t.num_nodes == 13
    assert t.height == 2
    assert len(t.leaves) == 9


def test_lca_depth_equals_p_adic_valuation():
    p, n = 3, 3
    t = build_padic_tree(p, n)
    anc = ancestor_matrix(t, t.leaves)
    lam = lca_depth_matrix(anc)
    for x in range(p**n):
        for y in range(p**n):
            d = x - y
            if d == 0:
                expected = n
            else:
                v = 0
                while d % p == 0:
                    d //= p
                    v += 1
                expected = min(v, n)
            assert lam[x, y] == expected, (x, y)


def test_digits_are_base_p_least_significant_first():
    p, n = 5, 2
    t = build_padic_tree(p, n)
    dig = digit_matrix(t, t.leaves)
    for x in range(p**n):
        assert dig[x, 0] == x % p
        assert dig[x, 1] == (x // p) % p


def test_from_parent_map_and_padding():
    #    root -> a -> a1, a2 ; root -> b   (b is a shallow leaf)
    parent_of = {"root": None, "a": "root", "b": "root", "a1": "a", "a2": "a"}
    t = from_parent_map(parent_of)
    assert t.height == 2
    padded, _tip = pad_to_uniform_depth(t)
    leaf_depths = padded.depth[padded.leaves]
    assert np.all(leaf_depths == padded.height)
    anc = ancestor_matrix(padded, padded.leaves)
    lam = lca_depth_matrix(anc)
    assert np.all(np.diag(lam) == padded.height)


def test_lca_depth_satisfies_strong_triangle():
    t = build_padic_tree(p=2, n=4)
    lam = lca_depth_matrix(ancestor_matrix(t, t.leaves))
    m = len(t.leaves)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        x, y, z = rng.integers(0, m, size=3)
        assert lam[x, z] >= min(lam[x, y], lam[y, z])


def test_rejects_forest():
    with pytest.raises(ValueError, match="exactly one root"):
        from_parent_map({"a": None, "b": None})


def test_truncate_at_depth_makes_deep_nodes_leaves():
    t = build_padic_tree(p=2, n=5)
    trunc = truncate_at_depth(t, 3)
    assert trunc.height == 3
    assert len(trunc.leaves) == 2**3
    assert trunc.num_nodes == 2**4 - 1


def test_truncation_preserves_lca_depth_up_to_the_cap():
    p, n, cap = 3, 4, 2
    full = build_padic_tree(p, n)
    trunc = truncate_at_depth(full, cap)
    lam = lca_depth_matrix(ancestor_matrix(trunc, trunc.leaves))
    # Truncated leaf r at depth cap represents the residue class r mod p**cap.
    for x in range(p**cap):
        for y in range(p**cap):
            d = x - y
            v = cap
            if d != 0:
                v = 0
                while d % p == 0:
                    d //= p
                    v += 1
                v = min(v, cap)
            assert lam[x, y] == v, (x, y)


def test_restrict_to_leaves_shrinks_the_tree_but_keeps_lca_depths():
    t = build_padic_tree(p=2, n=6)
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(t.leaves, size=8, replace=False))
    before = lca_depth_matrix(ancestor_matrix(t, sample))
    sub, remapped = restrict_to_leaves(t, sample)
    after = lca_depth_matrix(ancestor_matrix(sub, remapped))
    assert np.array_equal(before, after)
    assert sub.num_nodes < t.num_nodes
    assert sub.height == t.height


def test_restrict_then_pad_handles_mixed_leaf_depths():
    parent_of = {"r": None, "a": "r", "b": "r", "a1": "a", "a2": "a"}
    t = from_parent_map(parent_of)
    idx = {name: i for i, name in enumerate(t.labels)}
    sample = np.array([idx["a1"], idx["b"]])
    sub, remapped = restrict_to_leaves(t, sample)
    padded, tip = pad_to_uniform_depth(sub)
    anc = ancestor_matrix(padded, tip[remapped])
    lam = lca_depth_matrix(anc)
    assert lam[0, 1] == 0  # a1 and b share only the root
    assert np.all(np.diag(lam) == padded.height)


def test_max_branching():
    assert max_branching(build_padic_tree(4, 2)) == 4
    assert max_branching(build_padic_tree(2, 5)) == 2
