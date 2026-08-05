import numpy as np
import pytest

from padic_kernel.tree import (
    ancestor_matrix,
    build_padic_tree,
    digit_matrix,
    from_parent_map,
    lca_depth_matrix,
    pad_to_uniform_depth,
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
    padded = pad_to_uniform_depth(t)
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
