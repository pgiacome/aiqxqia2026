import numpy as np
import pytest

from padic_kernel.metrics import (
    dimension_lower_bound,
    gromov_delta,
    kernel_target_alignment,
    profile_residual,
    qubit_lower_bound,
    strong_triangle_violations,
    violation_rate,
)
from padic_kernel.profiles import geometric_profile
from padic_kernel.tree import ancestor_matrix, build_padic_tree, lca_depth_matrix


def _ultrametric_distance(p=2, n=3, s=1.0):
    tree = build_padic_tree(p, n)
    lca = lca_depth_matrix(ancestor_matrix(tree, tree.leaves))
    f = geometric_profile(n, p, s)
    return 1.0 - f(lca), lca, f


def test_true_ultrametric_has_no_violations():
    dist, _, _ = _ultrametric_distance()
    count, excess = strong_triangle_violations(dist)
    assert count == 0
    assert excess <= 0.0
    assert violation_rate(dist) == 0.0


def test_euclidean_distance_violates_the_strong_triangle():
    x = np.arange(8, dtype=float)[:, None]
    dist = np.abs(x - x.T)
    count, excess = strong_triangle_violations(dist)
    assert count > 0
    assert excess > 0.0


def test_gromov_delta_is_zero_for_an_ultrametric():
    dist, _, _ = _ultrametric_distance()
    assert gromov_delta(dist) == pytest.approx(0.0, abs=1e-12)


def test_dimension_lower_bound_matches_the_closed_form():
    # For the delta kernel K = I on L points, sum K = L, so the bound is exactly L.
    k = np.eye(16)
    assert dimension_lower_bound(k) == pytest.approx(16.0)
    assert qubit_lower_bound(k) == pytest.approx(4.0)


def test_dimension_lower_bound_is_one_for_the_constant_kernel():
    k = np.ones((16, 16))
    assert dimension_lower_bound(k) == pytest.approx(1.0)


def test_profile_residual_is_zero_for_the_exact_construction():
    from padic_kernel.encoding import EncodingFactory
    from padic_kernel.kernels import fidelity_gram

    tree = build_padic_tree(2, 3)
    lca = lca_depth_matrix(ancestor_matrix(tree, tree.leaves))
    f = geometric_profile(3, 2, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    assert profile_residual(fidelity_gram(psi), lca, f) < 1e-12


def test_alignment_is_one_for_a_perfectly_matched_kernel():
    labels = np.array([0, 0, 1, 1])
    target = (labels[:, None] == labels[None, :]).astype(float)
    assert kernel_target_alignment(target, labels) == pytest.approx(1.0)


def test_violation_scan_refuses_oversized_inputs():
    with pytest.raises(ValueError, match="512"):
        strong_triangle_violations(np.zeros((600, 600)))


def _kernel_for(name, p=2, n=4, **kwargs):
    from padic_kernel.encoding import EncodingFactory
    from padic_kernel.kernels import fidelity_gram

    tree = build_padic_tree(p, n)
    lca = lca_depth_matrix(ancestor_matrix(tree, tree.leaves))
    return fidelity_gram(EncodingFactory(name, **kwargs).states(tree, tree.leaves)), lca


def test_level_constancy_is_zero_for_the_exact_construction():
    from padic_kernel.metrics import level_constancy

    k, lca = _kernel_for("path_state", profile=geometric_profile(4, 2, 1.0))
    assert level_constancy(k, lca) < 1e-12


def test_level_constancy_is_zero_for_basis_encoding_too():
    """The delta kernel is ultrametric -- degenerately so.

    This is why the paper reports resolution depth alongside violations: a basis
    encoding passes every ultrametricity check while collapsing the hierarchy.
    """
    from padic_kernel.metrics import level_constancy, resolution_depth

    k, lca = _kernel_for("basis", radix=2)
    assert level_constancy(k, lca) < 1e-12
    assert resolution_depth(k, lca) == 2


def test_level_constancy_is_positive_for_product_and_zz_maps():
    from padic_kernel.metrics import level_constancy

    for name, kwargs in (
        ("angle", {"radix": 2}),
        ("zz", {"radix": 2}),
        ("random_product", {"radix": 2, "local_dim": 3, "seed": 0}),
    ):
        k, lca = _kernel_for(name, **kwargs)
        assert level_constancy(k, lca) > 1e-6, name


def test_resolution_depth_is_full_for_the_exact_construction():
    from padic_kernel.metrics import resolution_depth

    k, lca = _kernel_for("path_state", n=4, profile=geometric_profile(4, 2, 1.0))
    assert resolution_depth(k, lca) == 5
