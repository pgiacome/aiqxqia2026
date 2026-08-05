"""Numerical verification of Theorems A, B and C.

The paper cites these test names directly. A failure here means either the library
or a theorem statement is wrong; it must never be silenced.
"""

from __future__ import annotations

import numpy as np
import pytest

from padic_kernel.encoding import EncodingFactory, _kron_rows
from padic_kernel.kernels import fidelity_gram
from padic_kernel.metrics import (
    dimension_lower_bound,
    profile_residual,
    strong_triangle_violations,
)
from padic_kernel.profiles import geometric_profile, linear_profile
from padic_kernel.tree import (
    ancestor_matrix,
    build_padic_tree,
    digit_matrix,
    lca_depth_matrix,
)

CASES = [(2, 3), (3, 2), (2, 4), (5, 2)]


def _setup(p: int, n: int):
    tree = build_padic_tree(p, n)
    lca = lca_depth_matrix(ancestor_matrix(tree, tree.leaves))
    return tree, lca


def _simplex_factor(radix: int, t: float) -> np.ndarray:
    """``radix`` unit vectors with equal pairwise overlap ``(1 - t) - t / (radix - 1)``.

    Built as ``sqrt(1 - t) |phi> + sqrt(t) |s_a>`` where the ``|s_a>`` are the vertices
    of a regular simplex orthogonal to ``|phi>``. Equal pairwise overlap is exactly
    the condition for the induced kernel to be level-constant, i.e. ultrametric.
    """
    eye = np.eye(radix)
    simplex = eye - eye.mean(axis=0, keepdims=True)
    simplex /= np.linalg.norm(simplex, axis=1, keepdims=True)
    phi = np.zeros(radix + 1)
    phi[radix] = 1.0
    padded = np.concatenate([simplex, np.zeros((radix, 1))], axis=1)
    return (np.sqrt(1.0 - t) * phi[None, :] + np.sqrt(t) * padded).astype(np.complex128)


def _extremal_product_states(tree, leaves, radix: int, t: float, active_levels: int):
    """A product map whose first ``active_levels`` factors are non-trivial.

    Deeper factors are the trivial one-dimensional state, which is precisely the
    structure Theorem A forces on any ultrametric product map.
    """
    dig = digit_matrix(tree, leaves)
    table = _simplex_factor(radix, t)
    factors = []
    for i in range(dig.shape[1]):
        if i < active_levels:
            factors.append(table[dig[:, i]])
        else:
            factors.append(np.ones((dig.shape[0], 1), dtype=np.complex128))
    return _kron_rows(factors)


# --------------------------------------------------------------------------- Theorem A


@pytest.mark.parametrize("p,n", CASES)
@pytest.mark.parametrize("seed", range(5))
def test_theorem_a_random_product_maps_are_never_strictly_monotone(p, n, seed):
    """No product feature map induces a strictly monotone ultrametric kernel."""
    tree, lca = _setup(p, n)
    psi = EncodingFactory("random_product", radix=p, local_dim=p + 2, seed=seed).states(
        tree, tree.leaves
    )
    k = fidelity_gram(psi)
    # If K were ultrametric with profile f, K would be constant on each LCA level.
    spreads = [np.ptp(k[lca == v]) for v in range(n + 1) if np.any(lca == v)]
    assert max(spreads) > 1e-6, "a random product map came out level-constant"


@pytest.mark.parametrize("p,n", [(2, 3), (3, 3), (5, 2)])
def test_theorem_a_ultrametric_product_map_resolves_only_one_level(p, n):
    """The extremal object Theorem A predicts: exactly two kernel values.

    With only the level-1 factor non-trivial, the kernel is ultrametric and its profile
    is ``f = (f(0), 1, 1, ..., 1)`` -- the tree is resolved only to depth 1.
    """
    tree, lca = _setup(p, n)
    k = fidelity_gram(_extremal_product_states(tree, tree.leaves, p, t=0.3, active_levels=1))
    distinct = np.unique(np.round(k, 9))
    assert len(distinct) == 2, f"expected two kernel values, got {distinct}"
    assert np.allclose(k[lca >= 1], 1.0, atol=1e-9)
    assert np.ptp(k[lca == 0]) < 1e-9  # level-constant, hence ultrametric
    assert k[lca == 0][0] < 1.0  # but only the first level carries information


@pytest.mark.parametrize("p,n", [(2, 4), (3, 3)])
def test_theorem_a_a_nontrivial_deep_factor_destroys_ultrametricity(p, n):
    """The proof's mechanism, exercised directly.

    Turning on a second non-trivial factor multiplies the kernel by a digit-dependent
    term at fixed LCA depth, so the kernel stops being level-constant. This is exactly
    the contradiction the proof derives.
    """
    tree, lca = _setup(p, n)
    k = fidelity_gram(_extremal_product_states(tree, tree.leaves, p, t=0.3, active_levels=2))
    assert np.ptp(k[lca == 0]) > 1e-6, "deep factor left the kernel level-constant"


@pytest.mark.parametrize("block_size", [1, 2])
def test_theorem_a_block_product_maps_are_not_ultrametric(block_size):
    """Block-product maps with a non-trivial deep block fail level-constancy.

    Theorem A's block generalisation says an ultrametric block-product map must have
    every block beyond the one containing level ``v* + 1`` trivial. A generic block
    map has no trivial blocks, so it cannot be ultrametric.
    """
    p, n = 2, 4
    tree, lca = _setup(p, n)
    psi = EncodingFactory("block_product", radix=p, block_size=block_size, seed=0).states(
        tree, tree.leaves
    )
    k = fidelity_gram(psi)
    deep = [v for v in range(block_size + 1, n) if np.any(lca == v)]
    spreads = [np.ptp(k[lca == v]) for v in deep]
    assert max(spreads) > 1e-6


# --------------------------------------------------------------------------- Theorem B


@pytest.mark.parametrize("p,n", [(3, 2), (3, 3), (3, 4), (5, 2)])
def test_theorem_b_rules_out_every_n_qubit_encoding(p, n):
    """For p >= 3 the dimension bound exceeds 2**n, so no n-qubit map can work."""
    _, lca = _setup(p, n)
    f = geometric_profile(n, p, s=2.0)
    bound = dimension_lower_bound(f(lca))
    assert bound > 2**n, f"bound {bound} did not exceed 2**{n}"


def test_theorem_b_does_not_subsume_theorem_a_at_radix_two():
    """At p = 2 the bound falls below 2**n, so Theorem A is doing real work.

    This is stated explicitly in the paper: the two theorems cover different regimes
    and neither implies the other.
    """
    for n in (3, 4, 5, 6):
        _, lca = _setup(2, n)
        f = geometric_profile(n, 2, s=2.0)
        assert dimension_lower_bound(f(lca)) < 2**n


@pytest.mark.parametrize("p,n", CASES)
def test_theorem_b_bound_is_satisfied_by_the_exact_construction(p, n):
    """The path-state dimension |V| respects its own lower bound."""
    tree, lca = _setup(p, n)
    f = geometric_profile(n, p, s=2.0)
    assert tree.num_nodes >= dimension_lower_bound(f(lca)) - 1e-9


@pytest.mark.parametrize("p,n", [(3, 2), (3, 3)])
def test_theorem_b_zz_feature_map_is_below_the_bound(p, n):
    """The ZZ map lives in 2**n dimensions, which the bound forbids."""
    tree, lca = _setup(p, n)
    f = geometric_profile(n, p, s=2.0)
    psi = EncodingFactory("zz", radix=p, reps=2).states(tree, tree.leaves)
    assert psi.shape[1] < dimension_lower_bound(f(lca))
    assert profile_residual(fidelity_gram(psi), lca, f) > 1e-3


@pytest.mark.parametrize("p,s", [(3, 2.0), (2, 2.0), (5, 1.5)])
def test_theorem_b_closed_form_row_sum(p, s):
    """S(p, s, n) = 1 + ((p-1)/p) * sum_{m=1..n} p**(m(1-s)) matches the row sum."""
    n = 4 if p < 5 else 3
    _, lca = _setup(p, n)
    f = geometric_profile(n, p, s)
    empirical = float(f(lca)[0].sum())
    closed = 1.0 + ((p - 1) / p) * sum(p ** (m * (1 - s)) for m in range(1, n + 1))
    assert empirical == pytest.approx(closed, rel=1e-12)


def test_theorem_b_gram_rank_argument_is_tight_for_the_delta_kernel():
    """The bound is attained exactly when K = I, where the true minimum dimension is L."""
    assert dimension_lower_bound(np.eye(32)) == pytest.approx(32.0)


# --------------------------------------------------------------------------- Theorem C


@pytest.mark.parametrize("p,n", CASES)
@pytest.mark.parametrize("profile_name", ["geometric", "linear"])
def test_theorem_c_path_states_realise_the_profile_exactly(p, n, profile_name):
    tree, lca = _setup(p, n)
    f = geometric_profile(n, p, 1.0) if profile_name == "geometric" else linear_profile(n)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    residual = profile_residual(fidelity_gram(psi), lca, f)
    assert residual < 1e-12, f"residual {residual:.3e}"


@pytest.mark.parametrize("p,n", CASES)
def test_theorem_c_induced_distance_is_exactly_ultrametric(p, n):
    tree, _ = _setup(p, n)
    if len(tree.leaves) > 256:
        pytest.skip("triple scan limited to 256 leaves for runtime")
    f = geometric_profile(n, p, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    count, excess = strong_triangle_violations(1.0 - fidelity_gram(psi))
    assert count == 0
    assert excess <= 1e-12


def test_theorem_c_kernel_is_psd_via_the_hadamard_square_factorisation():
    """K = (A A^T) o 2 is PSD because the Schur product of PSD matrices is PSD."""
    p, n = 2, 4
    tree, _ = _setup(p, n)
    f = geometric_profile(n, p, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    assert np.linalg.eigvalsh(fidelity_gram(psi)).min() > -1e-10


def test_theorem_c_uniqueness_up_to_isometry_and_phase():
    """Any unitary rotation and per-point phase leaves the kernel unchanged.

    This is why the converse is uniqueness up to a global isometry and per-point
    phases, not up to a relabelling of the node set.
    """
    p, n = 2, 3
    tree, _ = _setup(p, n)
    f = geometric_profile(n, p, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    rng = np.random.default_rng(0)
    dim = psi.shape[1]
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(a)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=psi.shape[0]))[:, None]
    assert np.allclose(fidelity_gram(psi @ q.T * phases), fidelity_gram(psi), atol=1e-12)


def test_theorem_c_generalises_to_a_non_homogeneous_tree():
    """The construction needs only nesting of ancestors, not a regular branching factor."""
    from padic_kernel.profiles import Profile
    from padic_kernel.tree import from_parent_map, pad_to_uniform_depth

    parent_of = {
        "r": None,
        "a": "r", "b": "r", "c": "r",
        "a1": "a", "a2": "a", "a3": "a",
        "b1": "b",
        "c1": "c", "c2": "c",
    }
    tree, _tip = pad_to_uniform_depth(from_parent_map(parent_of))
    lca = lca_depth_matrix(ancestor_matrix(tree, tree.leaves))
    f = Profile((0.1, 0.4, 1.0))
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    assert profile_residual(fidelity_gram(psi), lca, f) < 1e-12
    count, _ = strong_triangle_violations(1.0 - fidelity_gram(psi))
    assert count == 0


def test_theorem_c_amplitudes_are_forced_by_the_profile():
    """Strict monotonicity determines a_i^2 = sqrt(f(i)) - sqrt(f(i-1)) uniquely."""
    f = geometric_profile(5, 3, 1.5)
    a = f.amplitudes()
    root = np.sqrt(np.asarray(f.values))
    assert np.allclose(a**2, np.diff(root, prepend=0.0), atol=1e-15)
    assert np.all(a**2 > 0)
    assert np.sum(a**2) == pytest.approx(1.0, abs=1e-15)


# --------------------------------------------------------------- cross-check vs baselines


@pytest.mark.parametrize("name,kwargs", [("angle", {"radix": 2}), ("zz", {"radix": 2})])
def test_baselines_violate_the_strong_triangle_inequality(name, kwargs):
    tree, _ = _setup(2, 3)
    psi = EncodingFactory(name, **kwargs).states(tree, tree.leaves)
    count, _ = strong_triangle_violations(1.0 - fidelity_gram(psi))
    assert count > 0
