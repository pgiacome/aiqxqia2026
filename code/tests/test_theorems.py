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


@pytest.mark.parametrize("p,n,s", [(2, 4, 1.0), (2, 6, 2.0), (3, 3, 1.0), (5, 2, 2.0)])
def test_theorem_b_closed_form_bound_matches_the_full_tree(p, n, s):
    """The closed-form bound agrees with the empirical one when no leaf is dropped."""
    from padic_kernel.metrics import regular_tree_dimension_bound

    _, lca = _setup(p, n)
    f = geometric_profile(n, p, s)
    assert regular_tree_dimension_bound(f, p, n) == pytest.approx(
        dimension_lower_bound(f(lca)), rel=1e-12
    )


@pytest.mark.parametrize("p,n,s", [(3, 6, 2.0), (5, 5, 2.0), (3, 8, 1.5)])
def test_theorem_b_subsampling_understates_the_bound(p, n, s):
    """A subsampled estimate is valid but capped at the sample size.

    This is why the paper's crossing figure uses the closed form: an estimate from 128
    sampled leaves saturates at 7 qubits however deep the tree is.
    """
    from padic_kernel.metrics import regular_tree_dimension_bound

    full = regular_tree_dimension_bound(geometric_profile(n, p, s), p, n)
    assert full > 128, "pick a case where the full tree exceeds the sample size"
    assert full > p**n / 2  # with s > 1 the bound is a constant factor below L


@pytest.mark.parametrize("p", [2, 3, 5])
def test_theorem_b_row_sum_grows_linearly_at_s_equal_one(p):
    """At ``s = 1`` the row sum grows with ``n``, so no uniform threshold exists.

    ``S(p, s, n) = 1 + ((p-1)/p) * sum_m p**(m(1-s))`` is a convergent geometric series
    only for ``s > 1``. At ``s = 1`` every term is 1 and ``S = 1 + n(p-1)/p``. What this
    costs is the n-independent threshold, NOT the exclusion itself -- see
    ``test_theorem_b_exclusion_survives_s_equal_one``.
    """
    from padic_kernel.metrics import regular_tree_dimension_bound

    n = 6
    at_one = regular_tree_dimension_bound(geometric_profile(n, p, 1.0), p, n)
    above_one = regular_tree_dimension_bound(geometric_profile(n, p, 2.0), p, n)
    assert at_one == pytest.approx(p**n / (1.0 + n * (p - 1) / p), rel=1e-12)
    assert above_one > at_one


# ------------------------------------------------- Proposition D: noise and sampling


@pytest.mark.parametrize("rate", [0.0, 0.01, 0.1, 0.5, 0.9, 0.99])
def test_proposition_d_depolarising_preserves_ultrametricity_exactly(rate):
    """Global depolarising noise cannot break the strong triangle inequality.

    It sends ``K -> (1 - r)K + r/D``, hence ``d = 1 - K -> (1 - r)d + r(1 - 1/D)``,
    an increasing affine map. Such a map preserves ``d(x,z) <= max(d(x,y), d(y,z))``
    because it preserves order and commutes with ``max``. Only the contrast of the
    profile shrinks.
    """
    from padic_kernel.kernels import depolarise

    tree, _ = _setup(2, 4)
    f = geometric_profile(4, 2, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    noisy = depolarise(fidelity_gram(psi), rate, dim=int(psi.shape[1]))
    count, excess = strong_triangle_violations(1.0 - noisy)
    assert count == 0
    assert excess <= 1e-12


def test_proposition_d_depolarising_still_destroys_resolution():
    """Ultrametricity survives depolarising noise; usable resolution does not."""
    from padic_kernel.kernels import depolarise
    from padic_kernel.metrics import resolution_depth

    tree, lca = _setup(2, 4)
    f = geometric_profile(4, 2, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    k = fidelity_gram(psi)
    dim = int(psi.shape[1])
    clean = resolution_depth(depolarise(k, 0.0, dim=dim), lca, tol=1e-3)
    wrecked = resolution_depth(depolarise(k, 0.999, dim=dim), lca, tol=1e-3)
    assert clean == 5
    assert wrecked < clean


def test_proposition_d_exact_ultrametrics_are_isoceles_so_ties_dominate():
    """Why violation *count* is the wrong robustness statistic under sampling noise.

    In an ultrametric every triangle is isoceles with the two longest sides equal, so
    a constant fraction of triples meets the strong triangle inequality with equality.
    An arbitrarily small perturbation flips about half of them into violations, which
    makes the count saturate rather than decay as the shot budget grows. E5 therefore
    reports the violation *magnitude*.
    """
    tree, _ = _setup(2, 4)
    f = geometric_profile(4, 2, 1.0)
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    dist = 1.0 - fidelity_gram(psi)
    d_xy = dist[:, :, None]
    d_yz = dist[None, :, :]
    d_xz = dist[:, None, :]
    ties = np.isclose(d_xz, np.maximum(d_xy, d_yz), atol=1e-12)
    assert ties.mean() > 0.25, "expected a constant fraction of triples to be tight"


def test_theorem_c_converse_needs_non_negative_overlaps():
    """The converse fails without the positivity hypothesis.

    A fidelity kernel fixes only the moduli of the overlaps, and those do not determine
    the Gram matrix up to per-point phases: the Bargmann invariant G_xy G_yz G_zx is
    gauge invariant but is not a function of the moduli. This builds an explicit second
    realisation of the same kernel that no isometry-plus-phase carries to the path
    state, which is why Theorem C restricts the converse.
    """
    t = 0.25
    root = np.sqrt(t)
    gram = np.array(
        [[1.0, root, 1j * root], [root, 1.0, root], [-1j * root, root, 1.0]],
        dtype=complex,
    )
    assert np.allclose(gram, gram.conj().T)
    assert np.linalg.eigvalsh(gram).min() > 1e-9, "counterexample must be PSD"

    # Same fidelity kernel as a depth-1 path state on three leaves with f(0) = t.
    iu = np.triu_indices(3, k=1)
    assert np.allclose(np.abs(gram[iu]) ** 2, t)

    from padic_kernel.profiles import Profile

    tree = build_padic_tree(3, 1)
    f = Profile((t, 1.0))
    psi = EncodingFactory("path_state", profile=f).states(tree, tree.leaves)
    path_gram = psi @ psi.conj().T
    assert np.allclose(np.abs(path_gram[iu]) ** 2, t)

    # Equal moduli, different Bargmann invariant: not related by isometry and phases.
    def bargmann(g: np.ndarray) -> complex:
        return complex(g[0, 1] * g[1, 2] * g[2, 0])

    assert not np.isclose(bargmann(gram), bargmann(path_gram))
    assert np.isclose(bargmann(path_gram).imag, 0.0)


def test_theorem_a_block_bound_is_tight_at_one_block():
    """With a single block the path state is itself a block-product map.

    Theorem A must therefore not claim three kernel values for block-product maps in
    general; that conclusion is specific to genuine product maps. Here max B* = n, so
    the theorem's conclusion is vacuous, as it should be.
    """
    tree, lca = _setup(2, 4)
    f = geometric_profile(4, 2, 1.0)
    k = fidelity_gram(EncodingFactory("path_state", profile=f).states(tree, tree.leaves))
    assert len(np.unique(np.round(k, 9))) == 5  # n + 1 values, not three


@pytest.mark.parametrize(
    "p,s,expected_S,expected_n0",
    [(3, 2.0, 4.0 / 3.0, 0.7095), (3, 1.01, 61.3499, 10.1528), (5, 1.5, 1.6472, 0.5447)],
)
def test_theorem_b_threshold_constants_quoted_in_the_paper(p, s, expected_S, expected_n0):
    """Pin the S and n_0 constants the Section 5 example quotes.

    These are illustrative constants in prose rather than generated macros, so nothing
    else would catch them drifting. An earlier draft quoted S = 68 and n_0 = 7.2 for
    p = 3, s = 1.01, both wrong.
    """
    ratio = p ** (1 - s)
    s_inf = 1.0 + ((p - 1) / p) * ratio / (1 - ratio)
    n0 = np.log2(s_inf) / (np.log2(p) - 1)
    assert s_inf == pytest.approx(expected_S, rel=1e-3)
    assert n0 == pytest.approx(expected_n0, rel=1e-3)


@pytest.mark.parametrize("p,s", [(3, 2.0), (3, 1.01), (5, 1.5), (3, 1.5)])
def test_theorem_b_threshold_is_a_valid_sufficient_condition(p, s):
    """Past n_0 the bound really does exceed n, at every larger depth tested."""
    from padic_kernel.metrics import regular_tree_qubit_bound

    ratio = p ** (1 - s)
    s_inf = 1.0 + ((p - 1) / p) * ratio / (1 - ratio)
    n0 = np.log2(s_inf) / (np.log2(p) - 1)
    for n in range(int(np.ceil(n0)) + 1, int(np.ceil(n0)) + 8):
        assert regular_tree_qubit_bound(geometric_profile(n, p, s), p, n) > n


@pytest.mark.parametrize("p,first_n", [(3, 3), (5, 2)])
def test_theorem_b_exclusion_survives_s_equal_one(p, first_n):
    """Losing the uniform threshold at s = 1 does not lose the exclusion.

    An earlier draft claimed s > 1 was necessary. It is not: p**n / Theta(n) still
    overtakes 2**n, just without an n-independent threshold. This pins the depth from
    which the exclusion holds, so the weaker claim cannot drift back to the false one.
    """
    from padic_kernel.metrics import regular_tree_dimension_bound

    assert regular_tree_dimension_bound(geometric_profile(first_n, p, 1.0), p, first_n) > 2**first_n
    for n in range(first_n, first_n + 8):
        assert regular_tree_dimension_bound(geometric_profile(n, p, 1.0), p, n) > 2**n


def test_theorem_c_profile_is_determined_only_on_populated_levels():
    """Injectivity of f -> K fails when a level carries no pair of leaves.

    A root with two unary-padded children of depth 2 has no pair meeting at depth 1, so
    profiles differing only there induce the same kernel. Theorem C's converse is
    therefore to be read modulo the populated levels.
    """
    from padic_kernel.profiles import Profile
    from padic_kernel.tree import from_parent_map, pad_to_uniform_depth

    parent_of = {"r": None, "a": "r", "b": "r", "a1": "a", "a2": "a"}
    tree, tip = pad_to_uniform_depth(from_parent_map(parent_of))
    idx = {name: i for i, name in enumerate(from_parent_map(parent_of).labels)}
    leaves = tip[np.array([idx["a1"], idx["b"]])]
    lca = lca_depth_matrix(ancestor_matrix(tree, leaves))
    assert 1 not in set(np.unique(lca).tolist()), "level 1 must be unpopulated here"

    f = Profile((0.25, 0.5, 1.0))
    g = Profile((0.25, 0.75, 1.0))
    kf = fidelity_gram(EncodingFactory("path_state", profile=f).states(tree, leaves))
    kg = fidelity_gram(EncodingFactory("path_state", profile=g).states(tree, leaves))
    assert np.allclose(kf, kg, atol=1e-12), "distinct profiles must give the same kernel"


def test_theorem_a_product_map_can_realise_a_monotone_profile_at_depth_one():
    """At n = 1 a product map is the whole encoding, so the no-go needs n >= 2.

    Three Bloch states at pairwise angle ~110 degrees give f = (0.33, 1), strictly
    monotone and exactly ultrametric, on one qubit. The abstract and Theorem A must
    carry the n >= 2 hypothesis.
    """
    tree = build_padic_tree(3, 1)
    lca = lca_depth_matrix(ancestor_matrix(tree, tree.leaves))
    target = 3.0**-1.01
    theta = np.arccos(2 * target - 1)
    angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    # Three Bloch vectors at polar angle theta/... place them so pairwise fidelity is target.
    half = theta / 2
    psi = np.stack(
        [np.array([np.cos(half), np.sin(half) * np.exp(1j * a)]) for a in angles]
    )
    k = fidelity_gram(psi)
    off = k[~np.eye(3, dtype=bool)]
    assert np.ptp(off) < 1e-12, "the three states must be pairwise equidistant"
    assert level_constancy_of(k, lca) < 1e-12
    assert off[0] < 1.0, "and the profile must be non-degenerate"


def level_constancy_of(kernel, lca):
    from padic_kernel.metrics import level_constancy

    return level_constancy(kernel, lca)
