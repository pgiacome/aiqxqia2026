import numpy as np
import pytest

from padic_kernel.encoding import EncodingFactory
from padic_kernel.kernels import depolarise, fidelity_gram, overlap_gram, sample_kernel
from padic_kernel.profiles import geometric_profile
from padic_kernel.tree import build_padic_tree


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("path_state", {"profile": geometric_profile(3, 2, 1.0)}),
        ("angle", {"radix": 2}),
        ("basis", {"radix": 2}),
        ("random_product", {"radix": 2, "local_dim": 3, "seed": 0}),
        ("zz", {"radix": 2, "reps": 2}),
        ("block_product", {"radix": 2, "block_size": 2, "seed": 0}),
    ],
)
def test_states_are_unit_vectors(name, kwargs):
    tree = build_padic_tree(2, 3)
    enc = EncodingFactory(name, **kwargs)
    psi = enc.states(tree, tree.leaves)
    assert psi.shape[0] == len(tree.leaves)
    assert np.allclose(np.linalg.norm(psi, axis=1), 1.0, atol=1e-12)


def test_path_state_dimension_is_node_count():
    tree = build_padic_tree(3, 2)
    enc = EncodingFactory("path_state", profile=geometric_profile(2, 3, 1.0))
    assert enc.states(tree, tree.leaves).shape[1] == tree.num_nodes


def test_angle_encoding_dimension_is_two_to_the_n():
    tree = build_padic_tree(2, 4)
    psi = EncodingFactory("angle", radix=2).states(tree, tree.leaves)
    assert psi.shape[1] == 2**4


def test_basis_encoding_gram_is_identity():
    tree = build_padic_tree(3, 2)
    psi = EncodingFactory("basis", radix=3).states(tree, tree.leaves)
    gram = np.abs(psi.conj() @ psi.T) ** 2
    assert np.allclose(gram, np.eye(len(tree.leaves)), atol=1e-12)


def test_zz_encoding_is_not_a_product_map():
    # A product map's kernel factorises across digits; the ZZ map's does not.
    tree = build_padic_tree(2, 3)
    psi = EncodingFactory("zz", radix=2, reps=2).states(tree, tree.leaves)
    k = np.abs(psi.conj() @ psi.T) ** 2
    # leaves 0b000=0, 0b001=1, 0b010=2, 0b011=3
    assert not np.isclose(k[0, 1] * k[0, 2], k[0, 3] * k[0, 0], atol=1e-9)


def test_unknown_encoding_raises():
    with pytest.raises(KeyError, match="nope"):
        EncodingFactory("nope")


def test_fidelity_gram_diagonal_is_one():
    tree = build_padic_tree(2, 3)
    psi = EncodingFactory("angle", radix=2).states(tree, tree.leaves)
    k = fidelity_gram(psi)
    assert np.allclose(np.diag(k), 1.0, atol=1e-12)
    assert np.allclose(k, k.T, atol=1e-12)
    assert k.min() >= -1e-12 and k.max() <= 1.0 + 1e-12


def test_overlap_gram_is_positive_semidefinite():
    tree = build_padic_tree(2, 3)
    psi = EncodingFactory("path_state", profile=geometric_profile(3, 2, 1.0)).states(
        tree, tree.leaves
    )
    eigs = np.linalg.eigvalsh(overlap_gram(psi))
    assert eigs.min() > -1e-10


def test_sampling_converges_to_the_exact_kernel():
    tree = build_padic_tree(2, 3)
    psi = EncodingFactory("path_state", profile=geometric_profile(3, 2, 1.0)).states(
        tree, tree.leaves
    )
    k = fidelity_gram(psi)
    rng = np.random.default_rng(0)
    est = sample_kernel(k, shots=200_000, rng=rng)
    assert np.max(np.abs(est - k)) < 0.02


def test_depolarise_moves_towards_the_maximally_mixed_value():
    k = np.array([[1.0, 0.2], [0.2, 1.0]])
    # dim = 16: the maximally mixed outcome probability 1/16 sits below K, so the
    # off-diagonal entry must decrease towards it.
    out = depolarise(k, rate=0.5, dim=16)
    assert out[0, 1] < k[0, 1]
    assert out[0, 1] > 1.0 / 16
    assert np.allclose(np.diag(out), 1.0)
    # A kernel entry already below 1/dim is pulled up towards it, not down.
    up = depolarise(np.array([[1.0, 0.01], [0.01, 1.0]]), rate=0.5, dim=16)
    assert 0.01 < up[0, 1] < 1.0 / 16


def test_depolarise_at_full_rate_flattens_the_kernel():
    k = np.array([[1.0, 0.2, 0.7], [0.2, 1.0, 0.4], [0.7, 0.4, 1.0]])
    out = depolarise(k, rate=1.0, dim=8)
    off = out[~np.eye(3, dtype=bool)]
    assert np.allclose(off, 1.0 / 8)


def test_reupload_reduces_to_path_state_at_zero_coupling():
    from padic_kernel.reupload import ReuploadEncoding

    tree = build_padic_tree(2, 3)
    f = geometric_profile(3, 2, 1.0)
    base = EncodingFactory("path_state", profile=f)
    enc = ReuploadEncoding(base=base, layers=3, gamma=0.0, entangle=0.0, seed=0)
    assert np.allclose(
        fidelity_gram(enc.states(tree, tree.leaves)),
        fidelity_gram(base.states(tree, tree.leaves)),
        atol=1e-12,
    )


def test_reupload_distortion_grows_with_gamma():
    from padic_kernel.metrics import strong_triangle_violations
    from padic_kernel.reupload import ReuploadEncoding

    tree = build_padic_tree(2, 3)
    f = geometric_profile(3, 2, 1.0)
    base = EncodingFactory("path_state", profile=f)
    excesses = []
    for gamma in (0.0, 0.3, 0.9):
        enc = ReuploadEncoding(base=base, layers=2, gamma=gamma, entangle=0.5, seed=0)
        _, excess = strong_triangle_violations(1.0 - fidelity_gram(enc.states(tree, tree.leaves)))
        excesses.append(max(excess, 0.0))
    assert excesses[0] <= 1e-12 < excesses[1] < excesses[2]


def test_register_dimension_rounds_up_to_a_power_of_two():
    """The depolarising floor is set by the register, not by the state's support.

    A path state on |V| basis vectors is held in 2**ceil(log2 |V|) amplitudes, and it is
    the register that depolarises. The p-adic tree's node count is essentially never a
    power of two, so the two differ in general.
    """
    from padic_kernel.kernels import register_dimension

    assert register_dimension(63) == 64
    assert register_dimension(64) == 64
    assert register_dimension(65) == 128
    assert register_dimension(1) == 1
    with pytest.raises(ValueError, match="positive"):
        register_dimension(0)

    tree = build_padic_tree(2, 5)
    assert tree.num_nodes == 63
    assert register_dimension(tree.num_nodes) == 64
