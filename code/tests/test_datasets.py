import numpy as np
import pytest

from padic_kernel.datasets import DatasetFactory, DatasetSpec
from padic_kernel.tree import ancestor_matrix, lca_depth_matrix


def test_synthetic_dataset_round_trips():
    spec = DatasetSpec(name="synthetic", max_leaves=64, max_depth=6, seed=0)
    ds = DatasetFactory("synthetic", p=2, n=6)(spec)
    assert len(ds.leaves) == 64
    assert ds.labels.shape == ds.leaves.shape
    assert ds.radix == 2
    assert ds.source_height == 6


def test_subsampling_is_deterministic_under_the_seed():
    spec = DatasetSpec(name="synthetic", max_leaves=32, max_depth=6, seed=7)
    a = DatasetFactory("synthetic", p=2, n=6)(spec)
    b = DatasetFactory("synthetic", p=2, n=6)(spec)
    assert np.array_equal(a.leaves, b.leaves)
    assert np.array_equal(a.labels, b.labels)


def test_all_sampled_leaves_sit_at_the_tree_height():
    spec = DatasetSpec(name="synthetic", max_leaves=16, max_depth=3, seed=1)
    ds = DatasetFactory("synthetic", p=3, n=3)(spec)
    assert np.all(ds.tree.depth[ds.leaves] == ds.tree.height)
    lca = lca_depth_matrix(ancestor_matrix(ds.tree, ds.leaves))
    assert np.all(np.diag(lca) == ds.tree.height)


def test_truncation_bounds_the_height():
    spec = DatasetSpec(name="synthetic", max_leaves=32, max_depth=3, seed=0)
    ds = DatasetFactory("synthetic", p=2, n=8)(spec)
    assert ds.tree.height == 3
    assert ds.source_height == 8


def test_restriction_keeps_lca_depths_of_the_sample():
    """The pipeline must not disturb the geometry it is meant to measure."""
    from padic_kernel.tree import build_padic_tree

    spec = DatasetSpec(name="synthetic", max_leaves=24, max_depth=6, seed=3)
    ds = DatasetFactory("synthetic", p=2, n=6)(spec)
    lca_pipeline = lca_depth_matrix(ancestor_matrix(ds.tree, ds.leaves))
    # Recover which original leaves were sampled and compare against the full tree.
    full = build_padic_tree(2, 6)
    rng = np.random.default_rng(3)
    sampled = np.sort(rng.choice(full.leaves, size=24, replace=False))
    lca_full = lca_depth_matrix(ancestor_matrix(full, sampled))
    assert np.array_equal(lca_pipeline, lca_full)


def test_pipeline_shrinks_the_node_count():
    spec = DatasetSpec(name="synthetic", max_leaves=8, max_depth=8, seed=0)
    ds = DatasetFactory("synthetic", p=2, n=8)(spec)
    assert ds.tree.num_nodes < ds.source_nodes
    assert ds.tree.num_nodes <= 8 * (ds.tree.height + 1)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="nope"):
        DatasetFactory("nope")


@pytest.mark.slow
def test_wordnet_loads_and_is_a_tree():
    spec = DatasetSpec(name="wordnet", max_leaves=200, max_depth=8, seed=0)
    ds = DatasetFactory("wordnet")(spec)
    assert len(ds.leaves) == 200
    assert ds.tree.height == 8
    assert ds.source_height >= 15
    assert ds.source_nodes > 50_000
    assert int((ds.tree.parent < 0).sum()) == 1
    lca = lca_depth_matrix(ancestor_matrix(ds.tree, ds.leaves))
    # LCA depth is a genuine ultrametric height function on the sample.
    m = len(ds.leaves)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        x, y, z = rng.integers(0, m, size=3)
        assert lca[x, z] >= min(lca[x, y], lca[y, z])


@pytest.mark.slow
@pytest.mark.parametrize("name", ["go", "ncbi"])
def test_ontology_datasets_load(name):
    spec = DatasetSpec(name=name, max_leaves=128, max_depth=8, seed=0)
    ds = DatasetFactory(name)(spec)
    assert len(ds.leaves) == 128
    assert ds.tree.height == 8
    assert len(np.unique(ds.labels)) >= 2
