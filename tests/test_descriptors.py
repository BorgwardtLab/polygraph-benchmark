import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_array

from graph_gen_gym.utils.graph_descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    EigenvalueHistogram,
    NormalizedDescriptor,
    OrbitCounts,
    SparseDegreeHistogram,
    WeisfeilerLehmanDescriptor,
)


@pytest.fixture
def sample_molecular_graphs():
    g1 = nx.Graph()
    g1.add_nodes_from(
        [(0, {"element": "C"}), (1, {"element": "O"}), (2, {"element": "N"})]
    )
    g1.add_edges_from(
        [(0, 1, {"bond_type": "single"}), (1, 2, {"bond_type": "double"})]
    )

    g2 = nx.Graph()
    g2.add_nodes_from(
        [(0, {"element": "C"}), (1, {"element": "C"}), (2, {"element": "O"})]
    )
    g2.add_edges_from(
        [(0, 1, {"bond_type": "single"}), (1, 2, {"bond_type": "single"})]
    )

    return [g1, g2]


def test_degree_histogram(sample_graphs):
    max_degree = 10
    descriptor = DegreeHistogram(max_degree)

    features = descriptor(sample_graphs)

    assert features.shape == (len(sample_graphs), max_degree)
    assert np.allclose(features.sum(axis=1), 1.0)

    for i, graph in enumerate(sample_graphs):
        degrees = list(dict(graph.degree()).values())
        for degree in degrees:
            assert features[i, degree] > 0


def test_sparse_degree_histogram(sample_graphs):
    descriptor = SparseDegreeHistogram()

    features = descriptor(sample_graphs)

    assert isinstance(features, csr_array)
    assert features.shape[0] == len(sample_graphs)

    for i, graph in enumerate(sample_graphs):
        degrees = list(dict(graph.degree()).values())
        unique_degrees = set(degrees)
        dense_row = features.toarray()[i].flatten()

        for degree in unique_degrees:
            assert dense_row[degree] > 0

        nonzero_sum = sum(dense_row[degree] for degree in unique_degrees)
        assert np.isclose(nonzero_sum, 1.0)


def test_clustering_histogram(sample_graphs):
    bins = 10
    descriptor = ClusteringHistogram(bins)

    features = descriptor(sample_graphs)

    assert features.shape == (len(sample_graphs), bins)
    assert np.allclose(features.sum(axis=1), 1.0)

    assert np.all(features >= 0)
    assert np.all(features <= 1)


def test_orbit_counts(sample_graphs):
    descriptor = OrbitCounts()

    features = descriptor(sample_graphs)

    assert features.shape[0] == len(sample_graphs)
    assert features.shape[1] > 0

    assert np.all(features >= 0)
    assert np.any(features > 0)


def test_eigenvalue_histogram(sample_graphs):
    descriptor = EigenvalueHistogram()

    features = descriptor(sample_graphs)

    assert features.shape == (len(sample_graphs), 200)
    assert np.allclose(features.sum(axis=1), 1.0)

    assert np.all(features >= 0)
    assert np.all(features <= 1)


def test_normalized_descriptor(sample_graphs):
    base_descriptor = DegreeHistogram(max_degree=10)

    normalized_descriptor = NormalizedDescriptor(base_descriptor, sample_graphs)

    features = normalized_descriptor(sample_graphs)

    assert features.shape == (len(sample_graphs), 10)

    assert np.isclose(features.mean(), 0, atol=1e-10)
    assert np.isclose(features.std(), 1, atol=1)


@pytest.mark.parametrize("iterations", [1, 2, 3])
@pytest.mark.parametrize("sparse", [True, False])
def test_weisfeiler_lehman_descriptor(sample_graphs, iterations, sparse):
    descriptor = WeisfeilerLehmanDescriptor(iterations=iterations, sparse=sparse)

    features = descriptor(sample_graphs)

    if sparse:
        assert isinstance(features, csr_array)
    else:
        assert isinstance(features, np.ndarray)

    assert features.shape[0] == len(sample_graphs)

    if sparse:
        for i in range(len(sample_graphs)):
            row_slice = features[i : i + 1]
            assert row_slice.nnz > 0
    else:
        for i in range(len(sample_graphs)):
            assert len([v for v in features[i] if v > 0]) > 0


@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("use_node_labels", [True, False])
def test_weisfeiler_lehman_descriptor_qm9(
    sample_molecules, iterations, sparse, use_node_labels
):
    descriptor = WeisfeilerLehmanDescriptor(
        iterations=iterations,
        sparse=sparse,
        use_node_labels=use_node_labels,
        node_label_key="element" if use_node_labels else None,
    )

    features = descriptor(sample_molecules)

    if sparse:
        assert isinstance(features, csr_array)
    else:
        assert isinstance(features, np.ndarray)

    assert features.shape[0] == len(sample_molecules)

    if sparse:
        feature_sets = []
        for i in range(len(sample_molecules)):
            row = features[i : i + 1]
            indices = row.indices
            counts = row.data
            feature_dict = {idx: count for idx, count in zip(indices, counts)}
            feature_sets.append(feature_dict)

        assert feature_sets[0] != feature_sets[1]
        assert feature_sets[0] != feature_sets[2]
        assert feature_sets[1] != feature_sets[2]
    else:
        assert not np.array_equal(features[0], features[1])
        assert not np.array_equal(features[0], features[2])
        assert not np.array_equal(features[1], features[2])
