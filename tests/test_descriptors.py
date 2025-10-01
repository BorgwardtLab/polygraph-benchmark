import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_array

from polygraph.utils.descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    EigenvalueHistogram,
    NormalizedDescriptor,
    OrbitCounts,
    SparseDegreeHistogram,
    WeisfeilerLehmanDescriptor,
)
from polygraph.datasets import PlanarGraphDataset


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


@pytest.mark.parametrize("num_bins", [10, 100, 200])
def test_sparse_equivalence(num_bins):
    ds = list(PlanarGraphDataset("train").to_nx())
    clustering_sparse = ClusteringHistogram(num_bins, sparse=True)
    clustering_dense = ClusteringHistogram(num_bins, sparse=False)
    eigenvalue_sparse = EigenvalueHistogram(num_bins, sparse=True)
    eigenvalue_dense = EigenvalueHistogram(num_bins, sparse=False)

    sparse_clustering_features = clustering_sparse(ds)
    dense_clustering_features = clustering_dense(ds)

    assert np.allclose(
        sparse_clustering_features.toarray(), dense_clustering_features
    )

    sparse_eigenvalue_features = eigenvalue_sparse(ds)
    dense_eigenvalue_features = eigenvalue_dense(ds)

    assert np.allclose(
        sparse_eigenvalue_features.toarray(), dense_eigenvalue_features
    )


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
def test_weisfeiler_lehman_descriptor(sample_graphs, iterations):
    descriptor = WeisfeilerLehmanDescriptor(iterations=iterations)

    features = descriptor(sample_graphs)

    assert isinstance(features, csr_array)

    assert features.shape[0] == len(sample_graphs)

    for i in range(len(sample_graphs)):
        row_slice = features[i : i + 1]
        assert row_slice.nnz > 0
        assert len([v for v in row_slice.data if v > 0]) > 0


@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("use_node_labels", [True, False])
def test_weisfeiler_lehman_descriptor_molecules(
    sample_molecules, iterations, use_node_labels
):
    descriptor = WeisfeilerLehmanDescriptor(
        iterations=iterations,
        use_node_labels=use_node_labels,
        node_label_key="atom_labels" if use_node_labels else None,
    )

    features = descriptor(sample_molecules)

    assert isinstance(features, csr_array)

    assert features.shape[0] == len(sample_molecules)
    for i in range(len(sample_molecules)):
        for j in range(i + 1, len(sample_molecules)):
            difference = features[i : i + 1] - features[j : j + 1]
            assert difference.max() > 0 or difference.min() < 0, (
                f"Features for molecules {i} and {j} are identical"
            )
