"""Graph descriptor functions for converting graphs into feature vectors.

This module provides various functions that convert networkx graphs into numerical
representations suitable for kernel methods. Each descriptor is callable with an iterable of graphs and returns either a dense
`numpy.ndarray` or sparse `scipy.sparse.csr_array` of shape `(n_graphs, n_features)`.
They implement the [`GraphDescriptor`][polygraph.descriptors.GraphDescriptor] interface.


Descriptors may, for example, be implemented as follows:

```python
from typing import Iterable
import networkx as nx
import numpy as np

def my_descriptor(graphs: Iterable[nx.Graph]) -> np.ndarray:
    hists = [nx.degree_histogram(graph) for graph in graphs]
    hists = [
        np.concatenate([hist, np.zeros(128 - len(hist))], axis=0)
        for hist in hists
    ]
    hists = np.stack(hists, axis=0)
    return hists / hists.sum(axis=1, keepdims=True) # shape: (n_graphs, n_features)
```

Available descriptors:
    - [`SparseDegreeHistogram`][polygraph.descriptors.SparseDegreeHistogram]: Sparse degree distribution
    - [`DegreeHistogram`][polygraph.descriptors.DegreeHistogram]: Dense degree distribution
    - [`ClusteringHistogram`][polygraph.descriptors.ClusteringHistogram]: Distribution of clustering coefficients
    - [`OrbitCounts`][polygraph.descriptors.OrbitCounts]: Graph orbit statistics
    - [`EigenvalueHistogram`][polygraph.descriptors.EigenvalueHistogram]: Eigenvalue histogram of normalized Laplacian
    - [`RandomGIN`][polygraph.descriptors.RandomGIN]: Embeddings of random Graph Isomorphism Network
    - [`WeisfeilerLehmanDescriptor`][polygraph.descriptors.WeisfeilerLehmanDescriptor]: Weisfeiler-Lehman subtree features
    - [`NormalizedDescriptor`][polygraph.descriptors.NormalizedDescriptor]: Standardized descriptor wrapper
"""

from polygraph.descriptors.interface import GraphDescriptor
from polygraph.descriptors.generic_descriptors import (
    SparseDegreeHistogram,
    DegreeHistogram,
    ClusteringHistogram,
    OrbitCounts,
    EigenvalueHistogram,
    RandomGIN,
    WeisfeilerLehmanDescriptor,
    NormalizedDescriptor,
)

__all__ = [
    "GraphDescriptor",
    "SparseDegreeHistogram",
    "DegreeHistogram",
    "ClusteringHistogram",
    "OrbitCounts",
    "EigenvalueHistogram",
    "RandomGIN",
    "WeisfeilerLehmanDescriptor",
    "NormalizedDescriptor",
]
