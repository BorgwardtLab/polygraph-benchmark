"""Graph descriptor functions for converting graphs into feature vectors.

This module provides various functions that convert graphs into numerical
representations suitable for kernel methods. Each descriptor is callable with an iterable of graphs and returns either a dense
`numpy.ndarray` or sparse `scipy.sparse.csr_array` of shape `(n_graphs, n_features)`.
They implement the [`GraphDescriptor`][polygraph.utils.descriptors.GraphDescriptor] interface.

Graphs may either be passed as `nx.Graph` or `rdkit.Chem.Mol` objects.

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

Generic graph descriptors:
    - [`SparseDegreeHistogram`][polygraph.utils.descriptors.SparseDegreeHistogram]: Sparse degree distribution
    - [`DegreeHistogram`][polygraph.utils.descriptors.DegreeHistogram]: Dense degree distribution
    - [`ClusteringHistogram`][polygraph.utils.descriptors.ClusteringHistogram]: Distribution of clustering coefficients
    - [`OrbitCounts`][polygraph.utils.descriptors.OrbitCounts]: Graph orbit statistics
    - [`EigenvalueHistogram`][polygraph.utils.descriptors.EigenvalueHistogram]: Eigenvalue histogram of normalized Laplacian
    - [`RandomGIN`][polygraph.utils.descriptors.RandomGIN]: Embeddings of random Graph Isomorphism Network
    - [`WeisfeilerLehmanDescriptor`][polygraph.utils.descriptors.WeisfeilerLehmanDescriptor]: Weisfeiler-Lehman subtree features
    - [`NormalizedDescriptor`][polygraph.utils.descriptors.NormalizedDescriptor]: Standardized descriptor wrapper


Molecule descriptors:
    - [`TopoChemicalDescriptor`][polygraph.utils.descriptors.molecule_descriptors.TopoChemicalDescriptor]: Topological features based on bond structure
    - [`FingerprintDescriptor`][polygraph.utils.descriptors.molecule_descriptors.FingerprintDescriptor]: Molecular fingerprints
    - [`LipinskiDescriptor`][polygraph.utils.descriptors.molecule_descriptors.LipinskiDescriptor]: Physico-chemical properties
    - [`ChemNetDescriptor`][polygraph.utils.descriptors.molecule_descriptors.ChemNetDescriptor]: Random projection of ChemNet embeddings, based on SMILES strings
    - [`MolCLRDescriptor`][polygraph.utils.descriptors.molecule_descriptors.MolCLRDescriptor]: Random projection of MolCLR embeddings from a GNN
"""

from polygraph.utils.descriptors.interface import GraphDescriptor
from polygraph.utils.descriptors.generic_descriptors import (
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
