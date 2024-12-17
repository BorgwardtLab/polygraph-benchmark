from typing import Collection

import networkx as nx

from graph_gen_gym.metrics.base.mmd import (
    DescriptorMMD2,
    DescriptorMMD2Interval,
)
from graph_gen_gym.utils.graph_descriptors import (
    ClusteringHistogram,
    EigenvalueHistogram,
    OrbitCounts,
    SparseDegreeHistogram,
)
from graph_gen_gym.utils.kernels import GaussianTV


__all__ = [
    "GRANOrbitMMD2",
    "GRANOrbitMMD2Interval",
    "GRANClusteringMMD2",
    "GRANClusteringMMD2Interval",
    "GRANDegreeMMD2",
    "GRANDegreeMMD2Interval",
    "GRANSpectralMMD2",
    "GRANSpectralMMD2Interval",
]

# Below follow the metrics introduced in "EFFICIENT GRAPH GENERATION WITH GRAPH RECURRENT ATTENTION NETWORKS", Liao et al.


class GRANOrbitMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=OrbitCounts(), bw=30),
            variant="biased",
        )


class GRANOrbitMMD2Interval(DescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=OrbitCounts(), bw=30),
            variant="biased",
        )


class GRANClusteringMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=ClusteringHistogram(bins=100), bw=1.0 / 10),
            variant="biased",
        )


class GRANClusteringMMD2Interval(DescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=ClusteringHistogram(bins=100), bw=1.0 / 10),
            variant="biased",
        )


class GRANDegreeMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=SparseDegreeHistogram(), bw=1.0),
            variant="biased",
        )


class GRANDegreeMMD2Interval(DescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=SparseDegreeHistogram(), bw=1.0),
            variant="biased",
        )


class GRANSpectralMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=EigenvalueHistogram(), bw=1.0),
            variant="biased",
        )


class GRANSpectralMMD2Interval(DescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=EigenvalueHistogram(), bw=1.0),
            variant="biased",
        )
