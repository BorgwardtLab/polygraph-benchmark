from typing import Collection

import networkx as nx

from polygraph.metrics.base.mmd import (
    DescriptorMMD2,
    DescriptorMMD2Interval,
)
from polygraph.utils.graph_descriptors import (
    ClusteringHistogram,
    EigenvalueHistogram,
    OrbitCounts,
    SparseDegreeHistogram,
)
from polygraph.utils.kernels import LinearKernel


__all__ = [
    "LinearOrbitMMD2",
    "LinearOrbitMMD2Interval",
    "LinearClusteringMMD2",
    "LinearClusteringMMD2Interval",
    "LinearDegreeMMD2",
    "LinearDegreeMMD2Interval",
    "LinearSpectralMMD2",
    "LinearSpectralMMD2Interval",
]

# Below follow the metrics introduced in "EFFICIENT GRAPH GENERATION WITH GRAPH RECURRENT ATTENTION NETWORKS", Liao et al.


class LinearOrbitMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(descriptor_fn=OrbitCounts()),
            variant="biased",
        )


class LinearOrbitMMD2Interval(DescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(descriptor_fn=OrbitCounts()),
            variant="biased",
        )


class LinearClusteringMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(descriptor_fn=ClusteringHistogram(bins=100)),
            variant="biased",
        )


class LinearClusteringMMD2Interval(DescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(descriptor_fn=ClusteringHistogram(bins=100)),
            variant="biased",
        )


class LinearDegreeMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(descriptor_fn=SparseDegreeHistogram()),
            variant="biased",
        )


class LinearDegreeMMD2Interval(DescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(descriptor_fn=SparseDegreeHistogram()),
            variant="biased",
        )


class LinearSpectralMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(descriptor_fn=EigenvalueHistogram()),
            variant="biased",
        )


class LinearSpectralMMD2Interval(DescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(descriptor_fn=EigenvalueHistogram()),
            variant="biased",
        )
