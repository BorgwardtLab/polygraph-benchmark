"""Concrete definitions of graph MMDs used in the literature."""

from typing import Collection

import networkx as nx
import numpy as np

from graph_gen_gym.metrics.mmd import (
    DescriptorMMD2,
    DescriptorMMD2Interval,
    MaxDescriptorMMD2,
    MaxDescriptorMMD2Interval,
)
from graph_gen_gym.metrics.utils.graph_descriptors import (
    ClusteringHistogram,
    EigenvalueHistogram,
    OrbitCounts,
    SparseDegreeHistogram,
)
from graph_gen_gym.metrics.utils.kernels import AdaptiveRBFKernel, GaussianTV

__all__ = [
    "GRANOrbitMMD2",
    "GRANOrbitMMD2Interval",
    "GRANClusteringMMD2",
    "GRANClusteringMMD2Interval",
    "GRANDegreeMMD2",
    "GRANDegreeMMD2Interval",
    "GRANSpectralMMD2",
    "GRANSpectralMMD2Interval",
    "RBFOrbitMMD2",
    "RBFOrbitMMD2Interval",
    "RBFClusteringMMD2",
    "RBFClusteringMMD2Interval",
    "RBFDegreeMMD2",
    "RBFDegreeMMD2Interval",
    "RBFSpectralMMD2",
    "RBFSpectralMMD2Interval",
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


# Below follow the metrics introduced in "ON EVALUATION METRICS  FOR GRAPH GENERATIVE MODELS", Thompson et al.


class RBFOrbitMMD2(MaxDescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=OrbitCounts(),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )


class RBFOrbitMMD2Interval(MaxDescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=OrbitCounts(),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )


class RBFClusteringMMD2(MaxDescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=ClusteringHistogram(bins=100),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )


class RBFClusteringMMD2Interval(MaxDescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=ClusteringHistogram(bins=100),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )


class RBFDegreeMMD2(MaxDescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=SparseDegreeHistogram(),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )


class RBFDegreeMMD2Interval(MaxDescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=SparseDegreeHistogram(),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )


class RBFSpectralMMD2(MaxDescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=EigenvalueHistogram(),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )


class RBFSpectralMMD2Interval(MaxDescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=EigenvalueHistogram(),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )
