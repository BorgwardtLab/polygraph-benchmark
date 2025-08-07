"""Concrete definitions of graph MMDs used in the literature."""

from typing import Collection, Optional

import networkx as nx
import numpy as np

from polygraph.metrics.base.mmd import (
    MaxDescriptorMMD2,
    MaxDescriptorMMD2Interval,
)
from polygraph.utils.graph_descriptors import (
    ClusteringHistogram,
    EigenvalueHistogram,
    OrbitCounts,
    SparseDegreeHistogram,
)
from polygraph.utils.kernels import AdaptiveRBFKernel

__all__ = [
    "RBFOrbitMMD2",
    "RBFOrbitMMD2Interval",
    "RBFClusteringMMD2",
    "RBFClusteringMMD2Interval",
    "RBFDegreeMMD2",
    "RBFDegreeMMD2Interval",
    "RBFSpectralMMD2",
    "RBFSpectralMMD2Interval",
]


# Below follow the metrics introduced in "ON EVALUATION METRICS  FOR GRAPH GENERATIVE MODELS", Thompson et al.


class RBFOrbitMMD2(MaxDescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=OrbitCounts(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="biased",
        )


class RBFOrbitMMD2Interval(MaxDescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph], subsample_size: int, num_samples: int = 500, coverage: Optional[float] = 0.95):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=OrbitCounts(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )


class RBFClusteringMMD2(MaxDescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=ClusteringHistogram(bins=100),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="biased",
        )


class RBFClusteringMMD2Interval(MaxDescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph], subsample_size: int, num_samples: int = 500, coverage: Optional[float] = 0.95):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=ClusteringHistogram(bins=100),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )


class RBFDegreeMMD2(MaxDescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=SparseDegreeHistogram(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="biased",
        )


class RBFDegreeMMD2Interval(MaxDescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph], subsample_size: int, num_samples: int = 500, coverage: Optional[float] = 0.95):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=SparseDegreeHistogram(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )


class RBFSpectralMMD2(MaxDescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=EigenvalueHistogram(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="biased",
        )


class RBFSpectralMMD2Interval(MaxDescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph], subsample_size: int, num_samples: int = 500, coverage: Optional[float] = 0.95):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=EigenvalueHistogram(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )
