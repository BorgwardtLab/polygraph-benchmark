from typing import Collection

import networkx as nx
import numpy as np

from polygraph.metrics.base.classifier_metric import (
    LogisticRegressionClassifierMetric,
    MultiKernelClassifierMetric,
)
from polygraph.utils.kernels import AdaptiveRBFKernel
from polygraph.utils.graph_descriptors import (
    OrbitCounts,
    ClusteringHistogram,
    SparseDegreeHistogram,
    EigenvalueHistogram,
)

__all__ = [
    "RBFOrbitInformedness",
    "LROrbitInformedness",
    "RBFClusteringInformedness",
    "LRClusteringInformedness",
    "RBFDegreeInformedness",
    "LRDegreeInformedness",
    "RBFSpectralInformedness",
    "LRSpectralInformedness",
]


class RBFOrbitInformedness(MultiKernelClassifierMetric):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=OrbitCounts(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="informedness",
        )


class LROrbitInformedness(LogisticRegressionClassifierMetric):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=OrbitCounts(),
            variant="informedness",
        )


class RBFClusteringInformedness(MultiKernelClassifierMetric):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=ClusteringHistogram(bins=100),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="informedness",
        )


class LRClusteringInformedness(LogisticRegressionClassifierMetric):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=ClusteringHistogram(bins=100),
            variant="informedness",
        )


class RBFDegreeInformedness(MultiKernelClassifierMetric):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=SparseDegreeHistogram(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="informedness",
        )


class LRDegreeInformedness(LogisticRegressionClassifierMetric):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=SparseDegreeHistogram(),
            variant="informedness",
        )


class RBFSpectralInformedness(MultiKernelClassifierMetric):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=EigenvalueHistogram(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="informedness",
        )


class LRSpectralInformedness(LogisticRegressionClassifierMetric):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=EigenvalueHistogram(),
            variant="informedness",
        )
