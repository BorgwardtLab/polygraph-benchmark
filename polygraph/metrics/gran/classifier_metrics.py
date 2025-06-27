from typing import Collection, Literal

import networkx as nx
import numpy as np

from polygraph.metrics.base.classifier_metric import (
    ClassifierMetric,
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
    "ClassifierOrbitMetric",
    "RBFClusteringInformedness",
    "ClassifierClusteringMetric",
    "RBFDegreeInformedness",
    "ClassifierDegreeeMetric",
    "RBFSpectralInformedness",
    "ClassifierSpectralMetric",
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
            variant="informedness-adaptive",
        )


class ClassifierOrbitMetric(ClassifierMetric):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal[
            "informedness", "informedness-adaptive", "jsd"
        ] = "informedness-adaptive",
        classifier: Literal["logistic", "tabpfn"] = "logistic",
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=OrbitCounts(),
            variant=variant,
            classifier=classifier,
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
            variant="informedness-adaptive",
        )


class ClassifierClusteringMetric(ClassifierMetric):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal[
            "informedness", "informedness-adaptive", "jsd"
        ] = "informedness-adaptive",
        classifier: Literal["logistic", "tabpfn"] = "logistic",
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=ClusteringHistogram(bins=100),
            variant=variant,
            classifier=classifier,
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
            variant="informedness-adaptive",
        )


class ClassifierDegreeeMetric(ClassifierMetric):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal[
            "informedness", "informedness-adaptive", "jsd"
        ] = "informedness-adaptive",
        classifier: Literal["logistic", "tabpfn"] = "logistic",
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=SparseDegreeHistogram(),
            variant=variant,
            classifier=classifier,
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
            variant="informedness-adaptive",
        )


class ClassifierSpectralMetric(ClassifierMetric):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal[
            "informedness", "informedness-adaptive", "jsd"
        ] = "informedness-adaptive",
        classifier: Literal["logistic", "tabpfn"] = "logistic",
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=EigenvalueHistogram(),
            variant=variant,
            classifier=classifier,
        )
