from typing import Collection, Literal

import networkx as nx

from polygraph.metrics.base.polygraphscore import (
    ClassifierMetric,
)
from polygraph.utils.graph_descriptors import (
    OrbitCounts,
    ClusteringHistogram,
    SparseDegreeHistogram,
    EigenvalueHistogram,
)

__all__ = [
    "ClassifierOrbitMetric",
    "ClassifierClusteringMetric",
    "ClassifierDegreeeMetric",
    "ClassifierSpectralMetric",
]



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
