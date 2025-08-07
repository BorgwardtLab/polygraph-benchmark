from typing import Collection, Literal, Optional, List, Union

import networkx as nx

from polygraph.metrics.base.polygraphscore import (
    ClassifierMetric,
    PolyGraphScore,
)
from polygraph.utils.graph_descriptors import (
    OrbitCounts,
    ClusteringHistogram,
    SparseDegreeHistogram,
    EigenvalueHistogram,
    NormalizedDescriptor,
    RandomGIN,
)

__all__ = [
    "PGS5",
    "ClassifierOrbitMetric",
    "ClassifierClusteringMetric",
    "ClassifierDegreeeMetric",
    "ClassifierSpectralMetric",
    "GraphNeuralNetworkClassifierMetric",
]


class PGS5(PolyGraphScore):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptors={
                "orbit": OrbitCounts(),
                "clustering": ClusteringHistogram(bins=100),
                "degree": SparseDegreeHistogram(),
                "spectral": EigenvalueHistogram(),
                "gin": NormalizedDescriptor(
                    RandomGIN(
                        node_feat_loc=None,
                        input_dim=1,
                        edge_feat_loc=None,
                        edge_feat_dim=0,
                        seed=42,
                    ),
                    reference_graphs,
                ),
            },
            variant="jsd",
            classifier="tabpfn",
        )


# Below are the definitions of individual classifier metrics


class ClassifierOrbitMetric(ClassifierMetric):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Literal["logistic", "tabpfn"] = "tabpfn",
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
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Literal["logistic", "tabpfn"] = "tabpfn",
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
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Literal["logistic", "tabpfn"] = "tabpfn",
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
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Literal["logistic", "tabpfn"] = "tabpfn",
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=EigenvalueHistogram(),
            variant=variant,
            classifier=classifier,
        )


class GraphNeuralNetworkClassifierMetric(ClassifierMetric):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Literal["logistic", "tabpfn"] = "tabpfn",
        node_feat_loc: Optional[List[str]] = None,
        node_feat_dim: int = 1,
        edge_feat_loc: Optional[List[str]] = None,
        edge_feat_dim: int = 0,
        seed: Union[int, None] = 42,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=NormalizedDescriptor(
                RandomGIN(
                    node_feat_loc=node_feat_loc,
                    input_dim=node_feat_dim,
                    edge_feat_loc=edge_feat_loc,
                    edge_feat_dim=edge_feat_dim,
                    seed=seed,
                ),
                reference_graphs,
            ),
            variant=variant,
            classifier=classifier,
        )
