"""Concrete definitions of kernel AUROCs using GIN descriptor."""

from typing import Collection, Literal

import networkx as nx

from polygraph.metrics.base.polygraphscore import (
    ClassifierMetric,
)
from polygraph.utils.graph_descriptors import (
    NormalizedDescriptor,
    RandomGIN,
)
from typing import Optional, List, Union


__all__ = [
    "GraphNeuralNetworkClassifierMetric",
]


class GraphNeuralNetworkClassifierMetric(ClassifierMetric):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal[
            "informedness", "jsd"
        ] = "jsd",
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
