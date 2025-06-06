"""Concrete definitions of kernel AUROCs using GIN descriptor."""

from typing import Collection

import networkx as nx
import numpy as np

from polygraph.metrics.base.classifier_metric import (
    LogisticRegressionClassifierMetric,
    MultiKernelClassifierMetric,
)
from polygraph.utils.graph_descriptors import (
    NormalizedDescriptor,
    RandomGIN,
)
from polygraph.utils.kernels import (
    AdaptiveRBFKernel,
)
from typing import Optional, List, Union


__all__ = [
    "RBFGraphNeuralNetworkInformedness",
    "LRGraphNeuralNetworkInformedness",
]


class RBFGraphNeuralNetworkInformedness(MultiKernelClassifierMetric):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        node_feat_loc: Optional[List[str]] = None,
        node_feat_dim: int = 1,
        edge_feat_loc: Optional[List[str]] = None,
        edge_feat_dim: int = 0,
        seed: Union[int, None] = 42,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=NormalizedDescriptor(
                    RandomGIN(
                        node_feat_loc=node_feat_loc,
                        input_dim=node_feat_dim,
                        edge_feat_loc=edge_feat_loc,
                        edge_feat_dim=edge_feat_dim,
                        seed=seed,
                    ),
                    reference_graphs,
                ),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="informedness",
        )


class LRGraphNeuralNetworkInformedness(LogisticRegressionClassifierMetric):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
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
            variant="informedness",
        )
