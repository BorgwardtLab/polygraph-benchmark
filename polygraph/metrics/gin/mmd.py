"""Concrete definitions of graph MMDs used in the literature."""

from typing import Collection

import networkx as nx
import numpy as np

from polygraph.metrics.base.mmd import (
    DescriptorMMD2,
    DescriptorMMD2Interval,
    MaxDescriptorMMD2,
    MaxDescriptorMMD2Interval,
)
from polygraph.utils.graph_descriptors import (
    NormalizedDescriptor,
    RandomGIN,
)
from polygraph.utils.kernels import (
    AdaptiveRBFKernel,
    LinearKernel,
)
from typing import Optional, List, Union


__all__ = [
    "RBFGraphNeuralNetworkMMD2",
    "RBFGraphNeuralNetworkMMD2Interval",
    "LinearGraphNeuralNetworkMMD2",
    "LinearGraphNeuralNetworkMMD2Interval",
]


class RBFGraphNeuralNetworkMMD2(MaxDescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph], node_feat_loc: Optional[List[str]] = None, node_feat_dim: int = 1, edge_feat_loc: Optional[List[str]] = None, edge_feat_dim: int = 0, seed: Union[int, None] = 42):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=NormalizedDescriptor(RandomGIN(node_feat_loc=node_feat_loc, input_dim=node_feat_dim, edge_feat_loc=edge_feat_loc, edge_feat_dim=edge_feat_dim, seed=seed), reference_graphs),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )


class RBFGraphNeuralNetworkMMD2Interval(MaxDescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph], node_feat_loc: Optional[List[str]] = None, node_feat_dim: int = 1, edge_feat_loc: Optional[List[str]] = None, edge_feat_dim: int = 0, seed: Union[int, None] = 42):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=NormalizedDescriptor(RandomGIN(node_feat_loc=node_feat_loc, input_dim=node_feat_dim, edge_feat_loc=edge_feat_loc, edge_feat_dim=edge_feat_dim, seed=seed), reference_graphs),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )


class LinearGraphNeuralNetworkMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph], node_feat_loc: Optional[List[str]] = None, node_feat_dim: int = 1, edge_feat_loc: Optional[List[str]] = None, edge_feat_dim: int = 0, seed: Union[int, None] = 42):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(
                descriptor_fn=NormalizedDescriptor(RandomGIN(node_feat_loc=node_feat_loc, input_dim=node_feat_dim, edge_feat_loc=edge_feat_loc, edge_feat_dim=edge_feat_dim, seed=seed), reference_graphs),
            ),
            variant="biased",
        )


class LinearGraphNeuralNetworkMMD2Interval(DescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph], node_feat_loc: Optional[List[str]] = None, node_feat_dim: int = 1, edge_feat_loc: Optional[List[str]] = None, edge_feat_dim: int = 0, seed: Union[int, None] = 42):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(
                descriptor_fn=NormalizedDescriptor(RandomGIN(node_feat_loc=node_feat_loc, input_dim=node_feat_dim, edge_feat_loc=edge_feat_loc, edge_feat_dim=edge_feat_dim, seed=seed), reference_graphs),
            ),
            variant="biased",
        )
