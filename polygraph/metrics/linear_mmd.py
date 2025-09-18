"""Concrete definitions of graph MMDs used in the literature."""

from typing import Collection

import networkx as nx

from polygraph.metrics.base.mmd import (
    DescriptorMMD2,
    DescriptorMMD2Interval,
)
from polygraph.utils.descriptors import (
    NormalizedDescriptor,
    RandomGIN,
)
from polygraph.utils.kernels import (
    LinearKernel,
)
from typing import Optional, List, Union


__all__ = [
    "LinearGraphNeuralNetworkMMD2",
    "LinearGraphNeuralNetworkMMD2Interval",
]


class LinearGraphNeuralNetworkMMD2(DescriptorMMD2[nx.Graph]):
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
            kernel=LinearKernel(
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
            ),
            variant="biased",
        )


class LinearGraphNeuralNetworkMMD2Interval(DescriptorMMD2Interval[nx.Graph]):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: Optional[float] = 0.95,
        node_feat_loc: Optional[List[str]] = None,
        node_feat_dim: int = 1,
        edge_feat_loc: Optional[List[str]] = None,
        edge_feat_dim: int = 0,
        seed: Union[int, None] = 42,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(
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
            ),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )
