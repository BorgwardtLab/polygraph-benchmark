from polygrapher.metrics.base.frechet_distance import FrechetDistance
from polygrapher.utils.graph_descriptors import (
    NormalizedDescriptor,
    RandomGIN,
)

from typing import Optional, List, Union

__all__ = ["GraphNeuralNetworkFrechetDistance"]


class GraphNeuralNetworkFrechetDistance(FrechetDistance):
    def __init__(self, reference_graphs, node_feat_loc: Optional[List[str]] = None, node_feat_dim: int = 1, edge_feat_loc: Optional[List[str]] = None, edge_feat_dim: int = 0, seed: Union[int, None] = 42):
        super().__init__(
            reference_graphs, NormalizedDescriptor(RandomGIN(node_feat_loc=node_feat_loc, input_dim=node_feat_dim, edge_feat_loc=edge_feat_loc, edge_feat_dim=edge_feat_dim, seed=seed), reference_graphs)
        )
