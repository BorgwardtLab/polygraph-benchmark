from graph_gen_gym.metrics.frechet_distance import FrechetDistance
from graph_gen_gym.metrics.utils.graph_descriptors import (
    NormalizedDescriptor,
    RandomGIN,
)

__all__ = ["GraphNeuralNetworkFrechetDistance"]


class GraphNeuralNetworkFrechetDistance(FrechetDistance):
    def __init__(self, reference_graphs):
        super().__init__(
            reference_graphs, NormalizedDescriptor(RandomGIN(), reference_graphs)
        )
