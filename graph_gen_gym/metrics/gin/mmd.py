"""Concrete definitions of graph MMDs used in the literature."""

from typing import Collection

import networkx as nx
import numpy as np

from graph_gen_gym.metrics.base.mmd import (
    DescriptorMMD2,
    DescriptorMMD2Interval,
    MaxDescriptorMMD2,
    MaxDescriptorMMD2Interval,
)
from graph_gen_gym.utils.graph_descriptors import (
    NormalizedDescriptor,
    RandomGIN,
)
from graph_gen_gym.utils.kernels import (
    AdaptiveRBFKernel,
    LinearKernel,
)

__all__ = [
    "RBFGraphNeuralNetworkMMD2",
    "RBFGraphNeuralNetworkMMD2Interval",
    "LinearGraphNeuralNetworkMMD2",
    "LinearGraphNeuralNetworkMMD2Interval",
]


class RBFGraphNeuralNetworkMMD2(MaxDescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=NormalizedDescriptor(RandomGIN(), reference_graphs),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )


class RBFGraphNeuralNetworkMMD2Interval(MaxDescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=NormalizedDescriptor(RandomGIN(), reference_graphs),
                bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
            ),
            variant="biased",
        )


class LinearGraphNeuralNetworkMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(
                descriptor_fn=NormalizedDescriptor(RandomGIN(), reference_graphs),
            ),
            variant="biased",
        )


class LinearGraphNeuralNetworkMMD2Interval(DescriptorMMD2Interval):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=LinearKernel(
                descriptor_fn=NormalizedDescriptor(RandomGIN(), reference_graphs),
            ),
            variant="biased",
        )
