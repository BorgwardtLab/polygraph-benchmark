from typing import List

import community
import networkx as nx
import numpy as np
from scipy import stats

from graph_gen_gym.datasets.dataset import OnlineGraphDataset
from graph_gen_gym.datasets.graph import Graph


class PlanarGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/f3kXPP4LICWKbBx/download",
        "val": "https://datashare.biochem.mpg.de/s/CN2zeY8EvIlUxN6/download",
        "test": "https://datashare.biochem.mpg.de/s/DNwtgo3mlOErxHX/download",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    def is_valid(self, graph: nx.Graph) -> bool:
        return nx.is_connected(graph) and nx.is_planar(graph)

    def sample(self, n_samples: int, replace: bool = False) -> List[Graph]:
        idx_to_sample = np.random.choice(len(self), n_samples, replace=replace)
        return self[idx_to_sample]


class SBMGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/INpX2k7JHjdfWaa/download",
        "val": "https://datashare.biochem.mpg.de/s/SwuNNa1RvCIJAg8/download",
        "test": "https://datashare.biochem.mpg.de/s/DwEdatPuPZ60Bpd/download",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    def is_valid(self, graph: nx.Graph) -> bool:
        # Community detection using Louvain method
        communities = community.best_partition(graph)
        unique_communities = set(communities.values())
        n_blocks = len(unique_communities)

        # Check number of communities
        if n_blocks < 2 or n_blocks > 5:
            return False

        # Count nodes per community
        node_counts = {}
        for comm in communities.values():
            node_counts[comm] = node_counts.get(comm, 0) + 1

        # Check community sizes
        if any(count < 20 or count > 40 for count in node_counts.values()):
            return False

        # Calculate edge densities
        edge_counts = np.zeros((n_blocks, n_blocks))
        for edge in graph.edges():
            c1, c2 = communities[edge[0]], communities[edge[1]]
            edge_counts[c1][c2] += 1
            if c1 != c2:
                edge_counts[c2][c1] += 1

        # Calculate probabilities
        node_counts_arr = np.array([node_counts[i] for i in range(n_blocks)])
        max_intra_edges = node_counts_arr * (node_counts_arr - 1)
        max_inter_edges = node_counts_arr.reshape((-1, 1)) @ node_counts_arr.reshape(
            (1, -1)
        )

        # TODO: if we decide to generate graphs with different intra-community densities, we need to change this line
        # Intra-community density (should be around 0.3)
        p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

        # TODO: if we decide to generate graphs with different inter-community densities, we need to change this line
        # Inter-community density (should be around 0.005)
        np.fill_diagonal(edge_counts, 0)
        p_inter = edge_counts / (max_inter_edges + 1e-6)

        # Chi-square test for goodness of fit
        W_intra = stats.chi2_contingency([p_intra, [0.3] * len(p_intra)])[1]
        W_inter = stats.chi2_contingency(
            [p_inter.flatten(), [0.005] * len(p_inter.flatten())]
        )[1]

        # Average p-value should be > 0.9
        return (W_intra + W_inter) / 2 > 0.9
