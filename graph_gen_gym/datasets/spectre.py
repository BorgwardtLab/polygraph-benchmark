from typing import List

import networkx as nx
import numpy as np
import torch
import torch_geometric
from scipy.stats import chi2

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

    @property
    def identifier(self):
        return "spectre_planar"

    def is_valid(self, graph: nx.Graph) -> bool:
        if isinstance(graph, nx.Graph):
            return nx.is_connected(graph) and nx.is_planar(graph)
        raise TypeError

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

    @property
    def identifier(self):
        return "spectre_sbm"

    def is_valid(self, graph: nx.Graph) -> bool:
        import graph_tool.all as gt

        adj = nx.adjacency_matrix(graph).toarray()
        idx = adj.nonzero()
        g = gt.Graph()
        g.add_edge_list(np.transpose(idx))
        try:
            state = gt.minimize_blockmodel_dl(g)
        except ValueError:
            return False

        # Refine using merge-split MCMC
        for i in range(100):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

        b = state.get_blocks()
        b = gt.contiguous_map(state.get_blocks())
        state = state.copy(b=b)
        e = state.get_matrix()
        n_blocks = state.get_nonempty_B()
        node_counts = state.get_nr().get_array()[:n_blocks]
        edge_counts = e.todense()[:n_blocks, :n_blocks]
        if (
            (node_counts > 40).sum() > 0
            or (node_counts < 20).sum() > 0
            or n_blocks > 5
            or n_blocks < 2
        ):
            return False

        max_intra_edges = node_counts * (node_counts - 1)
        est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

        max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
        np.fill_diagonal(edge_counts, 0)
        est_p_inter = edge_counts / (max_inter_edges + 1e-6)

        W_p_intra = (est_p_intra - 0.3) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
        W_p_inter = (est_p_inter - 0.005) ** 2 / (
            est_p_inter * (1 - est_p_inter) + 1e-6
        )

        W = W_p_inter.copy()
        np.fill_diagonal(W, W_p_intra)
        p = 1 - chi2.cdf(abs(W), 1)
        p = p.mean()
        return p > 0.9  # p value < 10 %
