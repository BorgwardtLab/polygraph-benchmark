from typing import List

import networkx as nx
import numpy as np
from loguru import logger
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
        import graph_tool.all as gt
        from scipy.stats import chi2

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
        return p > 0.9

    def is_valid_alt(self, graph: nx.Graph) -> bool:
        partition_methods = [
            lambda g: {
                node: cluster
                for node, cluster in enumerate(nx.community.louvain_communities(g))
            },  # NetworkX's Louvain
            lambda g: {
                node: cluster
                for node, cluster in enumerate(
                    nx.community.greedy_modularity_communities(g)
                )
            },  # Greedy modularity
            lambda g: {
                node: cluster
                for node, cluster in enumerate(
                    nx.community.label_propagation_communities(g)
                )
            },  # Label propagation
            lambda g: {
                node: cluster
                for node, cluster in enumerate(nx.community.kernighan_lin_bisection(g))
            },  # Kernighan-Lin
        ]

        best_partition = None
        best_modularity = -1

        for method in partition_methods:
            try:
                partition = method(graph)
                partition = [set(partition[i]) for i in range(len(partition))]
                mod = nx.community.modularity(graph, partition)
                if mod > best_modularity:
                    best_modularity = mod
                    best_partition = partition
            except Exception as e:
                logger.error(f"Method {method.__name__} failed")
                logger.error(f"Error: {e}")
                continue
        if best_partition is None:
            return False

        # Convert best_partition from list of sets to list of lists for consistency
        communities = [list(community) for community in best_partition]
        n_blocks = len(communities)

        # Check number of communities
        if n_blocks < 2 or n_blocks > 5:
            return False

        # Count nodes per community
        node_counts = {}
        for i, comm in enumerate(communities):
            node_counts[i] = len(comm)

        # Check community sizes
        if any(count < 20 or count > 40 for count in node_counts.values()):
            return False

        # Calculate edge densities with more precise counting
        edge_counts = np.zeros((n_blocks, n_blocks))
        node_mapping = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_mapping[node] = i

        for edge in graph.edges():
            c1, c2 = node_mapping[edge[0]], node_mapping[edge[1]]
            edge_counts[c1][c2] += 1
            if c1 != c2:
                edge_counts[c2][c1] += 1

        # Convert node counts to array
        node_counts_arr = np.array([node_counts[i] for i in range(n_blocks)])

        # Calculate max possible edges (similar to graph-tool)
        max_intra_edges = node_counts_arr * (node_counts_arr - 1) / 2
        max_inter_edges = node_counts_arr.reshape((-1, 1)) @ node_counts_arr.reshape(
            (1, -1)
        )

        # Calculate probabilities
        p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

        edge_counts_copy = edge_counts.copy()
        np.fill_diagonal(edge_counts_copy, 0)
        p_inter = edge_counts_copy / (max_inter_edges + 1e-6)

        # Calculate test statistics using the same approach as graph-tool
        W_p_intra = (p_intra - 0.3) ** 2 / (p_intra * (1 - p_intra) + 1e-6)
        W_p_inter = (p_inter - 0.005) ** 2 / (p_inter * (1 - p_inter) + 1e-6)

        # Combine test statistics
        W = W_p_inter.copy()
        np.fill_diagonal(W, W_p_intra)

        # Calculate p-value using chi-square CDF
        p = 1 - stats.chi2.cdf(abs(W), df=1)
        p_value = p.mean()
        return p_value > 0.9
