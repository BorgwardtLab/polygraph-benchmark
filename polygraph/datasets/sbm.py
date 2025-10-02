from typing import Literal, Optional, Tuple

import joblib
import networkx as nx
import numpy as np
from loguru import logger
from scipy import stats
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
from tqdm.rich import tqdm

from polygraph.datasets.base import ProceduralGraphDataset, SplitGraphDataset
from polygraph.datasets.base.graph_storage import GraphStorage


def is_sbm_graph(
    graph: nx.Graph,
    intra_p: float = 0.3,
    inter_p: float = 0.005,
    min_n_communities: int = 2,
    max_n_communities: int = 5,
    min_n_nodes_per_community: int = 20,
    max_n_nodes_per_community: int = 40,
) -> bool:
    import graph_tool.all as gt  # pyright: ignore
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
    edge_counts = e.todense()[:n_blocks, :n_blocks]  # pyright: ignore
    if (
        (node_counts > max_n_nodes_per_community).sum() > 0
        or (node_counts < min_n_nodes_per_community).sum() > 0
        or n_blocks > max_n_communities
        or n_blocks < min_n_communities
    ):
        return False

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape(
        (1, -1)
    )
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - intra_p) ** 2 / (
        est_p_intra * (1 - est_p_intra) + 1e-6
    )
    W_p_inter = (est_p_inter - inter_p) ** 2 / (
        est_p_inter * (1 - est_p_inter) + 1e-6
    )

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    return p > 0.9


def is_sbm_graph_alt(
    graph: nx.Graph,
    intra_p: float = 0.3,
    inter_p: float = 0.005,
    min_n_communities: int = 2,
    max_n_communities: int = 5,
    min_n_nodes_per_community: int = 20,
    max_n_nodes_per_community: int = 40,
) -> bool:
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
            for node, cluster in enumerate(
                nx.community.kernighan_lin_bisection(g)
            )
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
    if n_blocks < min_n_communities or n_blocks > max_n_communities:
        return False

    # Count nodes per community
    node_counts = {}
    for i, comm in enumerate(communities):
        node_counts[i] = len(comm)

    # Check community sizes
    if any(
        count < min_n_nodes_per_community or count > max_n_nodes_per_community
        for count in node_counts.values()
    ):
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
    max_inter_edges = node_counts_arr.reshape(
        (-1, 1)
    ) @ node_counts_arr.reshape((1, -1))

    # Calculate probabilities
    p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    edge_counts_copy = edge_counts.copy()
    np.fill_diagonal(edge_counts_copy, 0)
    p_inter = edge_counts_copy / (max_inter_edges + 1e-6)

    # Calculate test statistics using the same approach as graph-tool
    W_p_intra = (p_intra - intra_p) ** 2 / (p_intra * (1 - p_intra) + 1e-6)
    W_p_inter = (p_inter - inter_p) ** 2 / (p_inter * (1 - p_inter) + 1e-6)

    # Combine test statistics
    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)

    # Calculate p-value using chi-square CDF
    p = 1 - stats.chi2.cdf(abs(W), df=1)
    p_value = p.mean()
    return p_value > 0.9


class ProceduralSBMGraphDataset(ProceduralGraphDataset):
    """Procedural version of [`SBMGraphDataset`][polygraph.datasets.SBMGraphDataset].

    Graphs are generated by first sampling the number of communties and then the number of nodes per community.
    Finally, edges are sampled according to the intra- and inter-community edge probabilities.

    Args:
        split: Split to load.
        num_graphs: Number of graphs to generate for this split.
        seed: Seed for the random number generator.
        intra_p: Intra-community edge probability.
        inter_p: Inter-community edge probability.
        n_communities: Range of number of communities in the format (min, max).
        n_nodes_per_community: Range of number of nodes per community in the format (min, max).
        memmap: Whether to use memory mapping for the dataset.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        num_graphs: int,
        seed: int = 42,
        intra_p: float = 0.3,
        inter_p: float = 0.005,
        n_communities: Tuple[int, int] = (2, 5),
        n_nodes_per_community: Tuple[int, int] = (20, 40),
        memmap: bool = False,
        show_generation_progress: bool = False,
    ):
        config_hash: str = joblib.hash(  # pyright: ignore
            (
                num_graphs,
                intra_p,
                inter_p,
                n_communities,
                n_nodes_per_community,
                seed,
                split,
            ),
            hash_name="md5",
        )
        self._rng = np.random.default_rng(
            int.from_bytes(config_hash.encode(), "big")
        )
        self._num_graphs = num_graphs
        self._intra_p = intra_p
        self._inter_p = inter_p
        self._n_communities = n_communities
        self._n_nodes_per_community = n_nodes_per_community
        super().__init__(
            split,
            config_hash,
            memmap,
            show_generation_progress,
        )

    def generate_data(self) -> GraphStorage:
        graphs = [
            from_networkx(self._random_sbm())
            for _ in tqdm(
                range(self._num_graphs),
                desc="Generating SBM graphs",
                disable=not self.show_generation_progress,
            )
        ]
        return GraphStorage.from_pyg_batch(Batch.from_data_list(graphs))

    def is_valid(self, graph: nx.Graph) -> bool:
        """Check if a graph is a valid SBM graph."""
        return is_sbm_graph(
            graph,
            intra_p=self._intra_p,
            inter_p=self._inter_p,
            min_n_communities=self._n_communities[0],
            max_n_communities=self._n_communities[1],
            min_n_nodes_per_community=self._n_nodes_per_community[0],
            max_n_nodes_per_community=self._n_nodes_per_community[1],
        )

    def is_valid_alt(self, graph: nx.Graph) -> bool:
        return is_sbm_graph_alt(
            graph,
            intra_p=self._intra_p,
            inter_p=self._inter_p,
            min_n_communities=self._n_communities[0],
            max_n_communities=self._n_communities[1],
            min_n_nodes_per_community=self._n_nodes_per_community[0],
            max_n_nodes_per_community=self._n_nodes_per_community[1],
        )

    def _random_sbm(self):
        num_communities = self._rng.integers(
            self._n_communities[0], self._n_communities[1] + 1
        )
        num_nodes_per_community = self._rng.integers(
            self._n_nodes_per_community[0],
            self._n_nodes_per_community[1] + 1,
            size=num_communities,
        )
        community_labels = np.repeat(
            np.arange(num_communities), num_nodes_per_community
        )
        edge_probs = np.where(
            np.expand_dims(community_labels, 0)
            == np.expand_dims(community_labels, 1),
            self._intra_p,
            self._inter_p,
        )
        adj = (self._rng.random(edge_probs.shape) < edge_probs).astype(int)
        adj = np.triu(adj, 1)
        adj = adj + adj.transpose()
        g = nx.from_numpy_array(adj)

        for u, v, d in g.edges(data=True):
            if "weight" in d:
                del d["weight"]
        return g


class SBMGraphDataset(SplitGraphDataset):
    """SBM graph dataset proposed by Martinkus et al. [1].

    The graphs are sampled from stochastic block models with random parameters.

    - The number of communities is sampled uniformly at random from 2-5 (inclusive).
    - The number of nodes per community is sampled uniformly at random from 20-40 (inclusive).
    - The intra-community edge probability is set at 0.3.
    - The inter-community edge probability is set at 0.005.

    {{ plot_first_k_graphs("SBMGraphDataset", "train", 3, node_size=50) }}

    Dataset statistics:

    {{ summary_md_table("SBMGraphDataset", ["train", "val", "test"]) }}


    References:
        [1] Martinkus, K., Loukas, A., Perraudin, N., & Wattenhofer, R. (2022).
            [SPECTRE: Spectral Conditioning Helps to Overcome the Expressivity Limits
            of One-shot Graph Generators](https://arxiv.org/abs/2204.01613). In Proceedings of the 39th International
            Conference on Machine Learning (ICML).
    """

    _URL_FOR_SPLIT = {
        "train": "https://sandbox.zenodo.org/records/332447/files/sbm_train.pt?download=1",
        "val": "https://sandbox.zenodo.org/records/332447/files/sbm_val.pt?download=1",
        "test": "https://sandbox.zenodo.org/records/332447/files/sbm_test.pt?download=1",
    }

    _HASH_FOR_SPLIT = {
        "train": "48c9461fda3bde4a960e96b5ea21a6b4",
        "val": "24f1687b11bcbe456e96173672704636",
        "test": "00318495cebb55d96a84f59428c2df9e",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    def is_valid(self, graph: nx.Graph) -> bool:
        """Check if a graph is a valid SBM graph."""
        return is_sbm_graph(graph)

    def is_valid_alt(self, graph: nx.Graph) -> bool:
        return is_sbm_graph_alt(graph)

    def hash_for_split(self, split: str) -> Optional[str]:
        return self._HASH_FOR_SPLIT[split]


class SBMLGraphDataset(ProceduralSBMGraphDataset):
    def __init__(
        self,
        split: Literal["train", "val", "test"],
        num_graphs: int,
        seed: int = 42,
        memmap: bool = False,
        show_generation_progress: bool = False,
    ):
        if split == "train":
            num_graphs = 8192
        elif split == "val":
            num_graphs = 4096
        elif split == "test":
            num_graphs = 4096

        super().__init__(
            split,
            num_graphs,
            seed,
            memmap,
            show_generation_progress,
        )
