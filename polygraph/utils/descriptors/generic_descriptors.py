import copy
import warnings
from collections import Counter
from hashlib import blake2b
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    Literal,
    Generic,
)

import networkx as nx
import numpy as np
import orbit_count
import torch
from scipy.sparse import csr_array
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch
from torch_geometric.utils import degree, from_networkx

from polygraph import GraphType
from polygraph.utils.descriptors.interface import GraphDescriptor
from polygraph.utils.descriptors.gin import GIN
from polygraph.utils.parallel import batched_distribute_function, flatten_lists


def sparse_histogram(
    values: np.ndarray, bins: np.ndarray, density: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse version of np.histogram.

    Same as np.histogram but returns a tuple (index, counts) where index is a numpy
    containing the non-empty bins and counts is a numpy array counting the number of
    values in each non-emptybin. Uses right-open bins [a, b).
    """
    indices = np.minimum(
        np.digitize(values, bins, right=False) - 1, len(bins) - 2
    )
    unique_indices, counts = np.unique(indices, return_counts=True)
    sorting_perm = np.argsort(unique_indices)
    unique_indices = unique_indices[sorting_perm]
    counts = counts[sorting_perm]
    if density:
        counts = counts / np.sum(counts)
    return unique_indices, counts


def sparse_histograms_to_array(
    sparse_histograms: List[Tuple[np.ndarray, np.ndarray]],
    num_bins: int,
) -> csr_array:
    index = np.concatenate(
        [sparse_histogram[0] for sparse_histogram in sparse_histograms]
    )
    data = np.concatenate(
        [sparse_histogram[1] for sparse_histogram in sparse_histograms]
    )
    ptr = np.zeros(len(sparse_histograms) + 1, dtype=np.int32)
    ptr[1:] = np.cumsum(
        [len(sparse_histogram[0]) for sparse_histogram in sparse_histograms]
    ).astype(np.int32)
    return csr_array((data, index, ptr), (len(sparse_histograms), num_bins))


class DegreeHistogram(GraphDescriptor[nx.Graph]):
    """Computes normalized degree distributions of graphs.

    For each graph, computes a histogram of node degrees and normalizes it to sum to 1.
    Pads all histograms to a fixed maximum degree.

    Args:
        max_degree: Maximum degree to consider. Larger degrees are ignored
    """

    def __init__(self, max_degree: int):
        self._max_degree = max_degree

    def __call__(self, graphs: Iterable[nx.Graph]) -> np.ndarray:
        hists = [nx.degree_histogram(graph) for graph in graphs]
        hists = [
            np.concatenate(
                [hist, np.zeros(self._max_degree - len(hist))], axis=0
            )
            for hist in hists
        ]
        hists = np.stack(hists, axis=0)
        return hists / hists.sum(axis=1, keepdims=True)


class SparseDegreeHistogram(GraphDescriptor[nx.Graph]):
    """Memory-efficient version of degree distribution computation.

    Similar to DegreeHistogram but returns a sparse matrix, making it suitable for
    graphs with high maximum degree where most degree bins are empty.
    """

    def __call__(self, graphs: Iterable[nx.Graph]) -> csr_array:
        hists = [
            np.array(nx.degree_histogram(graph)) / graph.number_of_nodes()
            for graph in graphs
        ]
        index = [np.nonzero(hist)[0].astype(np.int32) for hist in hists]
        data = [hist[idx] for hist, idx in zip(hists, index)]
        ptr = np.zeros(len(index) + 1, dtype=np.int32)
        ptr[1:] = np.cumsum([len(idx) for idx in index]).astype(np.int32)
        result = csr_array(
            (np.concatenate(data), np.concatenate(index), ptr),
            (len(hists), 100_000),
        )
        return result


class ClusteringHistogram(GraphDescriptor[nx.Graph]):
    """Computes histograms of local clustering coefficients.

    For each graph, computes the distribution of local clustering coefficients
    across nodes. The clustering coefficient measures the fraction of possible
    triangles through each node that exist.

    Args:
        bins: Number of histogram bins covering [0,1]
        sparse: Whether to return a dense np.ndarray or a sparse csr_array. Sparse version may be faster when comparing many graphs.
    """

    def __init__(self, bins: int, sparse: bool = False):
        self._num_bins = bins
        self._sparse = sparse
        if sparse:
            self._bins = np.linspace(0.0, 1.0, bins + 1)
        else:
            self._bins = None

    def __call__(
        self, graphs: Iterable[nx.Graph]
    ) -> Union[np.ndarray, csr_array]:
        all_clustering_coeffs = [
            list(nx.clustering(graph).values())  # pyright: ignore
            for graph in graphs
        ]
        if self._sparse:
            assert self._bins is not None
            sparse_histograms = [
                sparse_histogram(
                    np.array(clustering_coeffs), self._bins, density=True
                )
                for clustering_coeffs in all_clustering_coeffs
            ]
            return sparse_histograms_to_array(sparse_histograms, self._num_bins)
        else:
            hists = [
                np.histogram(
                    clustering_coeffs,
                    bins=self._num_bins,
                    range=(0.0, 1.0),
                    density=False,
                )[0]
                for clustering_coeffs in all_clustering_coeffs
            ]
            hists = np.stack(hists, axis=0)
            return hists / hists.sum(axis=1, keepdims=True)


class OrbitCounts(GraphDescriptor[nx.Graph]):
    """Computes graph orbit statistics .

    Warning:
        Self-loops are automatically removed from input graphs.
    """

    _mode: Literal["node", "edge"]

    def __init__(
        self, graphlet_size: int = 4, mode: Literal["node", "edge"] = "node"
    ):
        self._graphlet_size = graphlet_size
        self._mode = mode

    def __call__(self, graphs: Iterable[nx.Graph]):
        # Check if any graph has a self-loop
        graphs = list(graphs)
        self_loops = [list(nx.selfloop_edges(g)) for g in graphs]
        if any(len(loops) > 0 for loops in self_loops):
            warnings.warn(
                "Graph with self-loop passed to orbit descriptor, deleting self-loops"
            )
            graphs = [copy.deepcopy(g) for g in graphs]
            for g, loops in zip(graphs, self_loops):
                g.remove_edges_from(loops)

        if self._mode == "node":
            counts = orbit_count.batched_node_orbit_counts(
                graphs, graphlet_size=self._graphlet_size
            )
        elif self._mode == "edge":
            counts = orbit_count.batched_edge_orbit_counts(
                graphs, graphlet_size=self._graphlet_size
            )
        else:
            raise ValueError(f"Invalid mode: {self._mode}")
        counts = [count.mean(axis=0) for count in counts]
        return np.stack(counts, axis=0)


class EigenvalueHistogram(GraphDescriptor[nx.Graph]):
    """Computes eigenvalue histogram of normalized Laplacian.

    For each graph, computes the eigenvalue spectrum of its normalized Laplacian
    matrix and returns a histogram of the eigenvalues.

    Args:
        n_bins: Number of histogram bins
        sparse: Whether to return a dense np.ndarray or a sparse csr_array. Sparse version may be faster when comparing many graphs.
    """

    def __init__(self, n_bins: int = 200, sparse: bool = False):
        self._sparse = sparse
        self._n_bins = n_bins
        if sparse:
            self._bins = np.linspace(-1e-5, 2, n_bins + 1)
        else:
            self._bins = None

    def __call__(
        self, graphs: Iterable[nx.Graph]
    ) -> Union[np.ndarray, csr_array]:
        all_eigs = []
        for g in graphs:
            eigs = np.linalg.eigvalsh(
                nx.normalized_laplacian_matrix(g).todense()
            )
            all_eigs.append(eigs)

        if self._sparse:
            assert self._bins is not None
            sparse_histograms = [
                sparse_histogram(np.array(eigs), self._bins, density=True)
                for eigs in all_eigs
            ]
            return sparse_histograms_to_array(sparse_histograms, self._n_bins)
        else:
            histograms = []
            for eigs in all_eigs:
                spectral_pmf, _ = np.histogram(
                    eigs, bins=self._n_bins, range=(-1e-5, 2), density=False
                )
                spectral_pmf = spectral_pmf / spectral_pmf.sum()
                histograms.append(spectral_pmf)
            return np.stack(histograms, axis=0)


class RandomGIN(GraphDescriptor[nx.Graph]):
    """Random Graph Isomorphism Network for graph embeddings.

    Initializes a randomly weighted Graph Isomorphism Network (GIN) and uses it
    to compute graph embeddings. The network parameters are fixed after random
    initialization. Node features default to node degrees if not specified.

    Args:
        num_layers: Number of GIN layers
        hidden_dim: Hidden dimension in each layer
        neighbor_pooling_type: How to aggregate neighbor features ('sum', 'mean', or 'max')
        graph_pooling_type: How to aggregate node features into graph features ('sum', 'mean', or 'max')
        input_dim: Dimension of input node features
        edge_feat_dim: Dimension of edge features (0 for no edge features)
        dont_concat: If True, only use final layer features instead of concatenating all layers
        num_mlp_layers: Number of MLP layers in each GIN layer
        output_dim: Dimension of final graph embedding
        device: Device to run the model on (e.g., 'cpu' or 'cuda')
        node_feat_loc: List of node attributes to use as features. If None, use degree as features.
        edge_feat_loc: List of edge attributes to use as features. If None, no edge features are used.
        seed: Random seed for weight initialization
    """

    def __init__(
        self,
        num_layers: int = 3,
        hidden_dim: int = 35,
        neighbor_pooling_type: str = "sum",
        graph_pooling_type: str = "sum",
        input_dim: int = 1,
        edge_feat_dim: int = 0,
        dont_concat: bool = False,
        num_mlp_layers: int = 2,
        output_dim: int = 1,
        device: str = "cpu",
        node_feat_loc: Optional[List[str]] = None,
        edge_feat_loc: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.model = GIN(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            neighbor_pooling_type=neighbor_pooling_type,
            graph_pooling_type=graph_pooling_type,
            input_dim=input_dim,
            edge_feat_dim=edge_feat_dim,
            num_mlp_layers=num_mlp_layers,
            output_dim=output_dim,
            init="orthogonal",
            seed=seed,
        )
        self._device = device
        self.model = self.model.to(device)

        self.model.eval()

        if dont_concat:
            self._feat_fn = self.model.get_graph_embed_no_cat
        else:
            self._feat_fn = self.model.get_graph_embed

        self.node_feat_loc = node_feat_loc
        self.edge_feat_loc = edge_feat_loc

    @torch.inference_mode()
    def __call__(self, graphs: Iterable[nx.Graph]) -> np.ndarray:
        pyg_graphs = [
            from_networkx(
                g,
                group_node_attrs=self.node_feat_loc,
                group_edge_attrs=self.edge_feat_loc,
            )
            for g in graphs
        ]

        if self.node_feat_loc is None:  # Use degree as features
            feats = (
                torch.cat(
                    [
                        degree(index=g.edge_index[0], num_nodes=g.num_nodes)
                        + degree(index=g.edge_index[1], num_nodes=g.num_nodes)
                        for g in pyg_graphs
                    ]
                )
                .unsqueeze(-1)
                .to(self._device)
            )
        else:
            feats = torch.cat([g.x for g in pyg_graphs]).to(self._device)

        if self.edge_feat_loc is None:
            edge_attr = None
        else:
            edge_attr = torch.cat([g.edge_attr for g in pyg_graphs]).to(
                self._device
            )

        batch = Batch.from_data_list(pyg_graphs).to(self._device)  # pyright: ignore

        graph_embeds = self._feat_fn(
            feats, batch.edge_index, batch.batch, edge_attr=edge_attr
        )
        return graph_embeds.cpu().detach().numpy()


class NormalizedDescriptor(GraphDescriptor[GraphType], Generic[GraphType]):
    """Standardizes graph descriptors using reference graph statistics.

    Wraps a graph descriptor to standardize its output features (zero mean, unit variance)
    based on statistics computed from a set of reference graphs. This is useful when
    different features have very different scales.

    The wrapped graph descriptor must return a dense numpy array.

    Args:
        descriptor_fn: Base descriptor function to normalize
        ref_graphs: Reference graphs used to compute normalization statistics
    """

    def __init__(
        self,
        descriptor_fn: Callable[[Iterable[GraphType]], np.ndarray],
        ref_graphs: Iterable[GraphType],
    ):
        self._descriptor_fn = descriptor_fn
        self._scaler = StandardScaler()
        self._scaler.fit(self._descriptor_fn(ref_graphs))

    def __call__(self, graphs: Iterable[GraphType]) -> np.ndarray:
        result = self._descriptor_fn(graphs)
        result = self._scaler.transform(result)
        assert isinstance(result, np.ndarray)
        return result


class WeisfeilerLehmanDescriptor(GraphDescriptor[nx.Graph]):
    """Weisfeiler-Lehman subtree features for graphs.

    Computes graph features by iteratively hashing node neighborhoods using the
    WL algorithm. Returns sparse feature vectors where each dimension corresponds
    to a subtree pattern.

    Warning:
        Hash collisions may occur, as at most $2^{31}$ unique hashes are used.

    Args:
        iterations: Number of WL iterations
        use_node_labels: Whether to use existing node labels instead of degrees
        node_label_key: Node attribute key for labels if use_node_labels is True
        digest_size: Number of bytes for hashing in intermediate WL iterations (1-4)
        n_jobs: Number of workers for parallel computation
        n_graphs_per_job: Number of graphs per worker
        show_progress: Whether to show a progress bar
    """

    def __init__(
        self,
        iterations: int = 3,
        use_node_labels: bool = False,
        node_label_key: Optional[str] = None,
        digest_size: int = 4,
        n_jobs: int = 1,
        n_graphs_per_job: int = 100,
        show_progress: bool = False,
    ):
        if use_node_labels and node_label_key is None:
            raise ValueError(
                "node_label_key must be provided if use_node_labels is True"
            )

        if digest_size > 4:
            raise ValueError("Digest size must be at most 4 bytes")

        if use_node_labels:
            assert node_label_key is not None, (
                "node_label_key must be provided if use_node_labels is True"
            )
            self._node_label_key: str = node_label_key
        else:
            self._node_label_key: str = "degree"

        self._iterations = iterations
        self._use_node_labels = use_node_labels
        self._digest_size = digest_size  # Number of bytes in the hash
        self._n_jobs = n_jobs
        self._n_graphs_per_job = n_graphs_per_job
        self._show_progress = show_progress

    def __call__(self, graphs: Iterable[nx.Graph]) -> csr_array:
        graph_list = list(graphs)

        if not self._use_node_labels:
            self._assign_node_degree_labels(graph_list)

        features = []
        if self._n_jobs == 1:
            for graph in graph_list:
                features.append(self._compute_wl_features(graph))
        else:
            features = batched_distribute_function(
                self._compute_wl_features_worker,
                graph_list,
                n_jobs=self._n_jobs,
                show_progress=self._show_progress,
                batch_size=self._n_graphs_per_job,
            )

        sparse_array = self._create_sparse_matrix(features)
        return sparse_array

    def _assign_node_degree_labels(self, graphs: List[nx.Graph]) -> None:
        for graph in graphs:
            for node in graph.nodes():
                graph.nodes[node][self._node_label_key] = graph.degree(node)  # pyright: ignore

    def _compute_wl_features_worker(self, graphs: List[nx.Graph]) -> List[dict]:
        return [self._compute_wl_features(graph) for graph in graphs]

    def _compute_wl_features(self, graph: nx.Graph) -> dict:
        hash_iter_0 = dict(
            Counter(list(dict(graph.nodes(self._node_label_key)).values()))
        )
        hashes = dict(
            Counter(
                flatten_lists(
                    list(
                        nx.weisfeiler_lehman_subgraph_hashes(
                            graph,
                            node_attr=self._node_label_key,
                            iterations=self._iterations,
                            digest_size=self._digest_size,
                        ).values()
                    )
                )
            )
        )
        all_hashes = hashes | hash_iter_0

        int_hashes = {}
        for hash_key, count in all_hashes.items():
            if not isinstance(hash_key, str):
                # This case catches hash_iter_0
                hash_key = blake2b(
                    str(hash_key).encode(), digest_size=self._digest_size
                ).hexdigest()

            assert (
                isinstance(hash_key, str)
                and len(hash_key) == 2 * self._digest_size
            ), "Hash key is not a hex string or has incorrect length"
            int_key = int(hash_key, 16)
            int_key = int_key & 0x7FFFFFFF
            int_hashes[int_key] = count
            assert 0 <= int_key <= (2**31 - 1), (
                f"Unexpected hash key {int_key} out of bounds"
            )

        if len(int_hashes) != len(all_hashes):
            # This might artificially inflate the resulting kernel value but not
            # by much in our experiments.
            warnings.warn(
                "Hash collision detected in Weisfeiler-Lehman descriptor"
            )
        return int_hashes

    def _create_sparse_matrix(self, all_features: list) -> csr_array:
        n_graphs = len(all_features)
        data = []
        indices = []
        indptr = [0]

        for features in all_features:
            sorted_features = sorted(features.items(), key=lambda x: x[0])
            for feature_idx, count in sorted_features:
                indices.append(feature_idx)
                data.append(count)
            indptr.append(len(indices))

        return csr_array(
            (
                np.array(data, dtype=np.int32),
                np.array(indices, dtype=np.int32),
                np.array(indptr, dtype=np.int32),
            ),
            shape=(n_graphs, 2**31),
        )
