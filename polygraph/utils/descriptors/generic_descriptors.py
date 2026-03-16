import copy
import warnings
from collections import Counter
from hashlib import blake2b
from typing import (
    Callable,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import networkx as nx
import numpy as np
import orbit_count
import torch
from scipy.sparse import csgraph, csr_array
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch
from torch_geometric.utils import degree, from_networkx

from polygraph import GraphType
from polygraph.utils.descriptors.gin import GIN
from polygraph.utils.descriptors.interface import GraphDescriptor
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


class ShortestPathHistogramDescriptor(GraphDescriptor[nx.Graph]):
    """Shortest-path kernel feature map.

    For each graph, counts unordered node pairs by the triple
    (label(u), label(v), quantized shortest-path length(u,v)) and returns
    a sparse count vector in a large, fixed hashed feature space,
    suitable for use with LinearKernel (dot product).

    Args:
        node_label_key: Node attribute to use as discrete label; if None, ignores labels.
        weight_key: Edge attribute for path lengths; if None, unweighted distances.
        bin_width: Optional distance bin width for weighted graphs; uses floor(d/bin_width).
        distance_decimals: Number of decimals to round distances if bin_width is None (weighted).
        digest_size: Bytes for hashing keys (1-4). Controls collision rate.
        n_jobs: Number of workers for parallel computation.
        n_graphs_per_job: Number of graphs per worker.
        show_progress: Whether to show a progress bar.
        seed: Random seed for TruncatedSVD projection.
        project_dim: Number of dimensions to project the descriptor to. If None, no projection is performed.
    """

    def __init__(
        self,
        node_label_key: Optional[str] = None,
        weight_key: Optional[str] = None,
        bin_width: Optional[float] = None,
        distance_decimals: int = 3,
        digest_size: int = 4,
        n_jobs: int = 1,
        n_graphs_per_job: int = 100,
        show_progress: bool = False,
        seed: int = 42,
        project_dim: Optional[int] = None,
    ):
        if digest_size > 4:
            raise ValueError("digest_size must be <= 4")

        self.node_label_key = node_label_key
        self.weight_key = weight_key
        self.bin_width = bin_width
        self.distance_decimals = distance_decimals
        self._digest_size = digest_size
        self._n_jobs = n_jobs
        self._n_graphs_per_job = n_graphs_per_job
        self._show_progress = show_progress
        self.project_dim = project_dim
        self._seed = seed

        if self.project_dim is not None:
            self._validate_project_dim()
            self._projector = TruncatedSVD(
                n_components=self.project_dim, random_state=self._seed
            )
            self._fitted = False

    def __call__(
        self, graphs: Iterable[nx.Graph]
    ) -> Union[csr_array, np.ndarray]:
        graph_list = list(graphs)

        if self._n_jobs == 1:
            features = [
                self._compute_graph_features(graph) for graph in graph_list
            ]
        else:
            features = batched_distribute_function(
                self._compute_graph_features_worker,
                graph_list,
                n_jobs=self._n_jobs,
                show_progress=self._show_progress,
                batch_size=self._n_graphs_per_job,
            )

        sparse_array = self._create_sparse_matrix(features)

        if self.project_dim is None:
            return sparse_array
        else:
            if not self._fitted:
                sparse_array = self._projector.fit_transform(sparse_array)
                self._fitted = True
            else:
                sparse_array = self._projector.transform(sparse_array)
            return sparse_array

    def _compute_graph_features(self, graph: nx.Graph) -> dict:
        """Compute shortest-path features for a single graph."""
        n = graph.number_of_nodes()
        if n <= 1:
            return {}

        nodes = list(graph.nodes())

        dists, rows, cols = self._compute_shortest_paths_vectorized(
            graph, nodes
        )
        if dists is None or rows is None or cols is None:
            return {}

        q_dists = self._quantize_distances_vectorized(dists)

        feats: dict[int, int] = {}
        labels = self._get_node_labels(graph)

        if labels is None:
            self._process_unlabeled_features(q_dists, feats)
        else:
            self._process_labeled_features(
                q_dists, rows, cols, labels, nodes, feats
            )

        return feats

    def _validate_project_dim(self) -> None:
        if self.project_dim is None or self.project_dim <= 0:
            raise ValueError("project_dim must be a positive integer")
        if self.project_dim > 2**31:
            raise ValueError("project_dim must be less than 2**31")

    def _compute_graph_features_worker(
        self, graphs: List[nx.Graph]
    ) -> List[dict]:
        return [self._compute_graph_features(graph) for graph in graphs]

    def _get_node_labels(self, graph: nx.Graph) -> Optional[dict]:
        if self.node_label_key is None:
            return None
        return {
            u: str(graph.nodes[u].get(self.node_label_key, "__UNK__"))
            for u in graph.nodes()
        }

    def _hash_feature_key(
        self, label_u: str, label_v: str, quantized_dist: Union[int, float]
    ) -> int:
        lu, lv = (
            (label_u, label_v) if label_u <= label_v else (label_v, label_u)
        )
        key = f"{lu}|{lv}|{quantized_dist}".encode()
        h = (
            int(blake2b(key, digest_size=self._digest_size).hexdigest(), 16)
            & 0x7FFFFFFF
        )
        return h

    def _compute_shortest_paths_vectorized(
        self, graph: nx.Graph, nodes: list
    ) -> tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        n = len(nodes)
        weight_kwargs = (
            {"weight": self.weight_key} if self.weight_key is not None else {}
        )
        adj = nx.to_scipy_sparse_array(
            graph, nodelist=nodes, format="csr", **weight_kwargs
        )
        dist_matrix = csgraph.shortest_path(
            adj, directed=False, unweighted=(self.weight_key is None)
        )
        rows, cols = np.triu_indices(n, k=1)
        dists = dist_matrix[rows, cols]
        finite_mask = np.isfinite(dists)
        if not np.any(finite_mask):
            return None, None, None
        return dists[finite_mask], rows[finite_mask], cols[finite_mask]

    def _quantize_distances_vectorized(self, dists: np.ndarray) -> np.ndarray:
        if self.bin_width is not None:
            return np.floor(dists / self.bin_width).astype(int)
        elif self.weight_key is None:
            return np.round(dists).astype(int)
        else:
            return np.round(dists, self.distance_decimals)

    def _process_unlabeled_features(
        self, q_dists: np.ndarray, feats: dict
    ) -> None:
        unique_dists, counts = np.unique(q_dists, return_counts=True)
        for d, count in zip(unique_dists, counts):
            k = self._hash_feature_key("*", "*", d)
            feats[k] = feats.get(k, 0) + int(count)

    def _process_labeled_features(
        self,
        q_dists: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        labels: dict,
        nodes: list,
        feats: dict,
    ) -> None:
        node_labels = np.array([labels[u] for u in nodes])
        unique_lbls, labels_inv = np.unique(node_labels, return_inverse=True)
        l_u_idx = labels_inv[rows]
        l_v_idx = labels_inv[cols]
        mask_swap = l_u_idx > l_v_idx
        idx1 = np.where(mask_swap, l_v_idx, l_u_idx)
        idx2 = np.where(mask_swap, l_u_idx, l_v_idx)
        triplets = np.column_stack((idx1, idx2, q_dists))
        unique_rows, counts = np.unique(triplets, axis=0, return_counts=True)
        for row, count in zip(unique_rows, counts):
            i1, i2, d_val = row
            lab1 = unique_lbls[int(i1)]
            lab2 = unique_lbls[int(i2)]
            if self.bin_width is not None or self.weight_key is None:
                d_val = int(d_val)
            k = self._hash_feature_key(lab1, lab2, d_val)
            feats[k] = feats.get(k, 0) + int(count)

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
                np.array(data, dtype=np.float32),
                np.array(indices, dtype=np.int32),
                np.array(indptr, dtype=np.int32),
            ),
            shape=(n_graphs, 2**31),
        )


class PyramidMatchDescriptor(GraphDescriptor[nx.Graph]):
    """Pyramid Match Kernel descriptor.

    Computes a histogram of node features at multiple resolutions (pyramid).
    The features are quantized into bins of increasing size $2^l \\delta$.
    The resulting vector is a concatenation of weighted histograms at each level.

    When used with a LinearKernel, this approximates the Pyramid Match Kernel.

    Args:
        num_levels: Number of levels in the pyramid (L).
        bin_width: Base bin width (delta) for level 0.
        node_label_key: Node attribute to use as feature. If None, uses node degree.
        digest_size: Number of bytes for hashing.
        n_jobs: Number of parallel jobs.
        n_graphs_per_job: Batch size for parallel jobs.
        show_progress: Show progress bar.
        project_dim: Number of dimensions to project the descriptor to. If None, no projection is performed.
        seed: Random seed for projection.
    """

    def __init__(
        self,
        num_levels: int = 4,
        bin_width: float = 1.0,
        node_label_key: Optional[str] = None,
        digest_size: int = 4,
        n_jobs: int = 1,
        n_graphs_per_job: int = 100,
        show_progress: bool = False,
        seed: int = 42,
        project_dim: Optional[int] = None,
    ):
        self.num_levels = num_levels
        self.bin_width = bin_width
        self.node_label_key = node_label_key
        self._digest_size = digest_size
        self._n_jobs = n_jobs
        self._n_graphs_per_job = n_graphs_per_job
        self._show_progress = show_progress
        self.project_dim = project_dim
        self._seed = seed

        if self._digest_size >= 4:
            self._feature_dim = 2**31
        else:
            self._feature_dim = 2 ** (8 * self._digest_size)

        if self.project_dim is not None:
            self._validate_project_dim()
            self._projector = TruncatedSVD(
                n_components=self.project_dim, random_state=self._seed
            )
            self._fitted = False

    def __call__(
        self, graphs: Iterable[nx.Graph]
    ) -> Union[csr_array, np.ndarray]:
        graph_list = list(graphs)
        if self._n_jobs == 1:
            features = [self._compute_features(g) for g in graph_list]
        else:
            features = batched_distribute_function(
                self._compute_features_worker,
                graph_list,
                n_jobs=self._n_jobs,
                show_progress=self._show_progress,
                batch_size=self._n_graphs_per_job,
            )

        sparse_array = self._create_sparse_matrix(features)

        if self.project_dim is None:
            return sparse_array
        else:
            if not self._fitted:
                sparse_array = self._projector.fit_transform(sparse_array)
                self._fitted = True
            else:
                sparse_array = self._projector.transform(sparse_array)
            return sparse_array

    def _validate_project_dim(self) -> None:
        if self.project_dim is None or self.project_dim <= 0:
            raise ValueError("project_dim must be a positive integer")
        if self.project_dim > 2**31:
            raise ValueError("project_dim must be less than 2**31")

    def _compute_features_worker(self, graphs: List[nx.Graph]) -> List[dict]:
        return [self._compute_features(g) for g in graphs]

    def _compute_features(self, graph: nx.Graph) -> dict:
        if self.node_label_key is None:
            feats = [d for _, d in graph.degree()]
            feats = np.array(feats).reshape(-1, 1)
        else:
            try:
                feats_list = []
                for _, data in graph.nodes(data=True):
                    val = data.get(self.node_label_key, 0)
                    if isinstance(val, (list, np.ndarray)):
                        feats_list.append(val)
                    else:
                        feats_list.append([val])
                feats = np.array(feats_list)
            except Exception:
                return {}

        if len(feats) == 0:
            return {}

        histogram: dict[int, float] = {}
        quantized = np.floor(feats / self.bin_width).astype(np.int64)
        unique_rows, counts = np.unique(quantized, axis=0, return_counts=True)

        for level in range(self.num_levels + 1):
            if level < self.num_levels:
                weight = np.sqrt(1.0 / (2 ** (level + 1)))
            else:
                weight = np.sqrt(1.0 / (2**level))

            level_bytes = level.to_bytes(4, "little")

            for row, count in zip(unique_rows, counts):
                key_bytes = level_bytes + row.tobytes()
                h = (
                    int(
                        blake2b(
                            key_bytes, digest_size=self._digest_size
                        ).hexdigest(),
                        16,
                    )
                    & 0x7FFFFFFF
                )
                histogram[h] = histogram.get(h, 0.0) + float(count) * weight

            if level < self.num_levels:
                unique_rows //= 2
                unique_rows, inverse_indices = np.unique(
                    unique_rows, axis=0, return_inverse=True
                )
                counts = np.bincount(inverse_indices, weights=counts)

        return histogram

    def _create_sparse_matrix(self, all_features: list) -> csr_array:
        n_graphs = len(all_features)
        data = []
        indices = []
        indptr = [0]
        for features in all_features:
            sorted_features = sorted(features.items(), key=lambda x: x[0])
            for feature_idx, val in sorted_features:
                indices.append(feature_idx)
                data.append(val)
            indptr.append(len(indices))
        return csr_array(
            (
                np.array(data, dtype=np.float32),
                np.array(indices, dtype=np.int32),
                np.array(indptr, dtype=np.int32),
            ),
            shape=(n_graphs, self._feature_dim),
        )
