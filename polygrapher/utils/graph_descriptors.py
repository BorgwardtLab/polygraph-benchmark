"""Graph descriptor functions for converting graphs into feature vectors.

This module provides various functions that convert networkx graphs into numerical 
representations suitable for kernel methods. Each descriptor is callable with an iterable of graphs and returns either a dense 
`numpy.ndarray` or sparse `scipy.sparse.csr_array` of shape `(n_graphs, n_features)`.

Available descriptors:
    - [`SparseDegreeHistogram`][polygrapher.utils.graph_descriptors.SparseDegreeHistogram]: Sparse degree distribution
    - [`DegreeHistogram`][polygrapher.utils.graph_descriptors.DegreeHistogram]: Dense degree distribution
    - [`ClusteringHistogram`][polygrapher.utils.graph_descriptors.ClusteringHistogram]: Distribution of clustering coefficients
    - [`OrbitCounts`][polygrapher.utils.graph_descriptors.OrbitCounts]: Graph orbit statistics
    - [`EigenvalueHistogram`][polygrapher.utils.graph_descriptors.EigenvalueHistogram]: Eigenvalue histogram of normalized Laplacian
    - [`RandomGIN`][polygrapher.utils.graph_descriptors.RandomGIN]: Embeddings of random Graph Isomorphism Network
    - [`WeisfeilerLehmanDescriptor`][polygrapher.utils.graph_descriptors.WeisfeilerLehmanDescriptor]: Weisfeiler-Lehman subtree features
    - [`NormalizedDescriptor`][polygrapher.utils.graph_descriptors.NormalizedDescriptor]: Standardized descriptor wrapper
"""

import copy
import warnings
from collections import Counter
from hashlib import blake2b
from typing import Callable, Iterable, List, Optional

import networkx as nx
import numpy as np
import orbit_count
import torch
from scipy.sparse import csr_array
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch
from torch_geometric.utils import degree, from_networkx

from polygrapher.utils.gin import GIN
from polygrapher.utils.parallel import batched_distribute_function, flatten_lists


class DegreeHistogram:
    """Computes normalized degree distributions of graphs.
    
    For each graph, computes a histogram of node degrees and normalizes it to sum to 1.
    Pads all histograms to a fixed maximum degree.
    
    Args:
        max_degree: Maximum degree to consider. Larger degrees are ignored
    """

    def __init__(self, max_degree: int):
        self._max_degree = max_degree

    def __call__(self, graphs: Iterable[nx.Graph]):
        hists = [nx.degree_histogram(graph) for graph in graphs]
        hists = [
            np.concatenate([hist, np.zeros(self._max_degree - len(hist))], axis=0)
            for hist in hists
        ]
        hists = np.stack(hists, axis=0)
        return hists / hists.sum(axis=1, keepdims=True)


class SparseDegreeHistogram:
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
            (np.concatenate(data), np.concatenate(index), ptr), (len(graphs), 100_000)
        )
        return result


class ClusteringHistogram:
    """Computes histograms of local clustering coefficients.
    
    For each graph, computes the distribution of local clustering coefficients
    across nodes. The clustering coefficient measures the fraction of possible
    triangles through each node that exist.
    
    Args:
        bins: Number of histogram bins covering [0,1]
    """

    def __init__(self, bins: int):
        self._num_bins = bins

    def __call__(self, graphs: Iterable[nx.Graph]):
        all_clustering_coeffs = [
            list(nx.clustering(graph).values()) for graph in graphs
        ]
        hists = [
            np.histogram(
                clustering_coeffs, bins=self._num_bins, range=(0.0, 1.0), density=False
            )[0]
            for clustering_coeffs in all_clustering_coeffs
        ]
        hists = np.stack(hists, axis=0)
        return hists / hists.sum(axis=1, keepdims=True)


class OrbitCounts:
    """Computes graph orbit statistics .
    
    Warning:
        Self-loops are automatically removed from input graphs.
    """

    def __call__(self, graphs: Iterable[nx.Graph]):
        # Check if any graph has a self-loop
        self_loops = [list(nx.selfloop_edges(g)) for g in graphs]
        if any(len(loops) > 0 for loops in self_loops):
            warnings.warn(
                "Graph with self-loop passed to orbit descriptor, deleting self-loops"
            )
            graphs = [copy.deepcopy(g) for g in graphs]
            for g, loops in zip(graphs, self_loops):
                g.remove_edges_from(loops)

        counts = orbit_count.batched_node_orbit_counts(graphs, graphlet_size=4)
        counts = [count.mean(axis=0) for count in counts]
        return np.stack(counts, axis=0)


class EigenvalueHistogram:
    """Computes eigenvalue histogram of normalized Laplacian.
    
    For each graph, computes the eigenvalue spectrum of its normalized Laplacian
    matrix and returns a histogram of the eigenvalues. 
    """

    def __call__(self, graphs: Iterable[nx.Graph]):
        histograms = []
        for g in graphs:
            eigs = np.linalg.eigvalsh(nx.normalized_laplacian_matrix(g).todense())
            spectral_pmf, _ = np.histogram(
                eigs, bins=200, range=(-1e-5, 2), density=False
            )
            spectral_pmf = spectral_pmf / spectral_pmf.sum()
            histograms.append(spectral_pmf)
        return np.stack(histograms, axis=0)


class RandomGIN:
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
    def __call__(self, graphs: Iterable[nx.Graph]):
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
            edge_attr = torch.cat([g.edge_attr for g in pyg_graphs]).to(self._device)

        batch = Batch.from_data_list(pyg_graphs).to(self._device)

        graph_embeds = self._feat_fn(
            feats, batch.edge_index, batch.batch, edge_attr=edge_attr
        )
        return graph_embeds.cpu().detach().numpy()


class NormalizedDescriptor:
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
        descriptor_fn: Callable[[Iterable[nx.Graph]], np.ndarray],
        ref_graphs: Iterable[nx.Graph],
    ):
        self._descriptor_fn = descriptor_fn
        self._scaler = StandardScaler()
        self._scaler.fit(self._descriptor_fn(ref_graphs))

    def __call__(self, graphs: Iterable[nx.Graph]):
        result = self._descriptor_fn(graphs)
        return self._scaler.transform(result)


class WeisfeilerLehmanDescriptor:
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

        self._iterations = iterations
        self._use_node_labels = use_node_labels
        self._node_label_key = node_label_key if use_node_labels else "degree"
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
                graph.nodes[node][self._node_label_key] = graph.degree(node)

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
                isinstance(hash_key, str) and len(hash_key) == 2 * self._digest_size
            ), "Hash key is not a hex string or has incorrect length"
            int_key = int(hash_key, 16)
            int_key = int_key & 0x7FFFFFFF
            int_hashes[int_key] = count
            assert (
                0 <= int_key <= (2**31 - 1)
            ), f"Unexpected hash key {int_key} out of bounds"

        if len(int_hashes) != len(all_hashes):
            # This might artificially inflate the resulting kernel value but not
            # by much in our experiments.
            warnings.warn("Hash collision detected in Weisfeiler-Lehman descriptor")
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
