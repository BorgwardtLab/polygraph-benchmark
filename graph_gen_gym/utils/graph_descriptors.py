from typing import Callable, Iterable, Optional, List

import networkx as nx
import numpy as np
import orbit_count
import torch
from scipy.sparse import csr_array
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch
from torch_geometric.utils import degree, from_networkx

from graph_gen_gym.utils.gin import GIN


class DegreeHistogram:
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
    def __call__(self, graphs: Iterable[nx.Graph]):
        counts = orbit_count.batched_node_orbit_counts(graphs, graphlet_size=4)
        counts = [count.mean(axis=0) for count in counts]
        return np.stack(counts, axis=0)


class EigenvalueHistogram:
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
        init: str = "orthogonal",
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
            init=init,
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
            feats = torch.cat(
                [
                    g.x
                    for g in pyg_graphs
                ]
            ).to(self._device)

        if self.edge_feat_loc is None:
            edge_attr = None
        else:
            edge_attr = torch.cat(
                [
                    g.edge_attr
                    for g in pyg_graphs
                ]
            ).to(self._device)

        batch = Batch.from_data_list(pyg_graphs).to(self._device)

        graph_embeds = self._feat_fn(
            feats, batch.edge_index, batch.batch, edge_attr=edge_attr
        )
        return graph_embeds.cpu().detach().numpy()


class NormalizedDescriptor:
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
