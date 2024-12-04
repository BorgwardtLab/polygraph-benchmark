import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import dgl
import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_array
from sklearn.preprocessing import StandardScaler

import graph_gen_gym
from graph_gen_gym.metrics.utils.gin import GIN


def _edge_list_reindexed(graph: nx.Graph):
    idx = 0
    id2idx = dict()
    for u in graph.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in graph.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def _orbit_descriptor(graph: nx.Graph) -> np.ndarray:
    tmp, fname = tempfile.mkstemp()
    try:
        os.close(tmp)
        with open(fname, "w") as tmp:
            tmp.write(
                str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n"
            )
            for u, v in _edge_list_reindexed(graph):
                tmp.write(str(u) + " " + str(v) + "\n")
        exec_path = str(Path(graph_gen_gym.__file__).parent.joinpath("orca"))
        output = subprocess.check_output([exec_path, "node", "4", fname, "std"])
    finally:
        os.unlink(fname)

    output = output.decode("utf8").strip()
    idx = output.find("orbit counts:") + len("orbit counts:") + 2
    output = output[idx:]
    node_orbit_counts = np.array(
        [
            list(map(int, node_cnts.strip().split(" ")))
            for node_cnts in output.strip("\n").split("\n")
        ]
    )

    return node_orbit_counts.sum(axis=0) / graph.number_of_nodes()


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
        index = [np.nonzero(hist)[0] for hist in hists]
        data = [hist[idx] for hist, idx in zip(hists, index)]
        ptr = np.zeros(len(index) + 1)
        ptr[1:] = np.cumsum([len(idx) for idx in index])
        result = csr_array(
            (np.concatenate(data), np.concatenate(index), ptr), (len(graphs), 1_000_000)
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
    def __init__(self, num_processes: int = 0):
        if num_processes > 0:
            raise NotImplementedError

    def __call__(self, graphs: Iterable[nx.Graph]):
        descriptors = [_orbit_descriptor(graph) for graph in graphs]
        return np.stack(descriptors, axis=0)


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
        node_feat_loc: str = "attr",
        edge_feat_loc: str = "attr",
        init: str = "orthogonal",
        device: str = "cpu",
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
        )

        self.model.node_feat_loc = node_feat_loc
        self.model.edge_feat_loc = edge_feat_loc

        self.model.eval()

        if dont_concat:
            self.model.forward = self.model.get_graph_embed_no_cat
        else:
            self.model.forward = self.model.get_graph_embed

        self.model.device = device
        self.model = self.model.to(device)

    @torch.inference_mode()
    def __call__(self, graphs: Iterable[nx.Graph]):
        node_feat_loc = self.model.node_feat_loc
        edge_feat_loc = self.model.edge_feat_loc

        dgl_graphs = [dgl.from_networkx(g) for g in graphs]

        ndata = [node_feat_loc] if node_feat_loc in dgl_graphs[0].ndata else "__ALL__"
        edata = [edge_feat_loc] if edge_feat_loc in dgl_graphs[0].edata else "__ALL__"
        graphs = dgl.batch(dgl_graphs, ndata=ndata, edata=edata).to(self.model.device)

        if node_feat_loc not in graphs.ndata:  # Use degree as features
            feats = graphs.in_degrees() + graphs.out_degrees()
            feats = feats.unsqueeze(1).type(torch.float32)
        else:
            feats = graphs.ndata[node_feat_loc]
        feats = feats.to(self.model.device)

        graph_embeds = self.model(graphs, feats)
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
