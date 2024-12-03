import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.sparse import csr_array

import graph_gen_gym


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
