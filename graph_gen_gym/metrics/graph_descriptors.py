import os
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np

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


def orbit_descriptor(graph: nx.Graph) -> np.ndarray:
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


def degree_descriptor(graph: nx.Graph) -> np.ndarray:
    hist = np.array(nx.degree_histogram(graph))
    return hist / hist.sum()


def clustering_descriptor(graph: nx.Graph, bins: int) -> np.ndarray:
    clustering_coeffs_list = list(nx.clustering(graph).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist / hist.sum()


def spectral_descriptor(graph: nx.Graph) -> np.ndarray:
    pas
