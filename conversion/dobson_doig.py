import os
import tempfile
import urllib.request

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx

from polygrapher.datasets.base import Graph


def _nx_graphs_to_storage(nx_graphs):
    pyg_graphs = [from_networkx(g, group_node_attrs=["node_label"]) for g in nx_graphs]
    batch = Batch.from_data_list(pyg_graphs)
    batch.residues = batch.x
    batch.is_enzyme = torch.Tensor([g.graph["graph_label"] for g in nx_graphs]).to(
        torch.int
    )
    return Graph.from_pyg_batch(
        batch,
        node_attrs=["residues"],
        graph_attrs=["is_enzyme"],
    )


def _get_dobson_doig_storages():
    """Based on https://github.com/KarolisMart/SPECTRE/blob/main/data.py."""

    min_num_nodes = 100
    max_num_nodes = 500

    url = "https://github.com/KarolisMart/SPECTRE/raw/refs/heads/main/data/DD"
    fnames = [
        "DD_A.txt",
        "DD_node_labels.txt",
        "DD_graph_indicator.txt",
        "DD_graph_labels.txt",
    ]

    all_nx_graphs = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in fnames:
            fpath = os.path.join(tmpdir, fname)
            urllib.request.urlretrieve(f"{url}/{fname}", fpath)

        data_adj = np.loadtxt(os.path.join(tmpdir, "DD_A.txt"), delimiter=",").astype(
            int
        )
        data_node_label = np.loadtxt(
            os.path.join(tmpdir, "DD_node_labels.txt"), delimiter=","
        ).astype(int)
        data_graph_indicator = np.loadtxt(
            os.path.join(tmpdir, "DD_graph_indicator.txt"), delimiter=","
        ).astype(int)
        data_graph_types = np.loadtxt(
            os.path.join(tmpdir, "DD_graph_labels.txt"), delimiter=","
        ).astype(int)

        data_tuple = list(map(tuple, data_adj))

        G = nx.Graph()

        # Add edges
        G.add_edges_from(data_tuple)
        assert G.number_of_nodes() == len(data_node_label)

        for n in G.nodes:
            G.nodes[n]["node_label"] = data_node_label[n - 1]

        G.remove_nodes_from(list(nx.isolates(G)))

        # remove self-loop
        G.remove_edges_from(nx.selfloop_edges(G))

        # Split into graphs
        graph_num = data_graph_indicator.max()
        node_list = np.arange(data_graph_indicator.shape[0]) + 1

        for i in range(graph_num):
            # Find the nodes for each graph
            nodes = node_list[data_graph_indicator == i + 1]
            G_sub = G.subgraph(nodes)
            G_sub.graph["graph_label"] = data_graph_types[i]
            if (
                G_sub.number_of_nodes() >= min_num_nodes
                and G_sub.number_of_nodes() <= max_num_nodes
            ):
                all_nx_graphs.append(G_sub.copy())

    test_len = int(round(len(all_nx_graphs) * 0.2))
    train_len = int(round((len(all_nx_graphs) - test_len) * 0.8))
    val_len = len(all_nx_graphs) - train_len - test_len
    train, val, test = torch.utils.data.random_split(
        all_nx_graphs,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(1234),
    )
    return (
        _nx_graphs_to_storage(train),
        _nx_graphs_to_storage(val),
        _nx_graphs_to_storage(test),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.destination, exist_ok=True)
    train, val, test = _get_dobson_doig_storages()
    torch.save(train.model_dump(), os.path.join(args.destination, "train.pt"))
    torch.save(val.model_dump(), os.path.join(args.destination, "val.pt"))
    torch.save(test.model_dump(), os.path.join(args.destination, "test.pt"))
