import os
import pickle as pkl
import random
import tempfile
import urllib.request

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx

from polygrapher.datasets.base import GraphStorage


def parse_index_file(filename):
    """From https://github.com/JiaxuanYou/graph-generation/blob/3444b8ad2fd7ecb6ade45086b4c75f8e2e9f29d1/data.py#L96"""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _graph_load(datadir, dataset):
    """Based on https://github.com/JiaxuanYou/graph-generation/blob/3444b8ad2fd7ecb6ade45086b4c75f8e2e9f29d1/data.py#L103C1-L131."""

    names = ["x", "tx", "allx", "graph"]
    objects = []
    for i in range(len(names)):
        load = pkl.load(
            open(os.path.join(datadir, "ind.{}.{}".format(dataset, names[i])), "rb"),
            encoding="latin1",
        )
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        os.path.join(datadir, "ind.{}.test.index".format(dataset))
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G


def _connected_component_subgraphs(g):
    for node_set in nx.connected_components(g):
        yield g.subgraph(node_set)


def _citeseer_to_egos(citeseer_graph, small=False):
    """Based on https://github.com/JiaxuanYou/graph-generation/blob/3444b8ad2fd7ecb6ade45086b4c75f8e2e9f29d1/create_graphs.py#L131C9-L138C37"""
    G = max(_connected_component_subgraphs(citeseer_graph), key=len)
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        if small:
            G_ego = nx.ego_graph(G, i, radius=1)
            if (G_ego.number_of_nodes() >= 4) and (G_ego.number_of_nodes() <= 20):
                graphs.append(G_ego)
        else:
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)

    if small:
        random.seed(123)
        random.shuffle(graphs)
        graphs = graphs[0:200]
    return graphs


def _create_ego_graphs(small=False):
    url = "https://github.com/JiaxuanYou/graph-generation/raw/3444b8ad2fd7ecb6ade45086b4c75f8e2e9f29d1/dataset"
    with tempfile.TemporaryDirectory() as tmpdir:
        extensions = ["x", "tx", "allx", "graph"]
        for extension in extensions:
            urllib.request.urlretrieve(
                f"{url}/ind.citeseer.{extension}",
                os.path.join(tmpdir, f"ind.citeseer.{extension}"),
            )

        urllib.request.urlretrieve(
            f"{url}/ind.citeseer.test.index",
            os.path.join(tmpdir, "ind.citeseer.test.index"),
        )

        _, _, citeseer_graph = _graph_load(tmpdir, "citeseer")
        egos = _citeseer_to_egos(citeseer_graph, small=small)
    return egos


def _get_ego_splits(small=False):
    """Splitting logic from https://github.com/JiaxuanYou/graph-generation/blob/3444b8ad2fd7ecb6ade45086b4c75f8e2e9f29d1/main.py#L33-L38"""
    egos = _create_ego_graphs(small=small)
    assert (small and len(egos) == 200) or (not small and len(egos) == 757)
    trainval_len = int(0.8 * len(egos))
    val_len = int(0.2 * len(egos))

    random.seed(123)
    random.shuffle(egos)
    graphs_test = Batch.from_data_list([from_networkx(g) for g in egos[trainval_len:]])
    graphs_train = Batch.from_data_list(
        [from_networkx(g) for g in egos[val_len:trainval_len]]
    )
    graphs_validate = Batch.from_data_list([from_networkx(g) for g in egos[0:val_len]])

    return (
        GraphStorage.from_pyg_batch(graphs_train),
        GraphStorage.from_pyg_batch(graphs_validate),
        GraphStorage.from_pyg_batch(graphs_test),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, required=True)
    parser.add_argument("--small", action="store_true", default=False)
    args = parser.parse_args()

    print("Using small:", args.small)
    train, val, test = _get_ego_splits(small=args.small)

    os.makedirs(args.destination, exist_ok=True)
    torch.save(train.model_dump(), os.path.join(args.destination, "train.pt"))
    torch.save(val.model_dump(), os.path.join(args.destination, "val.pt"))
    torch.save(test.model_dump(), os.path.join(args.destination, "test.pt"))
