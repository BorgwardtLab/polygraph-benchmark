import os

import networkx as nx
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx

from graph_gen_gym.datasets.base import Graph


def generate_lobster_graphs(num_graphs=100, seed=1234):
    "Based on https://github.com/lrjconan/GRAN/blob/fc9c04a3f002c55acf892f864c03c6040947bc6b/utils/data_helper.py#L169-L190"
    graphs = []
    p1 = 0.7
    p2 = 0.7
    count = 0
    min_node = 10
    max_node = 100
    max_edge = 0
    mean_node = 80

    seed_tmp = seed
    while count < num_graphs:
        G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
        if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
            graphs.append(G)
            if G.number_of_edges() > max_edge:
                max_edge = G.number_of_edges()

            count += 1

        seed_tmp += 1
    return graphs


def _generate_gran_splits(graphs):
    num_graphs = len(graphs)
    train_ratio = 0.8
    dev_ratio = 0.2
    num_train = int(float(num_graphs) * train_ratio)
    num_dev = int(float(num_graphs) * dev_ratio)

    graphs_train = Batch.from_data_list(
        [from_networkx(g) for g in graphs[num_dev:num_train]]
    )
    graphs_dev = Batch.from_data_list([from_networkx(g) for g in graphs[:num_dev]])
    graphs_test = Batch.from_data_list([from_networkx(g) for g in graphs[num_train:]])
    return (
        Graph.from_pyg_batch(graphs_train),
        Graph.from_pyg_batch(graphs_dev),
        Graph.from_pyg_batch(graphs_test),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, required=True)
    args = parser.parse_args()

    train, val, test = _generate_gran_splits(generate_lobster_graphs())
    os.makedirs(args.destination, exist_ok=True)
    torch.save(train.model_dump(), os.path.join(args.destination, "train.pt"))
    torch.save(val.model_dump(), os.path.join(args.destination, "val.pt"))
    torch.save(test.model_dump(), os.path.join(args.destination, "test.pt"))
