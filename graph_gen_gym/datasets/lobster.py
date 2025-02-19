from copy import deepcopy

import networkx as nx
import joblib
import numpy as np
from typing import Optional, Literal
from torch_geometric.data import Batch

from graph_gen_gym.datasets.base import OnlineGraphDataset, ProceduralGraphDataset
from graph_gen_gym.datasets.base.graph import Graph
from torch_geometric.utils import from_networkx


def is_lobster_graph(graph: nx.Graph) -> bool:
    """Based on https://github.com/lrjconan/GRAN/blob/fc9c04a3f002c55acf892f864c03c6040947bc6b/utils/eval_helper.py#L426C3-L446C17"""
    graph = deepcopy(graph)
    if nx.is_tree(graph):
        leaves = [n for n, d in graph.degree() if d == 1]
        graph.remove_nodes_from(leaves)

        leaves = [n for n, d in graph.degree() if d == 1]
        graph.remove_nodes_from(leaves)

        num_nodes = len(graph.nodes())
        num_degree_one = [d for n, d in graph.degree() if d == 1]
        num_degree_two = [d for n, d in graph.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        return False
    else:
        return False


class ProceduralLobsterGraphDataset(ProceduralGraphDataset):
    def __init__(self, split: Literal["train", "val", "test"], num_graphs: int, expected_num_nodes: int = 80, p1: float = 0.7, p2: float = 0.7, min_number_of_nodes: Optional[int] = 10, max_number_of_nodes: Optional[int] = 100, seed: int = 42, memmap: bool = False):
        config_hash = joblib.hash((num_graphs, expected_num_nodes, p1, p2, min_number_of_nodes, max_number_of_nodes, seed, split), hash_name='md5')
        self._rng = np.random.default_rng(int.from_bytes(config_hash.encode(), 'big'))
        self._num_graphs = num_graphs
        self._expected_num_nodes = expected_num_nodes
        self._p1 = p1
        self._p2 = p2
        self._min_number_of_nodes = min_number_of_nodes
        self._max_number_of_nodes = max_number_of_nodes
        super().__init__(split, config_hash, memmap)

    def generate_data(self) -> Graph:
        graphs = [from_networkx(self._random_lobster()) for _ in range(self._num_graphs)]
        return Graph.from_pyg_batch(Batch.from_data_list(graphs))

    @staticmethod
    def is_valid(graph: nx.Graph) -> bool:
        return is_lobster_graph(graph)

    def _random_lobster(self):
        while True:
            g = nx.random_lobster(self._expected_num_nodes, self._p1, self._p2, seed=int(self._rng.integers(1e9)))
            if (
                self._max_number_of_nodes is None or g.number_of_nodes() <= self._max_number_of_nodes
            ) and (
                self._min_number_of_nodes is None or g.number_of_nodes() >= self._min_number_of_nodes
            ):
                return g


class LobsterGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/mU8mA2GqfssxUFt/download",
        "val": "https://datashare.biochem.mpg.de/s/KTicVKdP6LgTKeV/download",
        "test": "https://datashare.biochem.mpg.de/s/eYS8K0E6IQ7gZ7j/download",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    @staticmethod
    def is_valid(graph: nx.Graph) -> bool:
        return is_lobster_graph(graph)
