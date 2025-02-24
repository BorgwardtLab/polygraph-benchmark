import numpy as np
import networkx as nx
import joblib
from graph_gen_gym.datasets.base import OnlineGraphDataset, ProceduralGraphDataset
from graph_gen_gym.datasets.base.graph import Graph
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch
from typing import Literal


class ProceduralPlanarGraphDataset(ProceduralGraphDataset):
    def __init__(self, split: Literal["train", "val", "test"], num_graphs: int, n_nodes: int = 64, seed: int = 42, memmap: bool = False):
        config_hash = joblib.hash((num_graphs, n_nodes, seed, split), hash_name='md5')
        self._rng = np.random.default_rng(int.from_bytes(config_hash.encode(), 'big'))
        self._num_graphs = num_graphs
        self._n_nodes = n_nodes
        super().__init__(split, config_hash, memmap)

    def generate_data(self) -> Graph:
        graphs = [from_networkx(self._random_planar()) for _ in range(self._num_graphs)]
        return Graph.from_pyg_batch(Batch.from_data_list(graphs))

    @staticmethod
    def is_valid(graph: nx.Graph) -> bool:
        return nx.is_connected(graph) and nx.is_planar(graph)

    def _random_planar(self):
        import scipy

        node_locations = self._rng.uniform(size=(self._n_nodes, 2))
        # Create the delaunay triangulation
        triangulation = scipy.spatial.Delaunay(node_locations)
        graph = nx.Graph()
        graph.add_nodes_from(range(self._n_nodes))
        graph.add_edges_from(
            (s[i], s[j])
            for s in triangulation.simplices
            for i in range(3)
            for j in range(3)
            if i < j
        )
        return graph


class PlanarGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/f3kXPP4LICWKbBx/download",
        "val": "https://datashare.biochem.mpg.de/s/CN2zeY8EvIlUxN6/download",
        "test": "https://datashare.biochem.mpg.de/s/DNwtgo3mlOErxHX/download",
    }

    _HASH_FOR_SPLIT = {
        "train": None,
        "val": None,
        "test": None,
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    @staticmethod
    def is_valid(graph: nx.Graph) -> bool:
        return nx.is_connected(graph) and nx.is_planar(graph)

    def hash_for_split(self, split: str) -> str:
        return self._HASH_FOR_SPLIT[split]
