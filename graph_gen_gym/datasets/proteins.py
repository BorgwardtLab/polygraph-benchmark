import networkx as nx

from graph_gen_gym.datasets.base import OnlineGraphDataset


class DobsonDoigGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/IUzyKrF6T1wjqqG/download",
        "val": "https://datashare.biochem.mpg.de/s/NhaictDUDb7UTpr/download",
        "test": "https://datashare.biochem.mpg.de/s/ecJCDZVTNOpbvy4/download",
    }

    _HASH_FOR_SPLIT = {
        "train": None,
        "val": None,
        "test": None,
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    @staticmethod
    def is_valid(graph: nx.Graph):
        return graph.number_of_nodes() > 0 and graph.number_of_edges() > 0

    def hash_for_split(self, split: str) -> str:
        return self._HASH_FOR_SPLIT[split]
