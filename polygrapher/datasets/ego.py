import networkx as nx

from polygrapher.datasets.base import OnlineGraphDataset


class EgoGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/F0MGpYS7sMGjMIS/download",
        "val": "https://datashare.biochem.mpg.de/s/o5wq4MRMTsA9uu3/download",
        "test": "https://datashare.biochem.mpg.de/s/bASBL8VCUVm2jai/download",
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


class SmallEgoGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/RtsHhHBTFkZMIap/download",
        "val": "https://datashare.biochem.mpg.de/s/dWUWhuRj1ipGOVw/download",
        "test": "https://datashare.biochem.mpg.de/s/ey00DsRG1Zm7SQt/download",
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
