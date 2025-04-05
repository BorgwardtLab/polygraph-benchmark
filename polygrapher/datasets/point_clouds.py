import networkx as nx

from polygrapher.datasets.base import OnlineGraphDataset


class PointCloudGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/ccnBfchstXblFCl/download",
        "val": "https://datashare.biochem.mpg.de/s/qYpuHH3HhhYAimi/download",
        "test": "https://datashare.biochem.mpg.de/s/w1CKclswdcxLbpK/download",
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
