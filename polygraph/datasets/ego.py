import networkx as nx

from polygraph.datasets.base import OnlineGraphDataset


class EgoGraphDataset(OnlineGraphDataset):
    """Dataset of ego networks extracted from Citeseer [1], introduced by You et al. [2].

    The graphs are 3-hop ego networks with 50 to 399 nodes.

    Available splits:
        - `train`: 454 graphs
        - `val`: 151 ego networks
        - `test`: 152 ego networks

    References:
        [1] Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B., and Eliassi-Rad, T. (2008).
            Collective classification in network data. AI Magazine, 29(3):93.

        [2] You, J., Ying, R., Ren, X., Hamilton, W., & Leskovec, J. (2018).
            [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models](https://arxiv.org/abs/1802.08773).
            In International Conference on Machine Learning (ICML).
    """

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
    """Dataset of smaller ego networks extracted from Citeseer.

    The graphs of this dataset have at most 18 nodes.

    Available splits:
        - `train`: 120 graphs
        - `val`: 40 graphs
        - `test`: 40 graphs
    """

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
