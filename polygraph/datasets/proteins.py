import networkx as nx

from polygraph.datasets.base import OnlineGraphDataset


class DobsonDoigGraphDataset(OnlineGraphDataset):
    """Dataset of protein graphs originally introduced by Dobson and Doig [1].

    This dataset was later adopted by You et al. [2] in the area of graph generation. The splits we provide are disjoint, unlike in [2].
    We use the splitting strategy proposed in [3].

    Available splits:
        - `train`: 587 graphs
        - `val`: 147 graphs
        - `test`: 184 graphs

    Graph Attributes:
        - `residues`: Node-level attribute indicating the amino acid types
        - `is_enyzme`: Graph-level attribute indicating whether protein is an enzyme (1 or 2)


    References:
        [1] Dobson, P. and Doig, A. (2003).
            [Distinguishing enzyme structures from non-enzymes without alignments](https://doi.org/10.1016/S0022-2836(03)00628-4).
            Journal of Molecular Biology, 330(4):771–783.

        [2] You, J., Ying, R., Ren, X., Hamilton, W., & Leskovec, J. (2018). 
            [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models](https://arxiv.org/abs/1802.08773). 
            In International Conference on Machine Learning (ICML).

        [3] Martinkus, K., Loukas, A., Perraudin, N., & Wattenhofer, R. (2022).
            [SPECTRE: Spectral Conditioning Helps to Overcome the Expressivity Limits 
            of One-shot Graph Generators](https://arxiv.org/abs/2204.01613). In Proceedings of the 39th International 
            Conference on Machine Learning (ICML).

    """
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
