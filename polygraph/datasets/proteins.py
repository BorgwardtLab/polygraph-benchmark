from typing import Optional

from polygraph.datasets.base import SplitGraphDataset


class DobsonDoigGraphDataset(SplitGraphDataset):
    """Dataset of protein graphs originally introduced by Dobson and Doig [1].

    This dataset was later adopted by You et al. [2] in the area of graph generation. The splits we provide are disjoint, unlike in [2].
    We use the splitting strategy proposed in [3].

    {{ plot_first_k_graphs("DobsonDoigGraphDataset", "train", 3, node_size=25) }}


    Dataset statistics:

    {{ summary_md_table("DobsonDoigGraphDataset", ["train", "val", "test"]) }}


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
        "train": "https://sandbox.zenodo.org/records/332447/files/dobson_doig_train.pt?download=1",
        "val": "https://sandbox.zenodo.org/records/332447/files/dobson_doig_val.pt?download=1",
        "test": "https://sandbox.zenodo.org/records/332447/files/dobson_doig_test.pt?download=1",
    }

    _HASH_FOR_SPLIT = {
        "train": "417af7f6d0b66a3a5247c217e1db4b35",
        "val": "64140c8f2e2e022eb7286f53bf51472c",
        "test": "1b68e61383048549c5d69f918d70272e",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    def hash_for_split(self, split: str) -> Optional[str]:
        return self._HASH_FOR_SPLIT[split]
