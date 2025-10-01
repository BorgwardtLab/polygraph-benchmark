from typing import Optional

from polygraph.datasets.base import SplitGraphDataset


class EgoGraphDataset(SplitGraphDataset):
    """Dataset of ego networks extracted from Citeseer [1], introduced by You et al. [2].

    The graphs are 3-hop ego networks with 50 to 399 nodes.

    {{ plot_first_k_graphs("EgoGraphDataset", "train", 3) }}

    Dataset statistics:

    {{ summary_md_table("EgoGraphDataset", ["train", "val", "test"]) }}

    References:
        [1] Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B., and Eliassi-Rad, T. (2008).
            Collective classification in network data. AI Magazine, 29(3):93.

        [2] You, J., Ying, R., Ren, X., Hamilton, W., & Leskovec, J. (2018).
            [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models](https://arxiv.org/abs/1802.08773).
            In International Conference on Machine Learning (ICML).
    """

    _URL_FOR_SPLIT = {
        "train": "https://sandbox.zenodo.org/records/332447/files/ego_train.pt?download=1",
        "val": "https://sandbox.zenodo.org/records/332447/files/ego_val.pt?download=1",
        "test": "https://sandbox.zenodo.org/records/332447/files/ego_test.pt?download=1",
    }

    _HASH_FOR_SPLIT = {
        "train": "143e014a419825059b6edccc5a6fd43a",
        "val": "2774618f6c05f60c87c90f7730cac03c",
        "test": "90e147b0fd2deb37b8922c1d2ab10a74",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    def hash_for_split(self, split: str) -> Optional[str]:
        return self._HASH_FOR_SPLIT[split]


class SmallEgoGraphDataset(SplitGraphDataset):
    """Dataset of smaller ego networks extracted from Citeseer.

    The graphs of this dataset have at most 18 nodes.

    {{ plot_first_k_graphs("SmallEgoGraphDataset", "train", 3) }}

    Dataset statistics:

    {{ summary_md_table("SmallEgoGraphDataset", ["train", "val", "test"]) }}

    """

    _URL_FOR_SPLIT = {
        "train": "https://sandbox.zenodo.org/records/332447/files/ego_small_train.pt?download=1",
        "val": "https://sandbox.zenodo.org/records/332447/files/ego_small_val.pt?download=1",
        "test": "https://sandbox.zenodo.org/records/332447/files/ego_small_test.pt?download=1",
    }

    _HASH_FOR_SPLIT = {
        "train": "703d9f95ef74a66f6d261ade0ac1fcdd",
        "val": "34fbe4cdd6b22ed3d3a494a392e76d4c",
        "test": "96ff64b4ef87a31ef7e15f489426a50e",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    def hash_for_split(self, split: str) -> Optional[str]:
        return self._HASH_FOR_SPLIT[split]
