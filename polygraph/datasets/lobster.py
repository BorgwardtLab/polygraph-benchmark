from copy import deepcopy
from typing import Literal, Optional

import joblib
import networkx as nx
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
from tqdm.rich import tqdm

from polygraph.datasets.base import ProceduralGraphDataset, SplitGraphDataset
from polygraph.datasets.base.graph_storage import GraphStorage


def is_lobster_graph(graph: nx.Graph) -> bool:
    """Based on https://github.com/lrjconan/GRAN/blob/fc9c04a3f002c55acf892f864c03c6040947bc6b/utils/eval_helper.py#L426C3-L446C17"""
    graph = deepcopy(graph)
    if nx.is_tree(graph):
        leaves = [n for n, d in graph.degree() if d == 1]  # pyright: ignore
        graph.remove_nodes_from(leaves)

        leaves = [n for n, d in graph.degree() if d == 1]  # pyright: ignore
        graph.remove_nodes_from(leaves)

        num_nodes = len(graph.nodes())  # pyright: ignore
        num_degree_one = [d for n, d in graph.degree() if d == 1]  # pyright: ignore
        num_degree_two = [d for n, d in graph.degree() if d == 2]  # pyright: ignore

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (
            num_nodes - 2
        ):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        return False
    else:
        return False


class ProceduralLobsterGraphDataset(ProceduralGraphDataset):
    """Procedural version of [`LobsterGraphDataset`][polygraph.datasets.LobsterGraphDataset].

    Args:
        split: Split to load.
        num_graphs: Number of graphs to generate for this split.
        expected_num_nodes: Steers the expected number of nodes in the generated graphs.
        p1: Probability of adding an edge to the backbone.
        p2: Probability of adding an edge one level beyond the backbone.
        min_number_of_nodes: Minimum number of nodes in the generated graphs.
        max_number_of_nodes: Maximum number of nodes in the generated graphs.
        seed: Seed for the random number generator.
        memmap: Whether to use memory mapping for the dataset.
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        num_graphs: int,
        expected_num_nodes: int = 80,
        p1: float = 0.7,
        p2: float = 0.7,
        min_number_of_nodes: Optional[int] = 10,
        max_number_of_nodes: Optional[int] = 100,
        seed: int = 42,
        memmap: bool = False,
        show_generation_progress: bool = False,
    ):
        config_hash: str = joblib.hash(  # pyright: ignore
            (
                num_graphs,
                expected_num_nodes,
                p1,
                p2,
                min_number_of_nodes,
                max_number_of_nodes,
                seed,
                split,
            ),
            hash_name="md5",
        )
        self._rng = np.random.default_rng(
            int.from_bytes(config_hash.encode(), "big")
        )
        self._num_graphs = num_graphs
        self._expected_num_nodes = expected_num_nodes
        self._p1 = p1
        self._p2 = p2
        self._min_number_of_nodes = min_number_of_nodes
        self._max_number_of_nodes = max_number_of_nodes
        super().__init__(
            split,
            config_hash,
            memmap,
            show_generation_progress,
        )

    def generate_data(self) -> GraphStorage:
        graphs = [
            from_networkx(self._random_lobster())
            for _ in tqdm(
                range(self._num_graphs),
                desc="Generating lobster graphs",
                disable=not self.show_generation_progress,
            )
        ]
        return GraphStorage.from_pyg_batch(Batch.from_data_list(graphs))

    def _random_lobster(self):
        while True:
            g = nx.random_lobster(
                self._expected_num_nodes,
                self._p1,
                self._p2,
                seed=int(self._rng.integers(1e9)),
            )
            if (
                self._max_number_of_nodes is None
                or g.number_of_nodes() <= self._max_number_of_nodes
            ) and (
                self._min_number_of_nodes is None
                or g.number_of_nodes() >= self._min_number_of_nodes
            ):
                return g

    def is_valid(self, graph: nx.Graph) -> bool:
        """Check if a graph is a valid lobster graph."""
        return is_lobster_graph(graph)


class LobsterGraphDataset(SplitGraphDataset):
    """Dataset of lobster graphs proposed by Liao et al. [1].

    A lobster graph is a tree which has a backbone path such that each node in the tree is at most two hops away from this backbone.

    {{ plot_first_k_graphs("LobsterGraphDataset", "train", 3, node_size=50) }}

    Dataset statistics:

    {{ summary_md_table("LobsterGraphDataset", ["train", "val", "test"]) }}

    Warning:
        In the original dataset [1], the validation set was a subset of the training set. Here, we use disjoint splits.

    References:
        [1] Liao, R., Li, Y., Song, Y., Wang, S., Hamilton, W., Duvenaud, D., Urtasun, R., & Zemel, R. (2019). [Efficient Graph Generation with Graph Recurrent Attention Networks](https://arxiv.org/abs/1910.00760). In Advances in Neural Information Processing Systems (NeurIPS).
    """

    _URL_FOR_SPLIT = {
        "train": "https://sandbox.zenodo.org/records/332447/files/lobster_train.pt?download=1",
        "val": "https://sandbox.zenodo.org/records/332447/files/lobster_val.pt?download=1",
        "test": "https://sandbox.zenodo.org/records/332447/files/lobster_test.pt?download=1",
    }

    _HASH_FOR_SPLIT = {
        "train": "1b2d7b2ac5e6507d09550031865cd29c",
        "val": "ff7941058df7c9770509e1ee3ab42d28",
        "test": "4082bf3c380ab921481bf24f25a2934c",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    def is_valid(self, graph: nx.Graph) -> bool:
        """Check if a graph is a valid lobster graph."""
        return is_lobster_graph(graph)

    def hash_for_split(self, split: str) -> Optional[str]:
        return self._HASH_FOR_SPLIT[split]


class LobsterLGraphDataset(ProceduralLobsterGraphDataset):
    def __init__(
        self,
        split: Literal["train", "val", "test"],
        num_graphs: int,
        seed: int = 42,
        memmap: bool = False,
        show_generation_progress: bool = False,
    ):
        if split == "train":
            num_graphs = 8192
        elif split == "val":
            num_graphs = 4096
        elif split == "test":
            num_graphs = 4096

        super().__init__(
            split,
            num_graphs,
            seed,
            memmap,
            show_generation_progress,
        )
