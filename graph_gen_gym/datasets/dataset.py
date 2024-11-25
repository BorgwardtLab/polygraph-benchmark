# -*- coding: utf-8 -*-
"""dataset.py
Implementation of datasets.
"""

from abc import ABC, abstractmethod
from typing import List, Union

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from graph_gen_gym.datasets.graph import Graph
from graph_gen_gym.datasets.utils import download_to_cache, load_from_cache


class AbstractDataset(ABC):
    @property
    def identifier(self) -> str:
        return f"{self.__module__}.{self.__class__.__qualname__}"

    def to_nx(self) -> "NetworkXView":
        return NetworkXView(self)

    def is_valid(self, graph: nx.Graph) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Data:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class NetworkXView:
    def __init__(self, base_dataset: AbstractDataset):
        self._base_dataset = base_dataset

    def __len__(self):
        return len(self._base_dataset)

    def __getitem__(self, idx: int) -> nx.Graph:
        pyg_graph = self._base_dataset[idx]
        return to_networkx(pyg_graph, to_undirected=True)


class GraphDataset(AbstractDataset):
    def __init__(self, data_store: Graph):
        super().__init__()
        self._data_store = data_store

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Graph, List[Graph]]:
        if isinstance(idx, int):
            return self._data_store.get_example(idx)
        return [self._data_store.get_example(i) for i in idx]

    def __len__(self):
        return len(self._data_store)


class OnlineGraphDataset(GraphDataset):
    def __init__(self, split: str, memmap: bool = False):
        try:
            storage = load_from_cache(self.identifier, split, mmap=memmap)
        except FileNotFoundError:
            download_to_cache(self.url_for_split(split), self.identifier, split)
            storage = load_from_cache(self.identifier, split, mmap=memmap)
        super().__init__(storage)

    @abstractmethod
    def url_for_split(self, split: str):
        ...
