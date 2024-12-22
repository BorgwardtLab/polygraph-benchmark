# -*- coding: utf-8 -*-
"""dataset.py
Implementation of datasets.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Callable

import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx

from graph_gen_gym.datasets.base.graph import Graph
from graph_gen_gym.datasets.base.caching import download_to_cache, load_from_cache
import numpy as np

class AbstractDataset(ABC):
    @property
    def identifier(self) -> str:
        return f"{self.__module__}.{self.__class__.__qualname__}"

    def to_nx(self) -> "NetworkXView":
        return NetworkXView(self)

    @staticmethod
    @abstractmethod
    def is_valid(graph: nx.Graph) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Data: ...

    @abstractmethod
    def __len__(self) -> int: ...


class NetworkXView:
    def __init__(self, base_dataset: AbstractDataset):
        self._base_dataset = base_dataset

    def __len__(self):
        return len(self._base_dataset)

    def __getitem__(self, idx: int) -> nx.Graph:
        pyg_graph = self._base_dataset[idx]
        return to_networkx(
            pyg_graph,
            node_attrs=list(self._base_dataset._data_store.node_attr.keys()),
            edge_attrs=list(self._base_dataset._data_store.edge_attr.keys()),
            graph_attrs=list(self._base_dataset._data_store.graph_attr.keys()),
            to_undirected=True,
        )


class GraphDataset(AbstractDataset):
    def __init__(
        self,
        data_store: Graph,
        pre_filter: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self._data_store = data_store
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Graph, List[Graph]]:
        if isinstance(idx, int):
            return self._data_store.get_example(idx)
        return [self._data_store.get_example(i) for i in idx]

    def __len__(self):
        return len(self._data_store)

    def sample(self, n_samples: int, replace: bool = False) -> "GraphDataset":
        """Sample n_samples from the dataset, preserving the original class type.

        Args:
            n_samples: Number of samples to draw
            replace: Whether to sample with replacement

        Returns:
            A new dataset of the same class containing the sampled graphs
        """
        idx_to_sample = np.random.choice(len(self), n_samples, replace=replace)
        data_list = self[idx_to_sample]
        pyg_batch = Batch.from_data_list(data_list)

        # Create a new dataset of the same class with the sampled data
        sampled_dataset = type(self)(
            data_store=Graph.from_pyg_batch(pyg_batch),
            pre_filter=self.pre_filter,
            pre_transform=self.pre_transform
        )

        return sampled_dataset


class OnlineGraphDataset(GraphDataset):
    def __init__(
        self,
        split: str,
        memmap: bool = False,
        data_store: Optional[Graph] = None,
        pre_filter: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        if data_store is None and split is not None:
            try:
                data_store = load_from_cache(self.identifier, split, mmap=memmap)
            except FileNotFoundError:
                download_to_cache(self.url_for_split(split), self.identifier, split)
                data_store = load_from_cache(self.identifier, split, mmap=memmap)
        super().__init__(data_store, pre_filter=pre_filter, pre_transform=pre_transform)

    @abstractmethod
    def url_for_split(self, split: str): ...
