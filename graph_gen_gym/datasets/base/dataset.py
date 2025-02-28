# -*- coding: utf-8 -*-
"""dataset.py
Implementation of datasets.
"""

import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Optional, Union

import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from graph_gen_gym.datasets.base.caching import (
    CacheLock,
    download_to_cache,
    load_from_cache,
    write_to_cache,
)
from graph_gen_gym.datasets.base.graph import Graph


class AbstractDataset(ABC):
    @property
    def identifier(self) -> str:
        return f"{self.__module__}.{self.__class__.__qualname__}"

    def to_nx(self) -> "NetworkXView":
        return NetworkXView(self)

    @staticmethod
    @abstractmethod
    def is_valid(graph: nx.Graph) -> bool: ...

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
    ):
        super().__init__()
        self._data_store = data_store

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Graph, List[Graph]]:
        if isinstance(idx, int):
            return self._data_store.get_example(idx)
        return [self._data_store.get_example(i) for i in idx]

    def __len__(self):
        return len(self._data_store)

    def sample_graph_size(
        self, n_samples: Optional[int] = None, seed: int = 42
    ) -> List[int]:
        samples = []
        rng = np.random.RandomState(seed)
        for _ in range(n_samples if n_samples is not None else 1):
            idx = rng.randint(len(self))
            node_left, node_right = self._data_store.indexing_info.node_slices[
                idx
            ].tolist()
            samples.append(node_right - node_left)

        return samples if n_samples is not None else samples[0]

    def sample(
        self,
        n_samples: int,
        replace: bool = False,
        as_nx: bool = True,
        seed: int = 42,
    ) -> list[nx.Graph]:
        rng = np.random.RandomState(seed)
        idx_to_sample = rng.choice(len(self), n_samples, replace=replace)
        data_list = self[idx_to_sample]
        if as_nx:
            to_nx = partial(
                to_networkx,
                node_attrs=list(self._data_store.node_attr.keys()),
                edge_attrs=list(self._data_store.edge_attr.keys()),
                graph_attrs=list(self._data_store.graph_attr.keys()),
                to_undirected=True,
            )
            if isinstance(data_list, list):
                return [to_nx(g) for g in data_list]
            return to_nx(data_list)
        return data_list


class OnlineGraphDataset(GraphDataset):
    def __init__(
        self,
        split: str,
        memmap: bool = False,
    ):
        with CacheLock(self.identifier):
            try:
                data_store = load_from_cache(
                    self.identifier,
                    split,
                    mmap=memmap,
                    data_hash=self.hash_for_split(split),
                )
            except FileNotFoundError:
                download_to_cache(self.url_for_split(split), self.identifier, split)
                data_store = load_from_cache(
                    self.identifier,
                    split,
                    mmap=memmap,
                    data_hash=self.hash_for_split(split),
                )
        self._split = split
        super().__init__(data_store)

    def sample_graph_size(self, n_samples: Optional[int] = None) -> List[int]:
        if self._split != "train":
            warnings.warn(f"Sampling from {self._split} set, not training set.")
        return super().sample_graph_size(n_samples)

    @abstractmethod
    def url_for_split(self, split: str): ...

    @abstractmethod
    def hash_for_split(self, split: str) -> str: ...


class ProceduralGraphDataset(GraphDataset):
    def __init__(self, split: str, config_hash: str, memmap: bool = False):
        self._identifier = config_hash
        with CacheLock(self.identifier):
            try:
                data_store = load_from_cache(self.identifier, split, mmap=memmap)
            except FileNotFoundError:
                write_to_cache(self.identifier, split, self.generate_data())
                data_store = load_from_cache(self.identifier, split, mmap=memmap)
        super().__init__(data_store)

    @property
    def identifier(self) -> str:
        return f"{self.__module__}.{self.__class__.__qualname__}.{self._identifier}"

    @abstractmethod
    def generate_data(self) -> Graph: ...
