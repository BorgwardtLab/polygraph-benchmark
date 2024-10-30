from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import cumsum, to_networkx


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
