# -*- coding: utf-8 -*-
"""Dataset classes for handling graph data.

This module implements base classes for working with graph datasets. It provides abstractions
for loading, caching and accessing collections of graphs in various formats.
"""

import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Optional, Union

import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from polygrapher.datasets.base.caching import (
    CacheLock,
    download_to_cache,
    load_from_cache,
    write_to_cache,
)
from polygrapher.datasets.base.graph import Graph


class AbstractDataset(ABC):
    """Abstract base class defining the dataset interface.
    
    This class defines the core functionality that all graph datasets must implement.
    It provides methods for accessing graphs and converting between formats.
    """

    @property
    def identifier(self) -> str:
        """Returns a unique identifier for dataset classes inheriting from this class.

        This identifier is used for caching purposes.
        
        Returns:
            String identifier combining the module path and class name
        """
        return f"{self.__module__}.{self.__class__.__qualname__}"

    def to_nx(self) -> "NetworkXView":
        """Creates a [`NetworkXView`][polygrapher.datasets.base.dataset.NetworkXView] view of this dataset that returns NetworkX graphs.
        
        Returns:
            NetworkX view wrapper around this dataset
        """
        return NetworkXView(self)

    @staticmethod
    @abstractmethod
    def is_valid(graph: nx.Graph) -> bool:
        """Checks if a graph is structurally valid in the context of this dataset.
        
        This method is optional and can be used in [`VUN`][polygrapher.metrics.base.vun.VUN] metrics.

        Args:
            graph: NetworkX graph to validate
            
        Returns:
            True if the graph is valid for this dataset, False otherwise
        """
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Data:
        """Gets a graph from the dataset by index.
        
        Args:
            idx: Index of the graph to retrieve
            
        Returns:
            Graph as a PyTorch Geometric Data object
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Gets the total number of graphs in the dataset.
        
        Returns:
            Number of graphs
        """
        ...


class NetworkXView:
    """View of a dataset that provides graphs in NetworkX format.
    
    This class wraps a dataset to provide access to graphs as NetworkX objects
    rather than PyTorch Geometric Data objects.
    
    Args:
        base_dataset: The dataset to wrap
    """

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
    """Basic dataset using a graph store.
    
    This class provides functionality for accessing and sampling from a collection
    of graphs stored in memory or on disk via a [`Graph`][polygrapher.datasets.base.graph.Graph] object.
    
    Args:
        data_store: Graph object containing the dataset
    """

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

    def sample_graph_size(self, n_samples: Optional[int] = None) -> List[int]:
        samples = []
        for _ in range(n_samples if n_samples is not None else 1):
            idx = np.random.randint(len(self))
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
    ) -> list[nx.Graph]:
        idx_to_sample = np.random.choice(len(self), n_samples, replace=replace)
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
    """Dataset that downloads and caches graph data.
    
    This class handles downloading graph data from a URL and caching it locally.
    Subclasses must implement methods to specify the data source.
    
    Args:
        split: Dataset split to load (e.g. 'train', 'test')
        memmap: Whether to memory-map the cached data. Useful for large datasets that do not fit into memory.
    """

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
    def url_for_split(self, split: str) -> str:
        """Gets the URL to download data for a specific split.
        
        Args:
            split: Dataset split (e.g. 'train', 'test')
            
        Returns:
            URL where the data can be downloaded
        """
        ...

    @abstractmethod
    def hash_for_split(self, split: str) -> str:
        """Gets the expected hash for a specific split's data.
        
        This hash is used to validate downloaded data.
        
        Args:
            split: Dataset split (e.g. 'train', 'test')
            
        Returns:
            Hash string for validating the split's data
        """
        ...


class ProceduralGraphDataset(GraphDataset):
    """Dataset that generates graphs procedurally.
    
    This class handles caching of procedurally generated graph data.
    Subclasses must implement the graph generation logic.
    
    Args:
        split: Dataset split to generate
        config_hash: Hash identifying the generation configuration
        memmap: Whether to memory-map the cached data
    """

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
    def generate_data(self) -> Graph:
        """Generates the graph data for this dataset.
        
        Returns:
            Generated graph data store
        """
        ...
