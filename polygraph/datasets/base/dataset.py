"""
We implement various base classes for working with graph datasets.
These provide abstractions for loading, caching and accessing collections of graphs.

Available classes:
    - [`AbstractDataset`][polygraph.datasets.base.dataset.AbstractDataset]: Abstract base class defining the dataset interface.
    - [`GraphDataset`][polygraph.datasets.base.dataset.GraphDataset]: A dataset that is initialized with a [`GraphStorage`][polygraph.datasets.base.graph_storage.GraphStorage] holding graphs.
    - [`OnlineGraphDataset`][polygraph.datasets.base.dataset.OnlineGraphDataset]: Abstract base class for downloading a [`GraphStorage`][polygraph.datasets.base.graph_storage.GraphStorage] from a URL and caching it on disk.
    - [`ProceduralGraphDataset`][polygraph.datasets.base.dataset.ProceduralGraphDataset]: Abstract base class for generating a [`GraphStorage`][polygraph.datasets.base.graph_storage.GraphStorage] procedurally.
"""

import hashlib
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Optional, Union

import networkx as nx
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_networkx

from polygraph.datasets.base.caching import (
    CacheLock,
    download_to_cache,
    load_from_cache,
    write_to_cache,
)
from polygraph.datasets.base.graph_storage import GraphStorage


class AbstractDataset(ABC):
    """Abstract base class defining the dataset interface.

    This class defines the core functionality that all graph datasets must implement.
    It provides methods for accessing graphs and converting between formats.
    """

    def to_nx(self) -> "NetworkXView":
        """Creates a [`NetworkXView`][polygraph.datasets.base.dataset.NetworkXView] view of this dataset that returns NetworkX graphs.

        Returns:
            NetworkX view wrapper around this dataset
        """
        return NetworkXView(self)

    @staticmethod
    @abstractmethod
    def is_valid(graph: nx.Graph) -> bool:
        """Checks if a graph is structurally valid in the context of this dataset.

        This method is optional and can be used in [`VUN`][polygraph.metrics.base.vun.VUN] metrics.

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
    """Basic dataset using a [`GraphStorage`][polygraph.datasets.base.graph_storage.GraphStorage] object for holding graphs.

    This class provides functionality for accessing and sampling from a collection
    of graphs stored in memory or on disk via a [`GraphStorage`][polygraph.datasets.base.graph_storage.GraphStorage] object.

    Args:
        data_store: GraphStorage object containing the dataset
    """

    def __init__(
        self,
        data_store: GraphStorage,
    ):
        super().__init__()
        self._data_store = data_store

    def __getitem__(
        self, idx: Union[int, List[int]]
    ) -> Union[Data, List[Data]]:
        if isinstance(idx, int):
            return self._data_store.get_example(idx)
        return [self._data_store.get_example(i) for i in idx]

    def __len__(self):
        return len(self._data_store)

    def sample_graph_size(self, n_samples: Optional[int] = None) -> List[int]:
        """From the empirical distribution of this dataset, draw a random sample of graph sizes.

        This is useful for generative models that are conditioned on graph size, e.g. DiGress.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            List of graph sizes, drawn from the empirical distribution with replacement.
        """
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

    @property
    def is_undirected(self) -> bool:
        """Whether the graphs in the dataset are undirected."""
        return is_undirected(self._data_store.edge_index)

    @property
    def min_nodes(self) -> int:
        """Minimum number of nodes in a graph in the dataset."""
        return (
            torch.unique(self._data_store.batch, return_counts=True)[1]
            .min()
            .item()
        )

    @property
    def max_nodes(self) -> int:
        """Maximum number of nodes in a graph in the dataset."""
        return (
            torch.unique(self._data_store.batch, return_counts=True)[1]
            .max()
            .item()
        )

    @property
    def avg_nodes(self) -> float:
        """Average number of nodes in a graph in the dataset."""
        return (
            torch.unique(self._data_store.batch, return_counts=True)[1]
            .float()
            .mean()
            .item()
        )

    @property
    def min_edges(self) -> int:
        """Minimum number of edges in a graph in the dataset."""
        min_edges = (
            (
                self._data_store.indexing_info.edge_slices[:, 1]
                - self._data_store.indexing_info.edge_slices[:, 0]
            )
            .min()
            .item()
        )
        if self.is_undirected:
            return min_edges // 2
        return min_edges

    @property
    def max_edges(self) -> int:
        """Maximum number of edges in a graph in the dataset."""
        max_edges = (
            (
                self._data_store.indexing_info.edge_slices[:, 1]
                - self._data_store.indexing_info.edge_slices[:, 0]
            )
            .max()
            .item()
        )
        if self.is_undirected:
            return max_edges // 2
        return max_edges

    @property
    def avg_edges(self) -> float:
        """Average number of edges in a graph in the dataset."""
        avg_edges = (
            (
                self._data_store.indexing_info.edge_slices[:, 1]
                - self._data_store.indexing_info.edge_slices[:, 0]
            )
            .float()
            .mean()
            .item()
        )
        if self.is_undirected:
            return avg_edges / 2
        return avg_edges

    @property
    def edge_node_ratio(self) -> float:
        """Average number of edges per node in the dataset."""
        return self.avg_edges / self.avg_nodes

    def summary(self, precision: int = 2):
        # Make sure we have a blank line before the table
        console = Console()
        console.print()

        if hasattr(self, "_split"):
            table = Table(
                title=f"Graph Dataset Statistics for {self.__class__.__name__} ({self._split} set)"
            )
        else:
            table = Table(
                title=f"Graph Dataset Statistics for {self.__class__.__name__}"
            )
        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="left", style="magenta", no_wrap=True)
        table.add_row("# of Graphs", str(len(self)))
        table.add_row("Min # of Nodes", str(self.min_nodes))
        table.add_row("Max # of Nodes", str(self.max_nodes))
        table.add_row("Avg # of Nodes", f"{self.avg_nodes:.{precision}f}")
        table.add_row("Min # of Edges", str(self.min_edges))
        table.add_row("Max # of Edges", str(self.max_edges))
        table.add_row("Avg # of Edges", f"{self.avg_edges:.{precision}f}")
        table.add_row(
            "Edge/Node Ratio", f"{self.edge_node_ratio:.{precision}f}"
        )

        console = Console()
        console.print(table)


class OnlineGraphDataset(GraphDataset):
    """Abstract base class for downloading and caching graph data.

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
        self._split = split
        with CacheLock(self.identifier):
            try:
                data_store = load_from_cache(
                    self.identifier,
                    split,
                    mmap=memmap,
                    data_hash=self.hash_for_split(split),
                )
            except FileNotFoundError:
                download_to_cache(
                    self.url_for_split(split), self.identifier, split
                )
                data_store = load_from_cache(
                    self.identifier,
                    split,
                    mmap=memmap,
                    data_hash=self.hash_for_split(split),
                )
        super().__init__(data_store)

    def sample_graph_size(self, n_samples: Optional[int] = None) -> List[int]:
        if self._split != "train":
            warnings.warn(f"Sampling from {self._split} set, not training set.")
        return super().sample_graph_size(n_samples)

    @property
    def identifier(self) -> str:
        """Identifier that incorporates the split."""
        url = self.url_for_split(self._split)
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"{self.__module__}.{self.__class__.__qualname__}.{self._split}.{url_hash}"

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

    def __init__(
        self,
        split: str,
        config_hash: str,
        memmap: bool = False,
        show_generation_progress: bool = False,
    ):
        self._identifier = config_hash
        self.show_generation_progress = show_generation_progress
        with CacheLock(self.identifier):
            try:
                data_store = load_from_cache(
                    self.identifier, split, mmap=memmap
                )
            except FileNotFoundError:
                write_to_cache(
                    self.identifier,
                    split,
                    self.generate_data(),
                )
                data_store = load_from_cache(
                    self.identifier, split, mmap=memmap
                )
        super().__init__(data_store)

    @property
    def identifier(self) -> str:
        return f"{self.__module__}.{self.__class__.__qualname__}.{self._identifier}"

    @abstractmethod
    def generate_data(self) -> GraphStorage:
        """Generates the graph data for this dataset.

        Returns:
            Generated graph data store
        """
        ...
