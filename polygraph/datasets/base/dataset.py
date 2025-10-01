"""
We implement various base classes for working with graph datasets.
These provide abstractions for loading, caching and accessing collections of graphs.

Available classes:
    - [`AbstractDataset`][polygraph.datasets.base.dataset.AbstractDataset]: Abstract base class defining the dataset interface.
    - [`GraphDataset`][polygraph.datasets.base.dataset.GraphDataset]: A dataset that is initialized with a [`GraphStorage`][polygraph.datasets.base.graph_storage.GraphStorage] holding graphs.
    - [`URLGraphDataset`][polygraph.datasets.base.dataset.URLGraphDataset]: A dataset for downloading a collection of graphs as a [`GraphStorage`][polygraph.datasets.base.graph_storage.GraphStorage] object from a URL and caching it on disk.
    - [`SplitGraphDataset`][polygraph.datasets.base.dataset.SplitGraphDataset]: An abstract base class for downloading several dataset splits in the form of [`GraphStorage`][polygraph.datasets.base.graph_storage.GraphStorage] objects from a URL and caching them on disk. We provide several concrete implementations of this class, e.g. [`PlanarGraphDataset`][polygraph.datasets.planar.PlanarGraphDataset].
    - [`ProceduralGraphDataset`][polygraph.datasets.base.dataset.ProceduralGraphDataset]: Abstract base class for generating a [`GraphStorage`][polygraph.datasets.base.graph_storage.GraphStorage] procedurally.
"""

import hashlib
import re
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

    def is_valid(self, graph: nx.Graph) -> bool:
        """Checks if a graph is structurally valid in the context of this dataset.

        This method is optional and can be used in [`VUN`][polygraph.metrics.VUN] metrics.

        Args:
            graph: NetworkX graph to validate

        Returns:
            True if the graph is valid for this dataset, False otherwise
        """
        raise NotImplementedError(
            "This method must be implemented by subclasses."
        )

    @abstractmethod
    def __getitem__(
        self, idx: Union[int, List[int], slice]
    ) -> Union[Data, List[Data]]:
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

    @property
    @abstractmethod
    def node_attrs(self) -> List[str]: ...

    @property
    @abstractmethod
    def edge_attrs(self) -> List[str]: ...

    @property
    @abstractmethod
    def graph_attrs(self) -> List[str]: ...


class NetworkXView:
    """View of a dataset that provides graphs in NetworkX format.

    This class wraps a dataset to provide access to graphs as NetworkX objects
    rather than PyTorch Geometric Data objects.

    Args:
        base_dataset: The dataset to wrap
    """

    def __init__(self, base_dataset: AbstractDataset):
        self._base_dataset = base_dataset

    def __len__(self) -> int:
        """Gets the number of graphs in the dataset."""
        return len(self._base_dataset)

    def __getitem__(self, idx: int) -> nx.Graph:
        """Gets a graph from the dataset by index.

        Args:
            idx: Index of the graph to retrieve

        Returns:
            Graph as a NetworkX object
        """
        pyg_graph = self._base_dataset[idx]
        return to_networkx(
            pyg_graph,
            node_attrs=self._base_dataset.node_attrs,
            edge_attrs=self._base_dataset.edge_attrs,
            graph_attrs=self._base_dataset.graph_attrs,
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
        self, idx: Union[int, List[int], slice]
    ) -> Union[Data, List[Data]]:
        if isinstance(idx, slice):
            idx = list(range(*idx.indices(len(self))))

        if isinstance(idx, int):
            return self._data_store.get_example(idx)
        else:
            return [self._data_store.get_example(i) for i in idx]

    def __len__(self):
        return len(self._data_store)

    def dump_data(self, path: str) -> None:
        """Dumps the data store to a file.

        This file may be used to load the data store later on.
        In particular, a link to the file may be used in a [`URLGraphDataset`][polygraph.datasets.base.dataset.URLGraphDataset].

        Example:
            ```python
            from polygraph.datasets import GraphDataset, GraphStorage
            import networkx as nx

            ds = GraphDataset(GraphStorage.from_nx_graphs([nx.erdos_renyi_graph(64, 0.1) for _ in range(100)]))
            ds.dump_data("/tmp/my_dataset.pt")

            ds2 = GraphDataset.load_data("/tmp/my_dataset.pt", memmap=True)
            assert len(ds2) == 100
            assert ds2.to_nx()[0].number_of_nodes() == 64
            ```

        Args:
            path: Path to dump the data store to, preferably with a .pt extension.
        """
        torch.save(self._data_store.model_dump(), path)

    @staticmethod
    def load_data(path: str, memmap: bool = False) -> "GraphDataset":
        """Loads a data store from a file.

        Args:
            path: Path to load the data store from
            memmap: Whether to memory-map the cached data. Useful for large datasets that do not fit into memory.
        """
        return GraphDataset(
            GraphStorage(**torch.load(path, weights_only=True, mmap=memmap))
        )

    @property
    def node_attrs(self) -> List[str]:
        return list(self._data_store.node_attr.keys())

    @property
    def edge_attrs(self) -> List[str]:
        return list(self._data_store.edge_attr.keys())

    @property
    def graph_attrs(self) -> List[str]:
        return list(self._data_store.graph_attr.keys())

    def sample_graph_size(self, n_samples: Optional[int] = None) -> List[int]:
        """From the empirical distribution of this dataset, draw a random sample of graph sizes.

        This is useful for generative models that are conditioned on graph size, e.g. DiGress.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            List of graph sizes, drawn from the empirical distribution with replacement.
        """
        samples = []
        assert self._data_store.indexing_info is not None
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
    ) -> Union[List[nx.Graph], List[Data]]:
        idx_to_sample = np.random.choice(
            len(self), n_samples, replace=replace
        ).tolist()
        data_list = self[idx_to_sample]
        assert isinstance(data_list, list)
        if as_nx:
            to_nx = partial(
                to_networkx,
                node_attrs=list(self._data_store.node_attr.keys()),
                edge_attrs=list(self._data_store.edge_attr.keys()),
                graph_attrs=list(self._data_store.graph_attr.keys()),
                to_undirected=True,
            )
            return [to_nx(g) for g in data_list]
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
        assert self._data_store.indexing_info is not None
        min_edges = (
            (
                self._data_store.indexing_info.edge_slices[:, 1]
                - self._data_store.indexing_info.edge_slices[:, 0]
            )
            .min()
            .item()
        )
        if self.is_undirected:
            return int(min_edges // 2)
        return int(min_edges)

    @property
    def max_edges(self) -> int:
        """Maximum number of edges in a graph in the dataset."""
        assert self._data_store.indexing_info is not None
        max_edges = (
            (
                self._data_store.indexing_info.edge_slices[:, 1]
                - self._data_store.indexing_info.edge_slices[:, 0]
            )
            .max()
            .item()
        )
        if self.is_undirected:
            return int(max_edges // 2)
        return int(max_edges)

    @property
    def avg_edges(self) -> float:
        """Average number of edges in a graph in the dataset."""
        assert self._data_store.indexing_info is not None
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

    def summary(self, precision: int = 2) -> None:
        """Prints a summary of the dataset statistics.

        Args:
            precision: Number of decimal places to display
        """
        # Make sure we have a blank line before the table
        console = Console()
        console.print()

        if isinstance(self, SplitGraphDataset):
            assert hasattr(self, "_split")
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


class URLGraphDataset(GraphDataset):
    """Dataset that downloads a single split from a URL.

    This class handles downloading graph data from a URL and caching it locally.

    Args:
        url: URL to download the data from
        memmap: Whether to memory-map the cached data. Useful for large datasets that do not fit into memory.
    """

    def __init__(
        self,
        url: str,
        file_hash: Optional[str] = None,
        memmap: bool = False,
    ):
        self._url = url
        self._hash = file_hash
        with CacheLock(self.identifier):
            try:
                data_store = load_from_cache(
                    self.identifier,
                    mmap=memmap,
                    data_hash=file_hash,
                )
            except FileNotFoundError:
                download_to_cache(url, self.identifier)
                data_store = load_from_cache(
                    self.identifier,
                    mmap=memmap,
                    data_hash=file_hash,
                )
        super().__init__(data_store)

    @staticmethod
    def _url_to_folder_name(url: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', "_", url)

    @property
    def identifier(self) -> str:
        url_hash = hashlib.md5(self._url.encode()).hexdigest()
        return f"{self._url_to_folder_name(self._url)}.{url_hash}"


class SplitGraphDataset(GraphDataset):
    """Abstract base class for downloading and caching graph data with multiple splits.

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
    def hash_for_split(self, split: str) -> Optional[str]:
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
