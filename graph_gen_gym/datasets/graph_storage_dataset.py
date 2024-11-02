"""Implementation of `AbstractDataset` via efficiently and safely serializable sparse graph representation."""

from typing import Any, List, Optional

import torch
from pydantic import BaseModel, ConfigDict
from torch_geometric.data import Batch, Data
from torch_geometric.utils import cumsum

from graph_gen_gym.datasets.abstract_dataset import AbstractDataset


class IndexingInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_slices: torch.Tensor
    edge_slices: torch.Tensor
    inc: torch.Tensor


class GraphStorage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch: torch.Tensor
    edge_index: torch.Tensor
    num_graphs: int
    edge_attr: Optional[torch.Tensor] = None
    node_attr: Optional[torch.Tensor] = None
    graph_attr: Optional[torch.Tensor] = None
    indexing_info: Optional[IndexingInfo] = None
    description: Optional[str] = None
    extra_data: Any = None  # Any additional primitive ddata

    @staticmethod
    def _contiguously_increasing(tensor):
        unique_incr = torch.unique(tensor[1:] - tensor[:-1])
        if len(unique_incr) > 2:
            return False
        if len(unique_incr) == 1 and unique_incr[0] != 0:
            return False
        if len(unique_incr) == 2 and (unique_incr[0] != 0 or unique_incr[1] != 1):
            return False
        return True

    @staticmethod
    def from_pyg_batch(
        batch: Batch, compute_indexing_info: bool = False
    ) -> "GraphStorage":
        result = GraphStorage(
            batch=batch.batch,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr if hasattr(batch, "edge_attr") else None,
            node_attr=batch.x if hasattr(batch, "x") else None,
            graph_attr=batch.graph_attr if hasattr(batch, "graph_attr") else None,
            num_graphs=batch.num_graphs,
        )
        if compute_indexing_info:
            result.compute_indexing_info()
        return result

    def compute_indexing_info(self):
        if self.indexing_info is not None:
            raise RuntimeError("Indexing info already computed")

        # We first need to ensure that self.batch starts with 0 and is contiguously increasing
        if self.batch[0] != 0:
            raise ValueError("The first entry of InMemoryBatch.batch must be 0.")

        if not self._contiguously_increasing(self.batch):
            raise ValueError(
                "The tensor InMemoryBatch.batch must have entries that are non-decreasing and increase by at most 1 in each step."
            )

        num_nodes_per_graph = torch.bincount(self.batch)
        if len(num_nodes_per_graph) != self.num_graphs:
            raise ValueError(
                f"The value of InMemoryBatch.num_graphs ({self.num_graphs}) does not match the number of graphs indicated by InMemoryBatch.batch ({len(num_nodes_per_graph)})"
            )

        inc = cumsum(num_nodes_per_graph)
        node_slices = torch.stack([inc[:-1], inc[1:]], axis=1)
        assert len(node_slices) == self.num_graphs

        # Now compute edge slices
        edge_to_graph_a = torch.searchsorted(inc, self.edge_index[0], right=True) - 1
        edge_to_graph_b = torch.searchsorted(inc, self.edge_index[1], right=True) - 1

        if (edge_to_graph_a != edge_to_graph_b).any():
            raise RuntimeError

        edge_to_graph = edge_to_graph_a

        # To be able to slice by the graph, the edges have to be ordered
        if ((edge_to_graph_a[1:] - edge_to_graph_a[:-1]) < 0).any():
            raise ValueError

        num_edges_per_graph = torch.bincount(edge_to_graph)
        assert len(num_edges_per_graph) == self.num_graphs
        edge_inc = cumsum(num_edges_per_graph)
        edge_slices = torch.stack([edge_inc[:-1], edge_inc[1:]], axis=1)
        assert len(edge_slices) == self.num_graphs
        self.indexing_info = IndexingInfo(
            node_slices=node_slices, inc=inc, edge_slices=edge_slices
        )

    def get_example(self, idx):
        node_left, node_right = self.indexing_info.node_slices[idx].tolist()
        edge_left, edge_right = self.indexing_info.edge_slices[idx].tolist()
        node_offset = self.indexing_info.inc[idx]
        edge_index = self.edge_index[..., edge_left:edge_right] - node_offset
        edge_attr = (
            None if self.edge_attr is None else self.edge_attr[edge_left:edge_right]
        )
        node_attr = (
            None if self.node_attr is None else self.node_attr[node_left:node_right]
        )
        graph_attr = None if self.graph_attr is None else self.graph_attr[idx]
        return Data(
            edge_index=edge_index,
            node_attr=node_attr,
            edge_attr=edge_attr,
            graph_attr=graph_attr,
            num_nodes=node_right - node_left,
        )

    def __len__(self):
        return self.num_graphs


class ShardedGraphStorage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storages: List[GraphStorage]
    agg_graph_count: Optional[List] = None

    def get_example(self, idx):
        if self.agg_graph_count is None:
            self.agg_graph_count = cumsum(
                torch.Tensor([storage.num_graphs for storage in self.storages])
            ).tolist()

        # Binary search for idx


class GraphStorageDataset(AbstractDataset):
    def __init__(self, data_store: GraphStorage):
        super().__init__()
        self._data_store = data_store

    def __getitem__(self, idx):
        return self._data_store.get_example(idx)

    def __len__(self):
        return len(self._data_store)
