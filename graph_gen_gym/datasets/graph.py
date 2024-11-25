# -*- coding: utf-8 -*-
"""graph.py

Implementation of Graphs base classes.
"""
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch_geometric.data import Batch, Data
from torch_geometric.utils import cumsum


class IndexingInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_slices: torch.Tensor
    edge_slices: torch.Tensor
    inc: torch.Tensor


class Graph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch: torch.Tensor
    edge_index: torch.Tensor
    num_graphs: int
    edge_attr: Dict[str, torch.Tensor] = Field(default_factory=dict)
    node_attr: Dict[str, torch.Tensor] = Field(default_factory=dict)
    graph_attr: Dict[str, torch.Tensor] = Field(default_factory=dict)
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
        batch: Batch,
        compute_indexing_info: bool = False,
        edge_attrs: Optional[List[str]] = None,
        node_attrs: Optional[List[str]] = None,
        graph_attrs: Optional[List[str]] = None,
    ) -> "Graph":
        result = Graph(
            batch=batch.batch,
            edge_index=batch.edge_index,
            edge_attr={key: getattr(batch, key) for key in edge_attrs}
            if edge_attrs is not None
            else {},
            node_attr={key: getattr(batch, key) for key in node_attrs}
            if node_attrs is not None
            else {},
            num_graphs=batch.num_graphs,
            graph_attr={key: getattr(batch, key) for key in graph_attrs}
            if graph_attrs is not None
            else {},
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

    def get_example(self, idx: int) -> Data:
        node_left, node_right = self.indexing_info.node_slices[idx].tolist()
        edge_left, edge_right = self.indexing_info.edge_slices[idx].tolist()
        node_offset = self.indexing_info.inc[idx]
        edge_index = self.edge_index[..., edge_left:edge_right] - node_offset
        edge_attrs = {
            key: val[edge_left:edge_right] for key, val in self.edge_attr.items()
        }
        node_attrs = {
            key: val[node_left:node_right] for key, val in self.node_attr.items()
        }
        graph_attrs = {key: val[idx] for key, val in self.graph_attr.items()}
        return Data(
            edge_index=edge_index,
            num_nodes=node_right - node_left,
            **node_attrs,
            **edge_attrs,
            **graph_attrs,
        )

    def __len__(self):
        return self.num_graphs


class ShardedGraph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storages: List[Graph]
    agg_graph_count: Optional[torch.Tensor] = None
    num_graphs: Optional[int] = None

    def model_post_init(self, __context: Any) -> None:
        self.agg_graph_count = cumsum(
            torch.Tensor([storage.num_graphs for storage in self.storages]).to(
                torch.int64
            )
        )
        self.num_graphs = sum(storage.num_graphs for storage in self.storages)

    def get_example(self, idx):
        if idx < 0 or idx >= self.num_graphs:
            raise IndexError

        # Binary search for idx
        shard_idx = torch.searchsorted(self.agg_graph_count, idx, right=True) - 1
        local_idx = idx - self.agg_graph_count[shard_idx]
        assert local_idx >= 0
        return self.storages[shard_idx].get_example(local_idx)

    @staticmethod
    def from_shard_files(shards: List[str]):
        all_shards = []
        for shard_path in shards:
            shard_data = torch.load(shard_path, weights_only=True, mmap=True)
            all_shards.append(Graph(**shard_data))
        return ShardedGraph(storages=all_shards)
