"""
We store graphs in [`GraphStorage`][polygrapher.datasets.base.graph_storage.GraphStorage] objects. These objects are safely serializable
and can be indexed efficiently to retrieve PyTorch Geometric `Data` objects.
"""

import os
from typing import Any, Dict, List, Optional, Union
from importlib.metadata import version

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch_geometric.data import Batch, Data


def _cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    r"""Returns the cumulative sum of elements of :obj:`x`.
    In contrast to :meth:`torch.cumsum`, prepends the output with zero.
    Taken from https://github.com/pyg-team/pytorch_geometric/blob/08697a7197504158e0b68a8e191e95233e5f8a32/torch_geometric/utils/functions.py#L5.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension to do the operation over.
            (default: :obj:`0`)

    Example:
        >>> x = torch.tensor([2, 4, 1])
        >>> cumsum(x)
        tensor([0, 2, 6, 7])

    """
    size = x.size()[:dim] + (x.size(dim) + 1, ) + x.size()[dim + 1:]
    out = x.new_empty(size)

    out.narrow(dim, 0, 1).zero_()
    torch.cumsum(x, dim=dim, out=out.narrow(dim, 1, x.size(dim)))

    return out


class IndexingInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_slices: torch.Tensor
    edge_slices: torch.Tensor
    inc: torch.Tensor


class GraphStorage(BaseModel):
    """Serializable collection of graphs.
    
    Attributes:
        description: Optional description of the collection of graphs.
        module_version: Version of the `polygrapher` package that created the `GraphStorage` object.
        extra_data: Any additional primitive metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch: torch.Tensor
    edge_index: torch.Tensor
    num_graphs: int
    edge_attr: Dict[str, torch.Tensor] = Field(default_factory=dict)
    node_attr: Dict[str, torch.Tensor] = Field(default_factory=dict)
    graph_attr: Dict[str, torch.Tensor] = Field(default_factory=dict)
    indexing_info: Optional[IndexingInfo] = None
    description: Optional[str] = None
    module_version: Optional[str] = None
    extra_data: Any = None  # Any additional primitive data

    def model_post_init(self, __context: Any) -> None:
        if self.indexing_info is None:
            self._compute_indexing_info()
        if self.module_version is None:
            self.module_version = version("polygrapher")
    
    def get_example(self, idx: int) -> Data:
        """Retrieve a single graph from the collection.
        
        Args:
            idx: Index of the graph to retrieve.

        Returns:
            PyTorch Geometric `Data` object containing the graph.
        """
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
        """Number of graphs in the collection."""
        return self.num_graphs

    @staticmethod
    def from_pyg_batch(
        batch: Batch,
        edge_attrs: Optional[List[str]] = None,
        node_attrs: Optional[List[str]] = None,
        graph_attrs: Optional[List[str]] = None,
    ) -> "GraphStorage":
        """Construct a `GraphStorage` object from a PyTorch Geometric `Batch` object.

        Args:
            batch: PyTorch Geometric `Batch` object.
            edge_attrs: List of edge-level attributes to include, must be present in the `Batch` object.
            node_attrs: List of node-level attributes to include, must be present in the `Batch` object.
            graph_attrs: List of graph-level attributes to include, must be present in the `Batch` object.
        """
        result = GraphStorage(
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
        return result
    
    @staticmethod
    def _contiguously_increasing(tensor: torch.Tensor) -> bool:
        unique_incr = torch.unique(tensor[1:] - tensor[:-1])
        if len(unique_incr) > 2:
            return False
        if len(unique_incr) == 1 and unique_incr[0] != 0:
            return False
        if len(unique_incr) == 2 and (unique_incr[0] != 0 or unique_incr[1] != 1):
            return False
        return True

    def _compute_indexing_info(self):
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

        inc = _cumsum(num_nodes_per_graph)
        node_slices = torch.stack([inc[:-1], inc[1:]], axis=1)
        assert len(node_slices) == self.num_graphs

        # Now compute edge slices
        edge_to_graph_a = torch.searchsorted(inc, self.edge_index[0], right=True) - 1
        edge_to_graph_b = torch.searchsorted(inc, self.edge_index[1], right=True) - 1

        if (edge_to_graph_a != edge_to_graph_b).any():
            raise RuntimeError("Edge to graph mapping is inconsistent")

        edge_to_graph = edge_to_graph_a

        # To be able to slice by the graph, the edges have to be ordered
        if ((edge_to_graph[1:] - edge_to_graph[:-1]) < 0).any():
            raise ValueError("Edge to graph mapping is not ordered")

        num_edges_per_graph = torch.bincount(edge_to_graph)
        assert len(num_edges_per_graph) == self.num_graphs
        edge_inc = _cumsum(num_edges_per_graph)
        edge_slices = torch.stack([edge_inc[:-1], edge_inc[1:]], axis=1)
        assert len(edge_slices) == self.num_graphs
        self.indexing_info = IndexingInfo(
            node_slices=node_slices, inc=inc, edge_slices=edge_slices
        )