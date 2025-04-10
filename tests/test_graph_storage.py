import pytest
import torch
from torch_geometric.data import Batch, Data

from polygraph.datasets.base.graph_storage import GraphStorage


def test_graph_storage_initialization():
    batch = torch.tensor([0, 0, 1, 1, 2, 2])
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])
    num_graphs = 3
    gs = GraphStorage(batch=batch, edge_index=edge_index, num_graphs=num_graphs)
    assert gs.num_graphs == num_graphs

    with pytest.raises(ValueError):
        GraphStorage(
            batch=torch.tensor([1, 1, 2]), edge_index=edge_index, num_graphs=2
        )


def test_get_example():
    batch = torch.tensor([0, 0, 1, 1, 2, 2])
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])
    num_graphs = 3
    gs = GraphStorage(batch=batch, edge_index=edge_index, num_graphs=num_graphs)
    example = gs.get_example(0)
    assert isinstance(example, Data)


def test_from_pyg_batch():
    data_list = [
        Data(
            x=torch.tensor([[1], [2]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        )
        for _ in range(3)
    ]
    batch = Batch.from_data_list(data_list)
    gs = GraphStorage.from_pyg_batch(batch)
    assert gs.num_graphs == 3


def test_compute_indexing_info():
    batch = torch.tensor([0, 0, 1, 1, 2, 2])
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])
    num_graphs = 3
    gs = GraphStorage(batch=batch, edge_index=edge_index, num_graphs=num_graphs)
    assert gs.indexing_info is not None
