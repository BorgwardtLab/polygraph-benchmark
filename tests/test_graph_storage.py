import pytest
import torch
from torch_geometric.data import Batch, Data
import numpy as np
import networkx as nx

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


@pytest.mark.parametrize("use_attrs", [True, False])
def test_from_nx_graphs(use_attrs):
    g1 = nx.Graph()
    g1.add_node(0, feat=np.ones(3), feat2=-1 * np.ones(3))
    g1.add_node(1, feat=2 * np.ones(3), feat2=-2 * np.ones(3))
    g1.add_edge(0, 1, e=3 * np.ones(2))
    g1.graph["g"] = 7

    g2 = nx.Graph()
    g2.add_node(0, feat=4 * np.ones(3), feat2=-4 * np.ones(3))
    g2.add_node(1, feat=5 * np.ones(3), feat2=-5 * np.ones(3))
    g2.add_node(2, feat=9 * np.ones(3), feat2=-9 * np.ones(3))
    g2.add_edge(0, 1, e=6 * np.ones(2))
    g2.graph["g"] = 8

    if not use_attrs:
        gs = GraphStorage.from_nx_graphs([g1, g2])
        assert len(gs) == 2
        ex0 = gs.get_example(0)
        ex1 = gs.get_example(1)
        assert ex0.edge_index.shape == (2, 2)
        assert ex1.edge_index.shape == (2, 2)
    else:
        gs = GraphStorage.from_nx_graphs(
            [g1, g2],
            node_attrs=["feat", "feat2"],
            edge_attrs=["e"],
            graph_attrs=["g"],
        )
        assert len(gs) == 2

        ex0 = gs.get_example(0)
        ex1 = gs.get_example(1)

        assert ex0.g.item() == 7 and ex1.g.item() == 8

        assert ex0.edge_index.shape == (2, 2)
        assert ex1.edge_index.shape == (2, 2)

        # Remove backward edges for assertions
        ex0e = ex0.e[ex0.edge_index[0] < ex0.edge_index[1]]
        ex1e = ex1.e[ex1.edge_index[0] < ex1.edge_index[1]]

        assert tuple(ex0.feat.shape) == (2, 3)
        assert tuple(ex0.feat2.shape) == (2, 3)
        assert tuple(ex0e.shape) == (1, 2)
        assert tuple(ex1.feat2.shape) == (3, 3)
        assert tuple(ex1e.shape) == (1, 2)
        assert gs.num_graphs == 2
        assert (ex0.feat == torch.tensor([[1, 1, 1], [2, 2, 2]])).all()
        assert (ex0.feat2 == torch.tensor([[-1, -1, -1], [-2, -2, -2]])).all()
        assert (ex0e == torch.tensor([[3, 3]])).all()
        assert (
            ex1.feat == torch.tensor([[4, 4, 4], [5, 5, 5], [9, 9, 9]])
        ).all()
        assert (
            ex1.feat2
            == torch.tensor([[-4, -4, -4], [-5, -5, -5], [-9, -9, -9]])
        ).all()
        assert (ex1e == torch.tensor([[6, 6]])).all()
