import networkx as nx
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graph_gen_gym.datasets import (
    DobsonDoigGraphDataset,
    EgoGraphDataset,
    LobsterGraphDataset,
    PlanarGraphDataset,
    SBMGraphDataset,
    SmallEgoGraphDataset,
)
from graph_gen_gym.datasets.dataset import AbstractDataset


@pytest.mark.parametrize(
    "ds_cls",
    [
        PlanarGraphDataset,
        SBMGraphDataset,
        LobsterGraphDataset,
        EgoGraphDataset,
        SmallEgoGraphDataset,
        DobsonDoigGraphDataset,
    ],
)
def test_loading(ds_cls):
    for split in ["train", "val", "test"]:
        ds = ds_cls(split)
        assert isinstance(ds, AbstractDataset), "Should inherit from AbstraactDataset"
        assert len(ds) > 0
        pyg_graphs = list(ds)
        assert len(pyg_graphs) == len(ds)
        assert all(
            isinstance(item, Data) for item in pyg_graphs
        ), "Dataset should return PyG graphs"
        nx_graphs = ds.to_nx()
        assert len(nx_graphs) == len(
            ds
        ), "NetworkX conversion should preserve dataset size"
        assert all(
            isinstance(g, nx.Graph) for g in nx_graphs
        ), "to_nx should return NetworkX graphs"


@pytest.mark.parametrize("ds_cls", [PlanarGraphDataset, LobsterGraphDataset])
def test_graph_properties(ds_cls):
    for split in ["train", "val", "test"]:
        ds = ds_cls(split)
        assert hasattr(ds, "is_valid")
        assert all(g.number_of_nodes() > 0 for g in ds.to_nx())
        assert all(g.number_of_edges() > 0 for g in ds.to_nx())
        assert all(ds.is_valid(g) for g in ds.to_nx())


@pytest.mark.skip
def test_graph_tool_validation():
    ds_sbm = SBMGraphDataset("train")
    validities = []
    for g in ds_sbm.to_nx():
        valid_gt = ds_sbm.is_valid(g)
        valid_alt = ds_sbm.is_valid_alt(g)
        validities.append([valid_gt, valid_alt])
    valid_gt = np.sum([val[0] for val in validities])
    valid_alt = np.sum([val[1] for val in validities])
    assert valid_gt / len(ds_sbm) > 0.8
    assert valid_alt / len(ds_sbm) > 0.8


def test_invalid_inputs():
    # Test invalid split name
    with pytest.raises(KeyError):
        PlanarGraphDataset("invalid_split")

    with pytest.raises(KeyError):
        SBMGraphDataset("invalid_split")


@pytest.mark.parametrize(
    "ds_cls",
    [
        PlanarGraphDataset,
        SBMGraphDataset,
        LobsterGraphDataset,
        EgoGraphDataset,
        SmallEgoGraphDataset,
        DobsonDoigGraphDataset,
    ],
)
def test_dataset_consistency(ds_cls):
    # Test if multiple loads give same data
    ds1 = ds_cls("train")
    ds2 = ds_cls("train")

    g1 = ds1[0]
    g2 = ds2[0]

    assert torch.equal(
        g1.edge_index, g2.edge_index
    ), "Multiple loads should give consistent data"
