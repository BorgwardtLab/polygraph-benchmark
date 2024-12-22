import networkx as nx
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from tqdm.rich import tqdm

from graph_gen_gym.datasets import (
    QM9,
    DobsonDoigGraphDataset,
    EgoGraphDataset,
    LobsterGraphDataset,
    PlanarGraphDataset,
    SBMGraphDataset,
    SmallEgoGraphDataset,
)
from graph_gen_gym.datasets.base import AbstractDataset
from graph_gen_gym.metrics.base import VUN

ALL_DATASETS = [
    PlanarGraphDataset,
    SBMGraphDataset,
    LobsterGraphDataset,
    SmallEgoGraphDataset,
    EgoGraphDataset,
    DobsonDoigGraphDataset,
    QM9,
]

VALIDATABLE_DATASETS = [PlanarGraphDataset, LobsterGraphDataset, QM9]

REPROCESSABLE_DATASETS = [QM9]


@pytest.mark.parametrize(
    "ds_cls",
    ALL_DATASETS,
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


@pytest.mark.skip
@pytest.mark.parametrize("ds_cls", VALIDATABLE_DATASETS)
def test_graph_properties_slow(ds_cls):
    for split in ["train", "val", "test"]:
        ds = ds_cls(split)
        assert hasattr(ds, "is_valid")
        assert all(g.number_of_nodes() > 0 for g in ds.to_nx())
        assert all(g.number_of_edges() > 0 for g in ds.to_nx())
        assert all(
            ds.is_valid(g)
            for g in tqdm(ds.to_nx(), desc=f"Validating {ds_cls.__name__} {split}")
        )


@pytest.mark.parametrize("ds_cls", VALIDATABLE_DATASETS)
def test_graph_properties_fast(ds_cls, sample_size):
    for split in ["train", "val", "test"]:
        ds = ds_cls(split)
        assert hasattr(ds_cls, "is_valid")
        sampled_graphs = ds.sample(sample_size)
        valid = []
        for g in tqdm(sampled_graphs, desc=f"Validating {ds_cls.__name__}"):
            valid.append(ds_cls.is_valid(g))
        assert all(valid)


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
    ALL_DATASETS,
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


# Ego datasets have non-unique graphs, which is apparently okay (?)
@pytest.mark.parametrize(
    "ds_cls",
    [
        PlanarGraphDataset,
        SBMGraphDataset,
        LobsterGraphDataset,
        DobsonDoigGraphDataset,
    ],
)
def test_split_disjointness(ds_cls):
    prev_splits = []

    for split in ["train", "val", "test"]:
        graphs = list(ds_cls(split).to_nx())
        vun = VUN(prev_splits, validity_fn=None)
        result = vun.compute(graphs)
        assert result["unique"].mle == 1
        assert result["novel"].mle == 1
        prev_splits.extend(graphs)


@pytest.mark.skip
@pytest.mark.parametrize("ds_cls", REPROCESSABLE_DATASETS)
def test_precomputed_false(ds_cls):
    # TODO: add attribute dimension checks
    for split in ["train", "val", "test"]:
        ds = ds_cls(split, use_precomputed=False)
        assert len(ds) > 0
        assert len(ds.to_nx()) > 0
