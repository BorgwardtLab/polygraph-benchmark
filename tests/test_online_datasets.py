import os
import warnings

import networkx as nx
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from tqdm.rich import tqdm

from polygraph.datasets import (
    URLGraphDataset,
    MOSES,
    QM9,
    DobsonDoigGraphDataset,
    EgoGraphDataset,
    Guacamol,
    LobsterGraphDataset,
    PlanarGraphDataset,
    SBMGraphDataset,
    SmallEgoGraphDataset,
)
from polygraph.datasets.base import AbstractDataset
from polygraph.datasets.base.caching import clear_cache, identifier_to_path
from polygraph.metrics import VUN

ALL_DATASETS = [
    QM9,
    MOSES,
    Guacamol,
    SmallEgoGraphDataset,
    EgoGraphDataset,
    SBMGraphDataset,
    PlanarGraphDataset,
    LobsterGraphDataset,
    DobsonDoigGraphDataset,
]

SYNTHETIC_DATASETS = [
    SBMGraphDataset,
    PlanarGraphDataset,
    LobsterGraphDataset,
]


@pytest.mark.parametrize(
    "ds_cls",
    [PlanarGraphDataset, SBMGraphDataset],
)
def test_sample_graph_size(ds_cls):
    ds = ds_cls("train")
    with warnings.catch_warnings():
        samples = ds.sample_graph_size(n_samples=100)
        assert isinstance(samples, list)
        assert isinstance(samples[0], int)
        assert len(samples) == 100
        assert np.all(np.array(samples) > 0)

        single_sample = ds.sample_graph_size()
        assert isinstance(single_sample, int)
        assert single_sample > 0

    ds_val = ds_cls("val")
    with pytest.warns(UserWarning):
        # This should warn because usually we want to sample from the training set
        _ = ds_val.sample_graph_size()


@pytest.mark.parametrize(
    "ds_cls",
    ALL_DATASETS,
)
def test_cache(ds_cls):
    ds = ds_cls("train")
    assert os.path.exists(identifier_to_path(ds.identifier))
    clear_cache(ds.identifier)
    assert not os.path.exists(identifier_to_path(ds.identifier))
    ds = ds_cls("train")
    assert os.path.exists(identifier_to_path(ds.identifier))
    _ = ds[0]
    assert isinstance(ds[0], Data)
    _ = ds_cls("train")


@pytest.mark.parametrize(
    "ds_cls",
    ALL_DATASETS,
)
def test_loading(ds_cls, sample_size):
    for split in ["train", "val", "test"]:
        ds = ds_cls(split)
        assert isinstance(ds, AbstractDataset), (
            "Should inherit from AbstractDataset"
        )
        assert len(ds) > 0, "Dataset should have at least one item"

        sample_size = min(sample_size, len(ds))

        pyg_graphs = ds.sample(sample_size, as_nx=False)
        assert len(pyg_graphs) == sample_size, (
            "Dataset should return same number of items"
        )
        assert all(isinstance(item, Data) for item in pyg_graphs), (
            "Dataset should return PyG graphs"
        )
        nx_graphs = ds.sample(sample_size, as_nx=True)
        assert len(nx_graphs) == sample_size, (
            "NetworkX conversion should preserve sample size"
        )
        assert all(isinstance(g, nx.Graph) for g in nx_graphs), (
            "to_nx should return NetworkX graphs"
        )


@pytest.mark.skip
@pytest.mark.parametrize("ds_cls", ALL_DATASETS)
def test_graph_properties_slow(ds_cls):
    for split in ["train", "val", "test"]:
        ds = ds_cls(split)
        assert hasattr(ds, "is_valid")
        assert all(g.number_of_nodes() > 0 for g in ds.to_nx())
        assert all(g.number_of_edges() > 0 for g in ds.to_nx())
        assert all(
            ds.is_valid(g)
            for g in tqdm(
                ds.to_nx(), desc=f"Validating {ds_cls.__name__} {split}"
            )
        )


@pytest.mark.parametrize("ds_cls", SYNTHETIC_DATASETS)
def test_graph_properties_fast(ds_cls, sample_size):
    for split in ["train", "val", "test"]:
        ds = ds_cls(split)
        assert hasattr(ds_cls, "is_valid")
        sampled_graphs = ds.sample(sample_size)
        valid = []
        for g in tqdm(sampled_graphs, desc=f"Validating {ds_cls.__name__}"):
            valid.append(ds.is_valid(g))
        if ds_cls.__name__ == "SBMGraphDataset":
            # Check that most graphs are valid, since the validity check is not
            # perfect
            assert np.sum(valid) / len(valid) >= 0.5
        else:
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

    assert torch.equal(g1.edge_index, g2.edge_index), (
        "Multiple loads should give consistent data"
    )


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
        assert result["unique"] == 1
        assert result["novel"] == 1
        prev_splits.extend(graphs)


@pytest.mark.parametrize("memmap", [True, False])
def test_url_dataset(memmap):
    ds = URLGraphDataset(
        "https://datashare.biochem.mpg.de/s/f3kXPP4LICWKbBx/download",
        memmap=memmap,
    )
    assert len(ds) == 128
    assert isinstance(ds[0], Data)
    assert ds.to_nx()[0].number_of_nodes() == 64
