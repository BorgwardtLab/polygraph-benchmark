import os
import pytest
import networkx as nx

from polygraph.metrics import VUN
from polygraph.datasets import (
    ProceduralLobsterGraphDataset,
    ProceduralPlanarGraphDataset,
    ProceduralSBMGraphDataset,
)
from polygraph.datasets.base.caching import clear_cache, identifier_to_path

ALL_PROCEDURAL_DATASETS = [
    ProceduralLobsterGraphDataset,
    ProceduralPlanarGraphDataset,
    ProceduralSBMGraphDataset,
]


@pytest.mark.parametrize("ds_cls", ALL_PROCEDURAL_DATASETS)
def test_split_disjointness(ds_cls):
    train = ds_cls("train", num_graphs=20)
    val = ds_cls("val", num_graphs=20)
    vun = VUN(
        train.to_nx(),
        validity_fn=train.is_valid,
    )
    metrics = vun.compute(val.to_nx())
    assert metrics["novel"] == 1.0
    assert metrics["unique"] == 1.0
    if ds_cls != ProceduralSBMGraphDataset:
        assert metrics["valid"] == 1.0
    else:
        assert metrics["valid"] >= 0.7


@pytest.mark.parametrize("ds_cls", ALL_PROCEDURAL_DATASETS)
def test_seed_disjointness(ds_cls):
    ds1 = ds_cls("train", num_graphs=20, seed=1)
    ds2 = ds_cls("train", num_graphs=20, seed=2)
    vun = VUN(ds1.to_nx())
    metrics = vun.compute(ds2.to_nx())
    assert metrics["novel"] == 1.0


@pytest.mark.parametrize("ds_cls", ALL_PROCEDURAL_DATASETS)
def test_reproducibility(ds_cls):
    ds = ds_cls("train", num_graphs=20, seed=1)
    graph1 = ds.to_nx()[0]
    cache_path = identifier_to_path(ds.identifier)
    assert os.path.exists(cache_path)
    clear_cache(ds.identifier)
    assert not os.path.exists(cache_path)
    ds = ds_cls("train", num_graphs=20, seed=1)
    assert os.path.exists(cache_path)
    graph2 = ds.to_nx()[0]
    assert nx.is_isomorphic(graph1, graph2)
