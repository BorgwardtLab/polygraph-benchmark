import pytest

from polygraph.datasets import (
    MOSES,
    QM9,
    DobsonDoigGraphDataset,
    EgoGraphDataset,
    Guacamol,
    LobsterGraphDataset,
    PlanarGraphDataset,
    PointCloudGraphDataset,
    ProceduralLobsterGraphDataset,
    ProceduralPlanarGraphDataset,
    ProceduralSBMGraphDataset,
    SBMGraphDataset,
    SmallEgoGraphDataset,
)
from polygraph.datasets.base.dataset import ProceduralGraphDataset

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
    ProceduralLobsterGraphDataset,
    ProceduralPlanarGraphDataset,
    ProceduralSBMGraphDataset,
    PointCloudGraphDataset,
]


@pytest.fixture(params=ALL_DATASETS)
def ds_instance(request):
    ds_cls = request.param
    if issubclass(ds_cls, ProceduralGraphDataset):
        ds = ds_cls(split="train", num_graphs=10)
    else:
        ds = ds_cls(split="train")
    return ds


def test_dataset_len(ds_instance):
    assert len(ds_instance) > 0


def test_dataset_sample_graph_size(ds_instance):
    sizes = ds_instance.sample_graph_size(n_samples=5)
    assert len(sizes) == 5


def test_dataset_sample(ds_instance):
    sample = ds_instance.sample(n_samples=5)
    assert len(sample) == 5


def test_dataset_statistics(ds_instance):
    ds_instance.summary()
