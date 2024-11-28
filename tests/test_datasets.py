import networkx as nx
import pytest
import torch
import torch_geometric as pyg

from graph_gen_gym.datasets.spectre import PlanarGraphDataset, SBMGraphDataset


def test_planar_dataset_loading():
    # Test different splits
    for split in ["train", "val", "test"]:
        ds = PlanarGraphDataset(split)
        assert len(ds) > 0, f"Planar {split} dataset should not be empty"

        # Test PyG format
        g = ds[0]
        assert isinstance(g, pyg.data.Data), "Dataset should return PyG graphs"
        assert g.edge_index is not None, "Graph should have edges"

        # Test NetworkX conversion
        nx_ds = ds.to_nx()
        assert len(nx_ds) == len(ds), "NetworkX conversion should preserve dataset size"
        nx_g = nx_ds[0]
        assert isinstance(nx_g, nx.Graph), "to_nx should return NetworkX graphs"


def test_sbm_dataset_loading():
    # Test different splits
    for split in ["train", "val", "test"]:
        ds = SBMGraphDataset(split)
        assert len(ds) > 0, f"SBM {split} dataset should not be empty"

        # Test PyG format
        g = ds[0]
        assert isinstance(g, pyg.data.Data), "Dataset should return PyG graphs"
        assert g.edge_index is not None, "Graph should have edges"

        # Test NetworkX conversion
        nx_ds = ds.to_nx()
        assert len(nx_ds) == len(ds), "NetworkX conversion should preserve dataset size"
        nx_g = nx_ds[0]
        assert isinstance(nx_g, nx.Graph), "to_nx should return NetworkX graphs"


def test_dataset_iteration():
    # Test iteration functionality
    ds_planar = PlanarGraphDataset("train")
    ds_sbm = SBMGraphDataset("train")

    # Test PyG iteration
    for g in ds_planar:
        assert isinstance(g, pyg.data.Data)
        break

    for g in ds_sbm:
        assert isinstance(g, pyg.data.Data)
        break

    # Test NetworkX iteration
    for g in ds_planar.to_nx():
        assert isinstance(g, nx.Graph)
        break

    for g in ds_sbm.to_nx():
        assert isinstance(g, nx.Graph)
        break


def test_graph_properties():
    ds_planar = PlanarGraphDataset("train")
    ds_sbm = SBMGraphDataset("train")

    # Test basic graph properties
    for ds in [ds_planar, ds_sbm]:
        g = ds[0]
        assert g.num_nodes > 0, "Graphs should have nodes"
        assert g.num_edges > 0, "Graphs should have edges"

        nx_g = ds.to_nx()[0]
        assert ds.is_valid(nx_g), "Graphs should be sampled from SBM"


def test_invalid_inputs():
    # Test invalid split name
    with pytest.raises(KeyError):
        PlanarGraphDataset("invalid_split")

    with pytest.raises(KeyError):
        SBMGraphDataset("invalid_split")


def test_dataset_consistency():
    # Test if multiple loads give same data
    ds1 = PlanarGraphDataset("train")
    ds2 = PlanarGraphDataset("train")

    g1 = ds1[0]
    g2 = ds2[0]

    assert torch.equal(
        g1.edge_index, g2.edge_index
    ), "Multiple loads should give consistent data"
