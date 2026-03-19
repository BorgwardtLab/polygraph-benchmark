from typing import List

import networkx as nx
import numpy as np
import pytest
import torch

from polygraph.metrics.frechet_distance import (
    GraphNeuralNetworkFrechetDistance,
)
from polygraph.metrics.linear_mmd import (
    LinearGraphNeuralNetworkMMD2,
)
from polygraph.metrics.rbf_mmd import (
    RBFGraphNeuralNetworkMMD2,
)

# Reference values precomputed with DGL 2.3.0 (PyTorch 2.3.1) using the
# ggm_implementation.evaluator.Evaluator with torch.manual_seed(42).
#
# DGL is incompatible with PyTorch>=2.4 so we cannot use it as a runtime
# dependency. The values below were produced once in an isolated environment
# and are checked in as frozen references.
#
# Comparisons use rtol=1e-3 because minor floating-point differences are
# expected across PyTorch versions (reference: 2.3.1, current: >=2.4).
#
# To regenerate, create a temporary pixi project and run:
#
#     pixi init /tmp/dgl_project --channel conda-forge
#     cd /tmp/dgl_project
#     pixi add python=3.10 "dgl=2.3.0" networkx pyyaml "torchdata<0.8"
#     pixi add --pypi torch_geometric pydantic rdkit joblib appdirs \
#         loguru pandas scikit-learn numba eden-kernel
#
#     PYTHONPATH="<repo>/tests:<repo>" KMP_DUPLICATE_LIB_OK=TRUE \
#     pixi run python -c "
#     import torch, dgl, networkx as nx, numpy as np
#     from ggm_implementation.evaluator import Evaluator
#     from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset
#
#     # Unattributed
#     planar = list(PlanarGraphDataset('train').to_nx())
#     sbm = list(SBMGraphDataset('train').to_nx())
#     torch.manual_seed(42)
#     ev = Evaluator('gin', device='cpu')
#     r = ev.evaluate_all(list(map(dgl.from_networkx, sbm)),
#                         list(map(dgl.from_networkx, planar)))
#     print(r)
#
#     # Attributed (4 combinations of node/edge features)
#     # See attributed_networkx_graphs fixture for graph definitions.
#     # ds1 = [g1, g2]*10, ds2 = [g3, g4]*10
#     for na in [True, False]:
#       for ea in [True, False]:
#         ...  # see test body for DGL conversion details
#         torch.manual_seed(42)
#         ev = Evaluator('gin', device='cpu',
#             node_feat_loc='feat' if na else None,
#             edge_feat_loc='edge_attr' if ea else None,
#             input_dim=2 if na else 1,
#             edge_feat_dim=3 if ea else 0)
#         r = ev.evaluate_all(ds2_dgl, ds1_dgl)
#         print(na, ea, r)
#     "

DGL_UNATTRIBUTED = {
    "fid": 2957766.3744324073,
    "mmd_rbf": 1.0580106228590012,
    "mmd_linear": 2132179.0,
}

DGL_ATTRIBUTED = {
    (True, True): {
        "fid": 48254.679086046475,
        "mmd_rbf": 1.4141081187990494,
        "mmd_linear": 36761.42,
    },
    (True, False): {
        "fid": 2151.094310505561,
        "mmd_rbf": 1.2381987534463406,
        "mmd_linear": 1836.352,
    },
    (False, True): {
        "fid": 145.31360315744712,
        "mmd_rbf": 1.0005450689932331,
        "mmd_linear": 105.7898,
    },
    (False, False): {
        "fid": 165.04879303953692,
        "mmd_rbf": 1.5,
        "mmd_linear": 126.114044,
    },
}


@pytest.fixture
def attributed_networkx_graphs():
    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (1, 2), (2, 3)])
    nx.set_node_attributes(
        g1,
        {
            0: np.array([0.1, 0.2]).astype(np.float32),
            1: np.array([0.3, 0.4]).astype(np.float32),
            2: np.array([0.5, 0.6]).astype(np.float32),
            3: np.array([0.7, 0.8]).astype(np.float32),
        },
        "feat",
    )
    nx.set_edge_attributes(
        g1,
        {
            (0, 1): np.array([0.1, 0.2, 0.3]).astype(np.float32),
            (1, 2): np.array([0.2, 0.3, 0.4]).astype(np.float32),
            (2, 3): np.array([0.3, 0.4, 0.5]).astype(np.float32),
        },
        "edge_attr",
    )

    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    nx.set_node_attributes(
        g2,
        {
            0: np.array([0.2, 0.3]).astype(np.float32),
            1: np.array([0.4, 0.5]).astype(np.float32),
            2: np.array([0.6, 0.7]).astype(np.float32),
            3: np.array([0.8, 0.9]).astype(np.float32),
        },
        "feat",
    )
    nx.set_edge_attributes(
        g2,
        {
            (0, 1): np.array([0.2, 0.3, 0.4]).astype(np.float32),
            (1, 2): np.array([0.3, 0.4, 0.5]).astype(np.float32),
            (2, 3): np.array([0.4, 0.5, 0.6]).astype(np.float32),
            (3, 0): np.array([0.5, 0.6, 0.7]).astype(np.float32),
        },
        "edge_attr",
    )

    g3 = nx.Graph()
    g3.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3)])
    nx.set_node_attributes(
        g3,
        {
            0: np.array([0.1, 0.3]).astype(np.float32),
            1: np.array([0.3, 0.5]).astype(np.float32),
            2: np.array([0.5, 0.7]).astype(np.float32),
            3: np.array([0.7, 0.9]).astype(np.float32),
        },
        "feat",
    )
    nx.set_edge_attributes(
        g3,
        {
            (0, 1): np.array([0.1, 0.3, 0.5]).astype(np.float32),
            (1, 2): np.array([0.3, 0.5, 0.7]).astype(np.float32),
            (0, 2): np.array([0.2, 0.4, 0.6]).astype(np.float32),
            (2, 3): np.array([0.4, 0.6, 0.8]).astype(np.float32),
        },
        "edge_attr",
    )

    g4 = nx.Graph()
    g4.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 1)])
    nx.set_node_attributes(
        g4,
        {
            0: np.array([0.2, 0.4]).astype(np.float32),
            1: np.array([0.4, 0.6]).astype(np.float32),
            2: np.array([0.6, 0.8]).astype(np.float32),
            3: np.array([0.8, 1.0]).astype(np.float32),
        },
        "feat",
    )
    nx.set_edge_attributes(
        g4,
        {
            (0, 1): np.array([0.2, 0.4, 0.6]).astype(np.float32),
            (1, 2): np.array([0.4, 0.6, 0.8]).astype(np.float32),
            (2, 3): np.array([0.6, 0.8, 1.0]).astype(np.float32),
            (3, 1): np.array([0.5, 0.7, 0.9]).astype(np.float32),
        },
        "edge_attr",
    )

    return [g1, g2, g3, g4]


def test_gin_metrics_unattributed(datasets):
    """Test GIN metrics against DGL 2.3.0 reference values.

    See DGL_UNATTRIBUTED above for how to regenerate.
    """
    planar, sbm = datasets
    planar, sbm = list(planar.to_nx()), list(sbm.to_nx())

    torch.manual_seed(42)
    our_rbf_mmd = RBFGraphNeuralNetworkMMD2(planar, seed=None).compute(sbm)
    assert np.isclose(our_rbf_mmd, DGL_UNATTRIBUTED["mmd_rbf"], rtol=1e-3)

    torch.manual_seed(42)
    our_linear_mmd = LinearGraphNeuralNetworkMMD2(planar, seed=None).compute(
        sbm
    )
    assert np.isclose(our_linear_mmd, DGL_UNATTRIBUTED["mmd_linear"], rtol=1e-3)

    torch.manual_seed(42)
    our_fid = GraphNeuralNetworkFrechetDistance(planar, seed=None).compute(sbm)
    assert np.isclose(our_fid, DGL_UNATTRIBUTED["fid"], rtol=1e-3)


@pytest.mark.parametrize(
    "metric_cls",
    [
        RBFGraphNeuralNetworkMMD2,
        LinearGraphNeuralNetworkMMD2,
        GraphNeuralNetworkFrechetDistance,
    ],
)
def test_gin_seeding(datasets, metric_cls):
    planar, sbm = datasets
    planar, sbm = list(planar.to_nx()), list(sbm.to_nx())

    metric_1 = metric_cls(planar, seed=42).compute(sbm)
    metric_2 = metric_cls(planar, seed=42).compute(sbm)
    metric_3 = metric_cls(planar, seed=43).compute(sbm)
    assert np.isclose(metric_1, metric_2)
    assert not np.isclose(metric_1, metric_3)


@pytest.mark.parametrize("node_attributed", [True, False])
@pytest.mark.parametrize("edge_attributed", [True, False])
def test_gin_metrics_attributed(
    attributed_networkx_graphs: List[nx.Graph],
    node_attributed: bool,
    edge_attributed: bool,
):
    """Test attributed GIN metrics against DGL 2.3.0 reference values.

    See DGL_ATTRIBUTED above for how to regenerate.
    """
    g1, g2, g3, g4 = attributed_networkx_graphs

    ds1 = [g1, g2] * 10
    ds2 = [g3, g4] * 10

    ref = DGL_ATTRIBUTED[(node_attributed, edge_attributed)]

    if node_attributed:
        node_feat_loc = ["feat"]
        node_feat_dim = 2
    else:
        node_feat_loc = None
        node_feat_dim = 1

    if edge_attributed:
        edge_feat_loc = ["edge_attr"]
        edge_feat_dim = 3
    else:
        edge_feat_loc = None
        edge_feat_dim = 0

    torch.manual_seed(42)
    our_rbf_mmd = RBFGraphNeuralNetworkMMD2(
        ds1,
        node_feat_loc=node_feat_loc,
        node_feat_dim=node_feat_dim,
        edge_feat_loc=edge_feat_loc,
        edge_feat_dim=edge_feat_dim,
        seed=None,
    ).compute(ds2)
    assert np.isclose(our_rbf_mmd, ref["mmd_rbf"], rtol=1e-3)

    torch.manual_seed(42)
    our_linear_mmd = LinearGraphNeuralNetworkMMD2(
        ds1,
        node_feat_loc=node_feat_loc,
        node_feat_dim=node_feat_dim,
        edge_feat_loc=edge_feat_loc,
        edge_feat_dim=edge_feat_dim,
        seed=None,
    ).compute(ds2)
    assert np.isclose(our_linear_mmd, ref["mmd_linear"], rtol=1e-3)

    torch.manual_seed(42)
    our_fid = GraphNeuralNetworkFrechetDistance(
        ds1,
        node_feat_loc=node_feat_loc,
        node_feat_dim=node_feat_dim,
        edge_feat_loc=edge_feat_loc,
        edge_feat_dim=edge_feat_dim,
        seed=None,
    ).compute(ds2)
    assert np.isclose(our_fid, ref["fid"], rtol=1e-3)

    if node_attributed or edge_attributed:
        torch.manual_seed(42)
        unattributed_mmd = RBFGraphNeuralNetworkMMD2(ds1, seed=None).compute(
            ds2
        )
        assert not np.isclose(unattributed_mmd, ref["mmd_rbf"])

        torch.manual_seed(42)
        unattributed_linear_mmd = LinearGraphNeuralNetworkMMD2(
            ds1, seed=None
        ).compute(ds2)
        assert not np.isclose(unattributed_linear_mmd, ref["mmd_linear"])

        torch.manual_seed(42)
        unattributed_fid = GraphNeuralNetworkFrechetDistance(
            ds1, seed=None
        ).compute(ds2)
        assert not np.isclose(unattributed_fid, ref["fid"])
