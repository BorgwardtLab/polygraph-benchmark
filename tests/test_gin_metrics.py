import numpy as np

from polygraph.metrics.frechet_distance import (
    GraphNeuralNetworkFrechetDistance,
)
from polygraph.metrics.rbf_mmd import (
    RBFGraphNeuralNetworkMMD2,
)
from polygraph.metrics.linear_mmd import (
    LinearGraphNeuralNetworkMMD2,
)
import networkx as nx
import pytest
from typing import List

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "ggm_implementation"))
import dgl  # noqa: E402
import torch  # noqa: E402
from ggm_implementation.evaluator import Evaluator  # noqa: E402


@pytest.fixture
def attributed_networkx_graphs():
    # Create 4 simple graphs with 2D node attributes and 3D edge attributes
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
    planar, sbm = datasets
    planar, sbm = list(planar.to_nx()), list(sbm.to_nx())
    planar_dgl = list(map(dgl.from_networkx, planar))
    sbm_dgl = list(map(dgl.from_networkx, sbm))

    torch.manual_seed(42)
    baseline_eval = Evaluator("gin", device="cpu")
    baseline_result = baseline_eval.evaluate_all(sbm_dgl, planar_dgl)
    fid, mmd_rbf, mmd_linear = (
        baseline_result["fid"],
        baseline_result["mmd_rbf"],
        baseline_result["mmd_linear"],
    )

    torch.manual_seed(42)
    our_rbf_mmd = RBFGraphNeuralNetworkMMD2(planar, seed=None).compute(sbm)
    assert np.isclose(our_rbf_mmd, mmd_rbf)

    torch.manual_seed(42)
    our_linear_mmd = LinearGraphNeuralNetworkMMD2(planar, seed=None).compute(
        sbm
    )
    assert np.isclose(our_linear_mmd, mmd_linear)

    torch.manual_seed(42)
    our_fid = GraphNeuralNetworkFrechetDistance(planar, seed=None).compute(sbm)
    assert np.isclose(our_fid, fid)


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
    g1, g2, g3, g4 = attributed_networkx_graphs

    ds1 = [g1, g2] * 10
    ds2 = [g3, g4] * 10
    if edge_attributed:
        # Need to first convert to directed graph for edge-attributed case
        def to_dgl(g: nx.Graph):
            return dgl.from_networkx(
                nx.DiGraph(g),
                node_attrs=["feat"] if node_attributed else None,
                edge_attrs=["edge_attr"],
            )
    else:

        def to_dgl(g: nx.Graph):
            return dgl.from_networkx(
                g, node_attrs=["feat"] if node_attributed else None
            )

    ds1_dgl = list(map(to_dgl, ds1))
    ds2_dgl = list(map(to_dgl, ds2))

    torch.manual_seed(42)
    baseline_eval = Evaluator(
        "gin",
        device="cpu",
        node_feat_loc="feat" if node_attributed else None,
        edge_feat_loc="edge_attr" if edge_attributed else None,
        input_dim=2 if node_attributed else 1,
        edge_feat_dim=3 if edge_attributed else 0,
    )
    baseline_result = baseline_eval.evaluate_all(ds2_dgl, ds1_dgl)
    fid, mmd_rbf, mmd_linear = (
        baseline_result["fid"],
        baseline_result["mmd_rbf"],
        baseline_result["mmd_linear"],
    )

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
    assert np.isclose(our_rbf_mmd, mmd_rbf)

    torch.manual_seed(42)
    our_linear_mmd = LinearGraphNeuralNetworkMMD2(
        ds1,
        node_feat_loc=node_feat_loc,
        node_feat_dim=node_feat_dim,
        edge_feat_loc=edge_feat_loc,
        edge_feat_dim=edge_feat_dim,
        seed=None,
    ).compute(ds2)
    assert np.isclose(our_linear_mmd, mmd_linear)

    torch.manual_seed(42)
    our_fid = GraphNeuralNetworkFrechetDistance(
        ds1,
        node_feat_loc=node_feat_loc,
        node_feat_dim=node_feat_dim,
        edge_feat_loc=edge_feat_loc,
        edge_feat_dim=edge_feat_dim,
        seed=None,
    ).compute(ds2)
    assert np.isclose(our_fid, fid)

    # Finally, check that the unattributed case is not the same as the attributed case
    if node_attributed or edge_attributed:
        torch.manual_seed(42)
        unattributed_mmd = RBFGraphNeuralNetworkMMD2(ds1, seed=None).compute(
            ds2
        )
        assert not np.isclose(unattributed_mmd, mmd_rbf)

        torch.manual_seed(42)
        unattributed_linear_mmd = LinearGraphNeuralNetworkMMD2(
            ds1, seed=None
        ).compute(ds2)
        assert not np.isclose(unattributed_linear_mmd, mmd_linear)

        torch.manual_seed(42)
        unattributed_fid = GraphNeuralNetworkFrechetDistance(
            ds1, seed=None
        ).compute(ds2)
        assert not np.isclose(unattributed_fid, fid)
