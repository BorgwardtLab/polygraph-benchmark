from typing import List

import networkx as nx
import pytest

from polygraph.datasets import PlanarGraphDataset
from polygraph.metrics import VUN


def create_test_graphs() -> List[nx.Graph]:
    """Create a set of test graphs for testing metrics.

    Returns:
        List[nx.Graph]: List containing [triangle, square, triangle] graphs
    """
    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle

    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # Square

    g3 = nx.Graph()
    g3.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle (duplicate of g1)

    return [g1, g2, g3]


def test_vun_scores() -> None:
    ref_graphs = create_test_graphs()[:2]  # Triangle and Square
    gen_graphs = (
        create_test_graphs()
    )  # Triangle, Square, and duplicate Triangle

    vun = VUN(ref_graphs, confidence_level=0.95)
    vun_scores = vun.compute(gen_graphs)

    assert vun_scores["unique"].mle == 2 / 3, (
        "Should have 2 unique graphs out of 3"
    )
    assert vun_scores["novel"].mle == 0.0, "No novel graphs expected"


def test_vun_empty_inputs() -> None:
    ref_graphs = create_test_graphs()

    with pytest.raises(ValueError):
        VUN(ref_graphs).compute([])

    with pytest.raises(ValueError):
        VUN([]).compute([])


def test_vun_without_uncertainty() -> None:
    ref_graphs = create_test_graphs()
    gen_graphs = create_test_graphs()

    vun1 = VUN(ref_graphs)
    vun2 = VUN(ref_graphs, confidence_level=0.95)
    vun_scores1 = vun1.compute(gen_graphs)
    vun_scores2 = vun2.compute(gen_graphs)

    assert len(vun_scores1) == len(vun_scores2)

    for key in vun_scores1.keys():
        assert vun_scores1[key] == vun_scores2[key].mle


def test_vun_with_real_dataset() -> None:
    ds = PlanarGraphDataset("train")
    ref_graphs = list(ds.to_nx())[:10]
    gen_graphs = list(ds.to_nx())[10:20]

    vun = VUN(ref_graphs, validity_fn=ds.is_valid, confidence_level=0.95)
    vun_scores = vun.compute(gen_graphs)

    assert vun_scores["unique"].mle == 1, "All graphs should be unique"
    assert vun_scores["novel"].mle == 1, "All graphs should be novel"
    assert vun_scores["valid"].mle == 1, "All graphs should be valid"

    vun_scores = vun.compute(ref_graphs)
    assert vun_scores["unique"].mle == 1, "All graphs should be unique"
    assert vun_scores["novel"].mle == 0, "No novel graphs expected"
    assert vun_scores["valid"].mle == 1, "All graphs should be valid"

    vun_scores = vun.compute([gen_graphs[0] for _ in range(10)])
    assert vun_scores["unique"].mle == 0.1, (
        "Only one of 10 graphs should be unique"
    )
    assert vun_scores["novel"].mle == 1.0, "All graphs should be novel"
    assert vun_scores["valid"].mle == 1, "All graphs should be valid"
