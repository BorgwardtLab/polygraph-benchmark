from typing import List

import networkx as nx
import numpy as np
import pytest

from graph_gen_gym.datasets.spectre import PlanarGraphDataset
from graph_gen_gym.metrics.frechet_distance import FrechetDistance
from graph_gen_gym.metrics.utils.graph_descriptors import OrbitCounts
from graph_gen_gym.metrics.vun import VUN


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
    gen_graphs = create_test_graphs()  # Triangle, Square, and duplicate Triangle

    vun = VUN(ref_graphs)
    vun_scores = vun.compute(gen_graphs)

    assert vun_scores["unique"].mle == 2 / 3, "Should have 2 unique graphs out of 3"
    assert vun_scores["novel"].mle == 0.0, "No novel graphs expected"


def test_vun_empty_inputs() -> None:
    ref_graphs = create_test_graphs()

    with pytest.raises(ValueError):
        VUN(ref_graphs).compute([])

    with pytest.raises(ValueError):
        VUN([]).compute([])


def test_vun_with_real_dataset() -> None:
    ds = PlanarGraphDataset("train")
    ref_graphs = list(ds.to_nx())[:10]
    gen_graphs = list(ds.to_nx())[10:20]

    vun = VUN(ref_graphs, validity_fn=ds.is_valid)
    vun_scores = vun.compute(gen_graphs)

    assert 0 <= vun_scores["unique"].mle <= 1, "Unique score should be between 0 and 1"
    assert 0 <= vun_scores["novel"].mle <= 1, "Novel score should be between 0 and 1"


def test_frechet_distance() -> None:
    ref_graphs = [nx.cycle_graph(3), nx.cycle_graph(4)]  # Triangle and square
    gen_graphs = [nx.cycle_graph(3), nx.cycle_graph(5)]  # Triangle and pentagon

    frechet_distance = FrechetDistance(ref_graphs, descriptor_fn=OrbitCounts())
    gen_distance = frechet_distance.compute(gen_graphs)

    assert gen_distance >= 0, "Frechet distance should be non-negative"
    assert isinstance(gen_distance, float), "Frechet distance should be a float"


def test_frechet_distance_identical() -> None:
    ref_graphs = [nx.cycle_graph(3), nx.cycle_graph(4)]
    gen_graphs = [nx.cycle_graph(3), nx.cycle_graph(4)]

    frechet_distance = FrechetDistance(ref_graphs, descriptor_fn=OrbitCounts())
    gen_distance = frechet_distance.compute(gen_graphs)

    assert np.isclose(
        gen_distance, 0.0, atol=1e-2
    ), "Frechet distance between identical distributions should be 0"


def test_frechet_distance_with_real_data() -> None:
    ds = PlanarGraphDataset("train")

    ref_graphs = list(ds.to_nx())[:50]
    gen_graphs = list(ds.to_nx())[50:100]

    frechet_distance = FrechetDistance(ref_graphs, descriptor_fn=OrbitCounts())
    gen_distance = frechet_distance.compute(gen_graphs)

    assert gen_distance >= 0, "Frechet distance should be non-negative"
    assert isinstance(gen_distance, float), "Frechet distance should be a float"
