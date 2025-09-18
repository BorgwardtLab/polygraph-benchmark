import networkx as nx
import numpy as np

from polygraph.datasets import PlanarGraphDataset
from polygraph.metrics.base import FrechetDistance
from polygraph.utils.descriptors import OrbitCounts


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

    assert np.isclose(gen_distance, 0.0, atol=1e-2), (
        "Frechet distance between identical distributions should be 0"
    )


def test_frechet_distance_with_real_data() -> None:
    ds = PlanarGraphDataset("train")

    ref_graphs = list(ds.to_nx())[:50]
    gen_graphs = list(ds.to_nx())[50:100]

    frechet_distance = FrechetDistance(ref_graphs, descriptor_fn=OrbitCounts())
    gen_distance = frechet_distance.compute(gen_graphs)

    assert gen_distance >= 0, "Frechet distance should be non-negative"
    assert isinstance(gen_distance, float), "Frechet distance should be a float"
