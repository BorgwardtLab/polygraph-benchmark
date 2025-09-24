from typing import Iterable, Collection
import numpy as np
import networkx as nx
from loguru import logger
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset
from polygraph.metrics.base import PolyGraphScore, DescriptorMMD2
from polygraph.utils.kernels import LinearKernel
from polygraph.utils.descriptors import ClusteringHistogram
from sklearn.linear_model import LogisticRegression

logger.disable("polygraph")


def betweenness_descriptor(graphs: Iterable[nx.Graph]) -> np.ndarray:
    """A custom graph descriptor that computes betweenness centrality.

    This implements the polygraph.utils.descriptors.GraphDescriptor interface.
    """
    histograms = []
    for graph in graphs:
        btw_values = list(nx.betweenness_centrality(graph).values())
        histograms.append(
            np.histogram(btw_values, bins=100, range=(0.0, 1.0), density=True)[
                0
            ]
        )
    return np.stack(histograms, axis=0)


def calculate_custom_mmd(
    reference: Collection[nx.Graph], generated: Collection[nx.Graph]
):
    """
    Calculate a customized MMD between a reference dataset and a generated dataset.

    This MMD uses a linear kernel based on betweenness centrality histograms.
    It is estimated using the unbiased minimum variance estimator.
    """
    print("Calculating custom MMD...")
    mmd = DescriptorMMD2(
        reference, kernel=LinearKernel(betweenness_descriptor), variant="umve"
    )
    print(f"Custom MMD: {mmd.compute(generated)} \n")


def calculate_custom_pgs(
    reference: Collection[nx.Graph], generated: Collection[nx.Graph]
):
    """
    Calculate a customized PGS between a reference dataset and a generated dataset.

    This PGS uses betweenness centrality and clustering coefficients as graph descriptors. Instead of TabPFN, it uses logistic regression.

    PolyGraphScore may be instantiated with any descriptors implementing the `polygraph.utils.descriptors.GraphDescriptor` interface
    and any classifier implementing the `polygraph.metrics.base.polygraphscore.ClassifierProtocol` interface (i.e., the sklearn interface).
    """
    print("Calculating custom PGS...")
    pgs = PolyGraphScore(
        reference,
        descriptors={
            "betweenness": betweenness_descriptor,
            "clustering": ClusteringHistogram(bins=100),
        },
        classifier=LogisticRegression(),
    )
    result = pgs.compute(generated)

    print(f"Overall PGS: {result['polygraphscore']:.6f}")
    print(f"Score Descriptor: {result['polygraphscore_descriptor']}")
    print("\nSubscores:")
    for metric, score in result["subscores"].items():
        print(f"  {metric.capitalize()}: {score:.6f}")
    print()


if __name__ == "__main__":
    reference = list(PlanarGraphDataset("val").to_nx())
    generated = list(SBMGraphDataset("val").to_nx())

    calculate_custom_mmd(reference, generated)
    print("=" * 50, "\n")
    calculate_custom_pgs(reference, generated)
