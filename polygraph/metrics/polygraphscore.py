"""PGS5 is a [`PolyGraphScore`][polygraph.metrics.base.polygraphscore.PolyGraphScore] metric based on five different graph descriptors.

- [`OrbitCounts`][polygraph.descriptors.OrbitCounts]: Counts of different graphlet orbits
- [`ClusteringHistogram`][polygraph.descriptors.ClusteringHistogram]: Distribution of clustering coefficients
- [`SparseDegreeHistogram`][polygraph.descriptors.SparseDegreeHistogram]: Distribution of node degrees
- [`EigenvalueHistogram`][polygraph.descriptors.EigenvalueHistogram]: Distribution of graph Laplacian eigenvalues
- [`RandomGIN`][polygraph.descriptors.RandomGIN]: Graph Neural Network embedding of the graph, combined with a normalization layer ([`NormalizedDescriptor`][polygraph.descriptors.NormalizedDescriptor]). Proposed by Thompson et al. [1].

By default, we use TabPFN for binary classification and evaluate it by data log-likelihood, obtaining a PolyGraphScore that provides an estimated lower bound on the Jensen-Shannon
distance between the generated and true graph distribution.

This metric is implemented in the [`PGS5`][polygraph.metrics.PGS5] class and can be used as follows:

```python
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset
from polygraph.metrics import PGS5

reference = PlanarGraphDataset("val").to_nx()
generated = SBMGraphDataset("val").to_nx()

benchmark = PGS5(reference)
print(benchmark.compute(generated))     # {'polygraphscore': 0.9902651620251016, 'polygraphscore_descriptor': 'clustering', 'subscores': {'orbit': 0.9962500491652303, 'clustering': 0.9902651620251016, 'degree': 0.9975117559449073, 'spectral': 0.9634302070519823, 'gin': 0.994213920319544}}
```

We also provide classes that implement individual [`ClassifierMetric`][polygraph.metrics.base.polygraphscore.ClassifierMetric]s:

- [`ClassifierOrbitMetric`][polygraph.metrics.polygraphscore.ClassifierOrbitMetric]: Classifier metric based on [`OrbitCounts`][polygraph.descriptors.OrbitCounts]
- [`ClassifierClusteringMetric`][polygraph.metrics.polygraphscore.ClassifierClusteringMetric]: Classifier metric based on [`ClusteringHistogram`][polygraph.descriptors.ClusteringHistogram]
- [`ClassifierDegreeMetric`][polygraph.metrics.polygraphscore.ClassifierDegreeMetric]: Classifier metric based on [`SparseDegreeHistogram`][polygraph.descriptors.SparseDegreeHistogram]
- [`ClassifierSpectralMetric`][polygraph.metrics.polygraphscore.ClassifierSpectralMetric]: Classifier metric based on [`EigenvalueHistogram`][polygraph.descriptors.EigenvalueHistogram]
- [`GraphNeuralNetworkClassifierMetric`][polygraph.metrics.polygraphscore.GraphNeuralNetworkClassifierMetric]: Classifier metric based on [`RandomGIN`][polygraph.descriptors.RandomGIN]
"""

from typing import Collection, Literal, Optional, List, Union

import networkx as nx

from polygraph.metrics.base.polygraphscore import (
    ClassifierMetric,
    PolyGraphScore,
    PolyGraphScoreInterval,
    ClassifierProtocol,
)
from polygraph.descriptors import (
    OrbitCounts,
    ClusteringHistogram,
    SparseDegreeHistogram,
    EigenvalueHistogram,
    NormalizedDescriptor,
    RandomGIN,
)

__all__ = [
    "PGS5",
    "PGS5Interval",
    "ClassifierOrbitMetric",
    "ClassifierClusteringMetric",
    "ClassifierDegreeMetric",
    "ClassifierSpectralMetric",
    "GraphNeuralNetworkClassifierMetric",
]


class PGS5(PolyGraphScore[nx.Graph]):
    """PolyGraphScore metric that combines five different graph descriptors.

    By default, we use TabPFN for binary classification and evaluate it by data log-likelihood, obtaining a PolyGraphScore that provides an estimated lower bound on the Jensen-Shannon
    distance between the generated and true graph distribution.

    Args:
        reference_graphs: Collection of reference graphs.
    """

    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptors={
                "orbit": OrbitCounts(),
                "clustering": ClusteringHistogram(bins=100),
                "degree": SparseDegreeHistogram(),
                "spectral": EigenvalueHistogram(),
                "gin": NormalizedDescriptor(
                    RandomGIN(
                        node_feat_loc=None,
                        input_dim=1,
                        edge_feat_loc=None,
                        edge_feat_dim=0,
                        seed=42,
                    ),
                    reference_graphs,
                ),
            },
            variant="jsd",
            classifier=None,
        )


class PGS5Interval(PolyGraphScoreInterval[nx.Graph]):
    """PGS5 with uncertainty quantification.

    Args:
        reference_graphs: Collection of reference graphs.
        subsample_size: Size of each subsample, should be consistent with the number
            of reference and generated graphs passed to [`PolyGraphScore`][polygraph.metrics.base.polygraphscore.PolyGraphScore]
            for point estimates.
        num_samples: Number of samples to draw for uncertainty quantification.
    """

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 10,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptors={
                "orbit": OrbitCounts(),
                "clustering": ClusteringHistogram(bins=100),
                "degree": SparseDegreeHistogram(),
                "spectral": EigenvalueHistogram(),
                "gin": NormalizedDescriptor(
                    RandomGIN(
                        node_feat_loc=None,
                        input_dim=1,
                        edge_feat_loc=None,
                        edge_feat_dim=0,
                        seed=42,
                    ),
                    reference_graphs,
                ),
            },
            variant="jsd",
            classifier=None,
            subsample_size=subsample_size,
            num_samples=num_samples,
        )


# Below are the definitions of individual classifier metrics


class ClassifierOrbitMetric(ClassifierMetric[nx.Graph]):
    """
    Classifier metric based on [`OrbitCounts`][polygraph.descriptors.OrbitCounts].

    Args:
        reference_graphs: Collection of reference graphs.
        variant: Probability metric to approximate.
        classifier: Binary classifier to fit.
    """

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Optional[ClassifierProtocol] = None,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=OrbitCounts(),
            variant=variant,
            classifier=classifier,
        )


class ClassifierClusteringMetric(ClassifierMetric[nx.Graph]):
    """
    Classifier metric based on [`ClusteringHistogram`][polygraph.descriptors.ClusteringHistogram].

    Args:
        reference_graphs: Collection of reference graphs.
        variant: Probability metric to approximate.
        classifier: Binary classifier to fit.
    """

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Optional[ClassifierProtocol] = None,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=ClusteringHistogram(bins=100),
            variant=variant,
            classifier=classifier,
        )


class ClassifierDegreeMetric(ClassifierMetric[nx.Graph]):
    """
    Classifier metric based on [`SparseDegreeHistogram`][polygraph.descriptors.SparseDegreeHistogram].

    Args:
        reference_graphs: Collection of reference graphs.
        variant: Probability metric to approximate.
        classifier: Binary classifier to fit.
    """

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Optional[ClassifierProtocol] = None,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=SparseDegreeHistogram(),
            variant=variant,
            classifier=classifier,
        )


class ClassifierSpectralMetric(ClassifierMetric[nx.Graph]):
    """
    Classifier metric based on [`EigenvalueHistogram`][polygraph.descriptors.EigenvalueHistogram].

    Args:
        reference_graphs: Collection of reference graphs.
        variant: Probability metric to approximate.
        classifier: Binary classifier to fit.
    """

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Optional[ClassifierProtocol] = None,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=EigenvalueHistogram(),
            variant=variant,
            classifier=classifier,
        )


class GraphNeuralNetworkClassifierMetric(ClassifierMetric[nx.Graph]):
    """
    Classifier metric based on [`RandomGIN`][polygraph.descriptors.RandomGIN].

    Args:
        reference_graphs: Collection of reference graphs.
        variant: Probability metric to approximate.
        classifier: Binary classifier to fit.
    """

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Optional[ClassifierProtocol] = None,
        node_feat_loc: Optional[List[str]] = None,
        node_feat_dim: int = 1,
        edge_feat_loc: Optional[List[str]] = None,
        edge_feat_dim: int = 0,
        seed: Union[int, None] = 42,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=NormalizedDescriptor(
                RandomGIN(
                    node_feat_loc=node_feat_loc,
                    input_dim=node_feat_dim,
                    edge_feat_loc=edge_feat_loc,
                    edge_feat_dim=edge_feat_dim,
                    seed=seed,
                ),
                reference_graphs,
            ),
            variant=variant,
            classifier=classifier,
        )
