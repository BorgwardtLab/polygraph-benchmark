"""StandardPGS is a [`PolyGraphScore`][polygraph.metrics.base.polygraphscore.PolyGraphScore] metric based on six different graph descriptors.

- [`OrbitCounts`][polygraph.utils.descriptors.OrbitCounts]: Counts of different graphlet orbits
- [`ClusteringHistogram`][polygraph.utils.descriptors.ClusteringHistogram]: Distribution of clustering coefficients
- [`SparseDegreeHistogram`][polygraph.utils.descriptors.SparseDegreeHistogram]: Distribution of node degrees
- [`EigenvalueHistogram`][polygraph.utils.descriptors.EigenvalueHistogram]: Distribution of graph Laplacian eigenvalues
- [`RandomGIN`][polygraph.utils.descriptors.RandomGIN]: Graph Neural Network embedding of the graph, combined with a normalization layer ([`NormalizedDescriptor`][polygraph.utils.descriptors.NormalizedDescriptor]). Proposed by Thompson et al. [1].

By default, we use TabPFN for binary classification and evaluate it by data log-likelihood, obtaining a PolyGraphScore that provides an estimated lower bound on the Jensen-Shannon
distance between the generated and true graph distribution.

This metric is implemented in the [`StandardPGS`][polygraph.metrics.StandardPGS] class and can be used as follows:

```python
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset
from polygraph.metrics import StandardPGS

reference = PlanarGraphDataset("val").to_nx()
generated = SBMGraphDataset("val").to_nx()

benchmark = StandardPGS(reference)
print(benchmark.compute(generated))     # {'polygraphscore': 0.9902651620251016, 'polygraphscore_descriptor': 'clustering', 'subscores': {'orbit': 0.9962500491652303, 'clustering': 0.9902651620251016, 'degree': 0.9975117559449073, 'spectral': 0.9634302070519823, 'gin': 0.994213920319544}}
```

We also provide classes that implement individual [`ClassifierMetric`][polygraph.metrics.base.polygraphscore.ClassifierMetric]s:

- [`ClassifierOrbit4Metric`][polygraph.metrics.standard_pgs.ClassifierOrbit4Metric]: Classifier metric based on [`OrbitCounts`][polygraph.utils.descriptors.OrbitCounts]
- [`ClassifierOrbit5Metric`][polygraph.metrics.standard_pgs.ClassifierOrbit5Metric]: Classifier metric based on [`OrbitCounts`][polygraph.utils.descriptors.OrbitCounts]
- [`ClassifierClusteringMetric`][polygraph.metrics.standard_pgs.ClassifierClusteringMetric]: Classifier metric based on [`ClusteringHistogram`][polygraph.utils.descriptors.ClusteringHistogram]
- [`ClassifierDegreeMetric`][polygraph.metrics.standard_pgs.ClassifierDegreeMetric]: Classifier metric based on [`SparseDegreeHistogram`][polygraph.utils.descriptors.SparseDegreeHistogram]
- [`ClassifierSpectralMetric`][polygraph.metrics.standard_pgs.ClassifierSpectralMetric]: Classifier metric based on [`EigenvalueHistogram`][polygraph.utils.descriptors.EigenvalueHistogram]
- [`GraphNeuralNetworkClassifierMetric`][polygraph.metrics.standard_pgs.GraphNeuralNetworkClassifierMetric]: Classifier metric based on [`RandomGIN`][polygraph.utils.descriptors.RandomGIN]
"""

from typing import Collection, Literal, Optional, List, Union

import networkx as nx

from polygraph.metrics.base.polygraphscore import (
    ClassifierMetric,
    PolyGraphScore,
    PolyGraphScoreInterval,
    ClassifierProtocol,
)
from polygraph.utils.descriptors import (
    OrbitCounts,
    ClusteringHistogram,
    SparseDegreeHistogram,
    EigenvalueHistogram,
    RandomGIN,
)

__all__ = [
    "StandardPGS",
    "StandardPGSInterval",
    "ClassifierOrbit4Metric",
    "ClassifierClusteringMetric",
    "ClassifierDegreeMetric",
    "ClassifierSpectralMetric",
    "GraphNeuralNetworkClassifierMetric",
]


class StandardPGS(PolyGraphScore[nx.Graph]):
    """PolyGraphScore metric that combines six different graph descriptors.

    By default, we use TabPFN for binary classification and evaluate it by data log-likelihood, obtaining a PolyGraphScore that provides an estimated lower bound on the Jensen-Shannon
    distance between the generated and true graph distribution.

    Args:
        reference_graphs: Collection of reference graphs.
    """

    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptors={
                "orbit4": OrbitCounts(graphlet_size=4),
                "orbit5": OrbitCounts(graphlet_size=5),
                "clustering": ClusteringHistogram(bins=100),
                "degree": SparseDegreeHistogram(),
                "spectral": EigenvalueHistogram(),
                "gin": RandomGIN(
                    node_feat_loc=None,
                    input_dim=1,
                    edge_feat_loc=None,
                    edge_feat_dim=0,
                    seed=42,
                ),
            },
            variant="jsd",
            classifier=None,
        )


class StandardPGSInterval(PolyGraphScoreInterval[nx.Graph]):
    """StandardPGS with uncertainty quantification.

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
                "orbit4": OrbitCounts(graphlet_size=4),
                "orbig5": OrbitCounts(graphlet_size=5),
                "clustering": ClusteringHistogram(bins=100),
                "degree": SparseDegreeHistogram(),
                "spectral": EigenvalueHistogram(),
                "gin": RandomGIN(
                    node_feat_loc=None,
                    input_dim=1,
                    edge_feat_loc=None,
                    edge_feat_dim=0,
                    seed=42,
                ),
            },
            variant="jsd",
            classifier=None,
            subsample_size=subsample_size,
            num_samples=num_samples,
        )


# Below are the definitions of individual classifier metrics


class ClassifierOrbit4Metric(ClassifierMetric[nx.Graph]):
    """
    Classifier metric based on [`OrbitCounts`][polygraph.utils.descriptors.OrbitCounts].

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
            descriptor=OrbitCounts(graphlet_size=4),
            variant=variant,
            classifier=classifier,
        )


class ClassifierOrbit5Metric(ClassifierMetric[nx.Graph]):
    """
    Classifier metric based on [`OrbitCounts`][polygraph.utils.descriptors.OrbitCounts].
    """

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        variant: Literal["informedness", "jsd"] = "jsd",
        classifier: Optional[ClassifierProtocol] = None,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            descriptor=OrbitCounts(graphlet_size=5),
            variant=variant,
            classifier=classifier,
        )


class ClassifierClusteringMetric(ClassifierMetric[nx.Graph]):
    """
    Classifier metric based on [`ClusteringHistogram`][polygraph.utils.descriptors.ClusteringHistogram].

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
    Classifier metric based on [`SparseDegreeHistogram`][polygraph.utils.descriptors.SparseDegreeHistogram].

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
    Classifier metric based on [`EigenvalueHistogram`][polygraph.utils.descriptors.EigenvalueHistogram].

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
    Classifier metric based on [`RandomGIN`][polygraph.utils.descriptors.RandomGIN].

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
            descriptor=RandomGIN(
                node_feat_loc=node_feat_loc,
                input_dim=node_feat_dim,
                edge_feat_loc=edge_feat_loc,
                edge_feat_dim=edge_feat_dim,
                seed=seed,
            ),
            variant=variant,
            classifier=classifier,
        )
