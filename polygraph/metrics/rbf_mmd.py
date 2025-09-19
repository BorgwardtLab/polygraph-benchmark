"""MMD metrics using RBF kernels with dynamic bandwidths, as proposed by Thompson et al. [1].

The following graph descriptors are available:

Graph Descriptors:
    - [`OrbitCounts`][polygraph.utils.descriptors.OrbitCounts]: Counts of different graphlet orbits
    - [`ClusteringHistogram`][polygraph.utils.descriptors.ClusteringHistogram]: Distribution of clustering coefficients
    - [`SparseDegreeHistogram`][polygraph.utils.descriptors.SparseDegreeHistogram]: Distribution of node degrees
    - [`EigenvalueHistogram`][polygraph.utils.descriptors.EigenvalueHistogram]: Distribution of graph Laplacian eigenvalues
    - [`RandomGIN`][polygraph.utils.descriptors.RandomGIN]: Graph Neural Network embedding of the graph, combined with a normalization layer ([`NormalizedDescriptor`][polygraph.utils.descriptors.NormalizedDescriptor]). Proposed by Thompson et al. [1].

Below, we demonstrate how to evaluate all metrics in the benchmark with point estimates and with uncertainty quantification.

```python
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset
from polygraph.metrics import RBFMMD2Benchmark, RBFMMD2BenchmarkInterval

reference = list(PlanarGraphDataset("val").to_nx())
generated = list(SBMGraphDataset("val").to_nx())

# Evaluate the benchmark with point estimates
benchmark = RBFMMD2Benchmark(reference[:20])
print(benchmark.compute(generated[:20]))

# Evaluate the benchmark with uncertainty quantification
benchmark_with_uncertainty = RBFMMD2BenchmarkInterval(
    reference,
    subsample_size=20,
    num_samples=100,
    coverage=0.95,
)
print(benchmark_with_uncertainty.compute(generated))
```


References:
    [1] Thompson, R., Knyazev, B., Ghalebi, E., Kim, J., & Taylor, G. W. (2022). [On Evaluation Metrics for Graph Generative Models](https://arxiv.org/abs/2201.09871). In International Conference on Learning Representations (ICLR).
"""

from typing import Collection, Optional, List, Union

import networkx as nx
import numpy as np

from polygraph.metrics.base.mmd import (
    MaxDescriptorMMD2,
    MaxDescriptorMMD2Interval,
)
from polygraph.utils.descriptors import (
    ClusteringHistogram,
    EigenvalueHistogram,
    NormalizedDescriptor,
    OrbitCounts,
    SparseDegreeHistogram,
    RandomGIN,
)
from polygraph.utils.kernels import AdaptiveRBFKernel
from polygraph.metrics.base import MetricCollection

__all__ = [
    "RBFMMD2Benchmark",
    "RBFMMD2BenchmarkInterval",
    "RBFOrbitMMD2",
    "RBFOrbitMMD2Interval",
    "RBFClusteringMMD2",
    "RBFClusteringMMD2Interval",
    "RBFDegreeMMD2",
    "RBFDegreeMMD2Interval",
    "RBFSpectralMMD2",
    "RBFSpectralMMD2Interval",
    "RBFGraphNeuralNetworkMMD2",
    "RBFGraphNeuralNetworkMMD2Interval",
]


class RBFMMD2Benchmark(MetricCollection[nx.Graph]):
    """Collection of MMD2 metrics using RBF kernels with dynamic bandwidths.

    Args:
        reference_graphs: Collection of reference networkx graphs.
    """

    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            metrics={
                "orbit": RBFOrbitMMD2(reference_graphs),
                "clustering": RBFClusteringMMD2(reference_graphs),
                "degree": RBFDegreeMMD2(reference_graphs),
                "spectral": RBFSpectralMMD2(reference_graphs),
                "gin": RBFGraphNeuralNetworkMMD2(reference_graphs),
            },
        )


class RBFMMD2BenchmarkInterval(MetricCollection[nx.Graph]):
    """Collection of MMD2 metrics using RBF kernels with dynamic bandwidths and uncertainty quantification.

    Args:
        reference_graphs: Collection of reference networkx graphs.
        subsample_size: Number of graphs used in each individual MMD2 sample. Should be consistent with the sample size in point estimates.
        num_samples: Number of MMD2 samples used to compute the uncertainty interval.
        coverage: Coverage of the uncertainty interval.
    """

    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: Optional[float] = 0.95,
    ):
        super().__init__(
            metrics={
                "orbit": RBFOrbitMMD2Interval(
                    reference_graphs, subsample_size, num_samples, coverage
                ),
                "clustering": RBFClusteringMMD2Interval(
                    reference_graphs, subsample_size, num_samples, coverage
                ),
                "degree": RBFDegreeMMD2Interval(
                    reference_graphs, subsample_size, num_samples, coverage
                ),
                "spectral": RBFSpectralMMD2Interval(
                    reference_graphs, subsample_size, num_samples, coverage
                ),
                "gin": RBFGraphNeuralNetworkMMD2Interval(
                    reference_graphs, subsample_size, num_samples, coverage
                ),
            },
        )


# Below are the definitions of individual MMD2 metrics


class RBFOrbitMMD2(MaxDescriptorMMD2[nx.Graph]):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=OrbitCounts(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="biased",
        )


class RBFOrbitMMD2Interval(MaxDescriptorMMD2Interval[nx.Graph]):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: Optional[float] = 0.95,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=OrbitCounts(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )


class RBFClusteringMMD2(MaxDescriptorMMD2[nx.Graph]):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=ClusteringHistogram(bins=100),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="biased",
        )


class RBFClusteringMMD2Interval(MaxDescriptorMMD2Interval[nx.Graph]):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: Optional[float] = 0.95,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=ClusteringHistogram(bins=100),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )


class RBFDegreeMMD2(MaxDescriptorMMD2[nx.Graph]):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=SparseDegreeHistogram(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="biased",
        )


class RBFDegreeMMD2Interval(MaxDescriptorMMD2Interval[nx.Graph]):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: Optional[float] = 0.95,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=SparseDegreeHistogram(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )


class RBFSpectralMMD2(MaxDescriptorMMD2[nx.Graph]):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=EigenvalueHistogram(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="biased",
        )


class RBFSpectralMMD2Interval(MaxDescriptorMMD2Interval[nx.Graph]):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: Optional[float] = 0.95,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=EigenvalueHistogram(),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )


class RBFGraphNeuralNetworkMMD2(MaxDescriptorMMD2[nx.Graph]):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        node_feat_loc: Optional[List[str]] = None,
        node_feat_dim: int = 1,
        edge_feat_loc: Optional[List[str]] = None,
        edge_feat_dim: int = 0,
        seed: Union[int, None] = 42,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=NormalizedDescriptor(
                    RandomGIN(
                        node_feat_loc=node_feat_loc,
                        input_dim=node_feat_dim,
                        edge_feat_loc=edge_feat_loc,
                        edge_feat_dim=edge_feat_dim,
                        seed=seed,
                    ),
                    reference_graphs,
                ),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            variant="biased",
        )


class RBFGraphNeuralNetworkMMD2Interval(MaxDescriptorMMD2Interval[nx.Graph]):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: Optional[float] = 0.95,
        node_feat_loc: Optional[List[str]] = None,
        node_feat_dim: int = 1,
        edge_feat_loc: Optional[List[str]] = None,
        edge_feat_dim: int = 0,
        seed: Union[int, None] = 42,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=AdaptiveRBFKernel(
                descriptor_fn=NormalizedDescriptor(
                    RandomGIN(
                        node_feat_loc=node_feat_loc,
                        input_dim=node_feat_dim,
                        edge_feat_loc=edge_feat_loc,
                        edge_feat_dim=edge_feat_dim,
                        seed=seed,
                    ),
                    reference_graphs,
                ),
                bw=np.array(
                    [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
                ),
            ),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )
