"""MMD metrics based on graph descriptors introduced by You et al. [1] and Liao et al. [2], using a Gaussian TV kernel.

We provide both point estimates of MMD and uncertainty quantifications. The following graph descriptors are available:

Graph Descriptors:
    - [`OrbitCounts`][polygraph.utils.descriptors.OrbitCounts]: Counts of different graphlet orbits
    - [`ClusteringHistogram`][polygraph.utils.descriptors.ClusteringHistogram]: Distribution of clustering coefficients
    - [`SparseDegreeHistogram`][polygraph.utils.descriptors.SparseDegreeHistogram]: Distribution of node degrees
    - [`EigenvalueHistogram`][polygraph.utils.descriptors.EigenvalueHistogram]: Distribution of graph Laplacian eigenvalues


The [`GaussianTV`][polygraph.utils.kernels.GaussianTV] kernel is used with descriptor-specific bandwidths.


Warning:
    The Gaussian TV kernel is not positive definite, as shown by O'Bray et al. [3]. While it is most widely used in the literature, consider also evaluating the linear and RBF kernels.


Below, we demonstrate how to evaluate all metrics in the benchmark with point estimates and with uncertainty quantification.
Note that the parameter `subsample_size` in [`GaussianTVMMD2BenchmarkInterval`][polygraph.metrics.GaussianTVMMD2BenchmarkInterval]
should match the number of generated and reference graphs in [`GaussianTVMMD2Benchmark`][polygraph.metrics.GaussianTVMMD2Benchmark]
to obtain comparable results:

```python
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset
from polygraph.metrics import GaussianTVMMD2Benchmark, GaussianTVMMD2BenchmarkInterval

reference = list(PlanarGraphDataset("val").to_nx())
generated = list(SBMGraphDataset("val").to_nx())

# Evaluate the benchmark with point estimates
benchmark = GaussianTVMMD2Benchmark(reference[:20])
print(benchmark.compute(generated[:20]))

# Evaluate the benchmark with uncertainty quantification
benchmark_with_uncertainty = GaussianTVMMD2BenchmarkInterval(
    reference,
    subsample_size=20,
    num_samples=100,
    coverage=0.95,
)
print(benchmark_with_uncertainty.compute(generated))
```

The individual metrics are also provided in seperate classes, e.g. [`GaussianTVOrbitMMD2`][polygraph.metrics.gaussian_tv_mmd.GaussianTVOrbitMMD2] and [`GaussianTVOrbitMMD2Interval`][polygraph.metrics.gaussian_tv_mmd.GaussianTVOrbitMMD2Interval].

References:
    [1] You, J., Ying, R., Ren, X., Hamilton, W., & Leskovec, J. (2018). [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models](https://arxiv.org/abs/1802.08773). In International Conference on Machine Learning (ICML).

    [2] Liao, R., Li, Y., Song, Y., Wang, S., Hamilton, W., Duvenaud, D., Urtasun, R., & Zemel, R. (2019). [Efficient Graph Generation with Graph Recurrent Attention Networks](https://arxiv.org/abs/1910.00760). In Advances in Neural Information Processing Systems (NeurIPS).

    [3] O'Bray, L., Horn, M., Rieck, B., & Borgwardt, K. (2022). [Evaluation Metrics for Graph Generative Models: Problems, Pitfalls, and Practical Solutions](https://arxiv.org/abs/2106.01098). In International Conference on Learning Representations (ICLR).
"""

from typing import Collection, Optional

import networkx as nx

from polygraph.metrics.base import MetricCollection
from polygraph.metrics.base.mmd import (
    DescriptorMMD2,
    DescriptorMMD2Interval,
)
from polygraph.utils.descriptors import (
    ClusteringHistogram,
    EigenvalueHistogram,
    OrbitCounts,
    SparseDegreeHistogram,
)
from polygraph.utils.kernels import GaussianTV


__all__ = [
    "GaussianTVMMD2Benchmark",
    "GaussianTVMMD2BenchmarkInterval",
    "GaussianTVOrbitMMD2",
    "GaussianTVOrbitMMD2Interval",
    "GaussianTVClusteringMMD2",
    "GaussianTVClusteringMMD2Interval",
    "GaussianTVDegreeMMD2",
    "GaussianTVDegreeMMD2Interval",
    "GaussianTVSpectralMMD2",
    "GaussianTVSpectralMMD2Interval",
]


class GaussianTVMMD2Benchmark(MetricCollection[nx.Graph]):
    """Collection of MMD2 metrics using the Gaussian TV kernel.

    This graphs combines the following graph descriptors into one benchmark:

    - [`OrbitCounts`][polygraph.utils.descriptors.OrbitCounts]
    - [`ClusteringHistogram`][polygraph.utils.descriptors.ClusteringHistogram]
    - [`SparseDegreeHistogram`][polygraph.utils.descriptors.SparseDegreeHistogram]
    - [`EigenvalueHistogram`][polygraph.utils.descriptors.EigenvalueHistogram]

    Args:
        reference_graphs: Collection of reference graphs to fit the metric to.
    """

    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            metrics={
                "orbit": GaussianTVOrbitMMD2(reference_graphs),
                "clustering": GaussianTVClusteringMMD2(reference_graphs),
                "degree": GaussianTVDegreeMMD2(reference_graphs),
                "spectral": GaussianTVSpectralMMD2(reference_graphs),
            },
        )


class GaussianTVMMD2BenchmarkInterval(MetricCollection[nx.Graph]):
    """Collection of MMD2 metrics using the Gaussian TV kernel with uncertainty quantification.

    This class provides the same metrics as [`GaussianTVMMD2Benchmark`][polygraph.metrics.GaussianTVMMD2Benchmark] but with uncertainty quantification.

    Args:
        reference_graphs: Collection of reference graphs to fit the metric to.
        subsample_size: Size of the subsample used to compute the uncertainty interval.
        num_samples: Number of samples used to compute the uncertainty interval.
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
                "orbit": GaussianTVOrbitMMD2Interval(
                    reference_graphs, subsample_size, num_samples, coverage
                ),
                "clustering": GaussianTVClusteringMMD2Interval(
                    reference_graphs, subsample_size, num_samples, coverage
                ),
                "degree": GaussianTVDegreeMMD2Interval(
                    reference_graphs, subsample_size, num_samples, coverage
                ),
                "spectral": GaussianTVSpectralMMD2Interval(
                    reference_graphs, subsample_size, num_samples, coverage
                ),
            },
        )


# Below are the definitions of individual MMD2 metrics


class GaussianTVOrbitMMD2(DescriptorMMD2[nx.Graph]):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=OrbitCounts(), bw=30),
            variant="biased",
        )


class GaussianTVOrbitMMD2Interval(DescriptorMMD2Interval[nx.Graph]):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: Optional[float] = 0.95,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=OrbitCounts(), bw=30),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )


class GaussianTVClusteringMMD2(DescriptorMMD2[nx.Graph]):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(
                descriptor_fn=ClusteringHistogram(bins=100), bw=1.0 / 10
            ),
            variant="biased",
        )


class GaussianTVClusteringMMD2Interval(DescriptorMMD2Interval[nx.Graph]):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: Optional[float] = 0.95,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(
                descriptor_fn=ClusteringHistogram(bins=100), bw=1.0 / 10
            ),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )


class GaussianTVDegreeMMD2(DescriptorMMD2[nx.Graph]):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=SparseDegreeHistogram(), bw=1.0),
            variant="biased",
        )


class GaussianTVDegreeMMD2Interval(DescriptorMMD2Interval[nx.Graph]):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: Optional[float] = 0.95,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=SparseDegreeHistogram(), bw=1.0),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )


class GaussianTVSpectralMMD2(DescriptorMMD2[nx.Graph]):
    def __init__(self, reference_graphs: Collection[nx.Graph]):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=EigenvalueHistogram(), bw=1.0),
            variant="biased",
        )


class GaussianTVSpectralMMD2Interval(DescriptorMMD2Interval[nx.Graph]):
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        subsample_size: int,
        num_samples: int = 500,
        coverage: Optional[float] = 0.95,
    ):
        super().__init__(
            reference_graphs=reference_graphs,
            kernel=GaussianTV(descriptor_fn=EigenvalueHistogram(), bw=1.0),
            subsample_size=subsample_size,
            num_samples=num_samples,
            coverage=coverage,
            variant="biased",
        )
