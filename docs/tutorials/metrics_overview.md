# Overview of Metrics

As noted in the previous tutorial, most metrics provided in this package follow a similar interface:
First, an instance of the metric is initialized by fitting it on the reference set.
Then, it is evaluated against a generated set by calling `.compute(generated)`.
For a technical documentation of this interface, we refer to the [API reference](../api_reference/metrics/interface.md).

For convenience, `polygraph` allows metrics that follow this interface to be bundled and evaluated jointly by using the [`MetricCollection`][polygraph.metrics.MetricCollection] class.

```python
from polygraph.metrics import MetricCollection
from polygraph.metrics import MMD2CollectionRBF, MMD2CollectionGaussianTV
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset

reference_graphs = PlanarGraphDataset("val").to_nx()
generated_graphs = SBMGraphDataset("val").to_nx()

metrics = MetricCollection(
    metrics={
        "rbf_mmd": MMD2CollectionRBF(reference_graphs),
        "tv_mmd": MMD2CollectionGaussianTV(reference_graphs),
    }
)
print(metrics.compute(generated_graphs))        # Dictionary of metrics
```

We now proceed to give a high-level overview over the different types of metrics that we implement.

## Maximum Mean Discrepancy

[Maximum Mean Discrepancy (MMD)](../api_reference/metrics/mmd.md) is the predominant method for comparing graph distributions.
The two distributions are embedded in a reproducing kernel Hilbert space (RKHS) and their distance is then computed in this space.

In `polygraph`, we bundle the most commonly used MMD metrics in two benchmark classes: [`MMD2CollectionGaussianTV`][polygraph.metrics.MMD2CollectionGaussianTV] and [`MMD2CollectionRBF`][polygraph.metrics.MMD2CollectionRBF]. These benchmarks may be evaluated in the following fashion:

```python
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset
from polygraph.metrics import MMD2CollectionGaussianTV, MMD2IntervalCollectionGaussianTV

reference = PlanarGraphDataset("val").to_nx()
generated = SBMGraphDataset("val").to_nx()

# Evaluate the benchmark with point estimates
benchmark = MMD2CollectionGaussianTV(reference)
print(benchmark.compute(generated))     # {'orbit': 1.067700488335175, 'clustering': 0.32549637224264394, 'degree': 0.3375409762261701, 'spectral': 0.0830197437100697}
```

For more details on these collections we refer to the documentation on the [Gaussian TV metrics](../metrics/gaussian_tv_mmd.md) and [RBF metrics](../metrics/rbf_mmd.md).

Polygraph also allows you to construct custom MMD metrics. To construct an MMD metric, one must choose two components:

- [Descriptor](../api_reference/utils/graph_descriptors.md) - A function that transforms graphs into vectorial descriptions
- [Kernel](../api_reference/utils/graph_kernels.md) - A kernel function operating on the vectors produced by the descriptor

We implement a large number of different descriptors and kernels in `polygraph`.
An MMD metric operating on orbit counts with a linear kernel may thus be constructed in the following fashion:

```python
from polygraph.utils.graph_descriptors import OrbitCounts
from polygraph.utils.kernels import LinearKernel
from polygraph.metrics.base import DescriptorMMD2
from polygraph.datasets import PlanarGraphDataset

metric = DescriptorMMD2(
    PlanarGraphDataset("test").to_nx(),
    kernel=LinearKernel(OrbitCounts(graphlet_size=4)),
    variant="biased",
)
```

The MMD may be computed via a biased estimator (`"biased"`) or via an unbiased one (`"umve"`).
In the large sample size limit, the two should converge to the same value. However, at low sample sizes the differences may be substantial.
In practice the biased estimator is oftentimes used. We refer to the documentation of the [base MMD classes](../api_reference/metrics/mmd.md).


!!! warning
    MMD metrics that are computed with different estimators, metrics, or kernels lie on different scales and are not comparable to each other.

## PolyGraphScore

The [PolyGraphScore metric](../api_reference/metrics/polygraphscore.md) compares two graph distributions by determining how well they can be distinguished by a binary classifier.
It aims to make metrics comparable across graph descriptors and produces interpretable values between 0 and 1.
The PolyGraphScore is computed for several graph descriptors and produces a summary metric for these descriptors.
This summary metric is an estimated lower bound on a probability metric that is intrinsic to the graph distributions and independent of the descriptors

We provide [`PGS5`][polygraph.metrics.PGS5], a standardized version of the PolyGraphScore that combines 5 different graph descriptors:

```python
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset
from polygraph.metrics import PGS5

metric = PGS5(reference_graphs=PlanarGraphDataset("test").to_nx())
metric.compute(SBMGraphDataset("test").to_nx()) # {'polygraphscore': 0.999301797449604, 'polygraphscore_descriptor': 'degree', 'subscores': {'orbit': 0.9986018004713674, 'clustering': 0.9933180272388359, 'degree': 0.999301797449604, 'spectral': 0.9690467491487502, 'gin': 0.9984711185804029}}
```

As with MMD metrics, you may also construct custom PolyGraphScore variants using other graph descriptors, evaluation metrics, or binary classification approaches.
E.g., you may construct the following metric

```python
from polygraph.utils.graph_descriptors import OrbitCounts, SparseDegreeHistogram
from polygraph.metrics.base import PolyGraphScore

metric = PolyGraphScore(
    reference_graphs=PlanarGraphDataset("test").to_nx(),
    descriptors={
        "orbit": OrbitCounts(),
        "degree": SparseDegreeHistogram(),
    },
    classifier="logistic",
    variant="informedness",
)
metric.compute(SBMGraphDataset("test").to_nx())         # {'polygraphscore': 0.9, 'polygraphscore_descriptor': 'orbit', 'subscores': {'orbit': 0.9, 'degree': 0.9}}
```

We refer to the [API reference](../api_reference/metrics/polygraphscore.md) for further details.

## Validity, Uniqueness, Novelty

Synthetic datasets oftentimes satisfy structural constraints (e.g. that they contain tree graphs) and thereby have a notion of *validity* of graphs.
Consequently, generative models may be evaluated by the fraction of generated graphs that fulfill these constraints and are thereby valid.
Moreover, one may compute the fraction of graphs that are *unique* within the generated set and the fraction of graphs that are *novel* in the sense
that they do not occur in the training set. The fraction of graphs that satisfy all three properties (validity, uniqueness, and novelty) is referred to as VUN.

We implement these metrics in the [`VUN`][polygraph.metrics.VUN] class.
To determine novelty, this metric must be passed the training set on which the generative model was optimized. The test set, on the other hand, must not be passed to this class:

```python
from polygraph.metrics import VUN

train = PlanarGraphDataset("train").to_nx()
generated = SBMGraphDataset("val").to_nx()

metric = VUN(
    train_graphs=train,         # Pass the training set to determine novelty
    validity_fn=PlanarGraphDataset.is_valid
)
print(metric.compute(generated))        # {'unique': 1.0, 'novel': 1.0, 'unique_novel': 1.0, 'valid': 0.0, 'valid_unique_novel': 0.0, 'valid_novel': 0.0, 'valid_unique': 0.0}
```

All synthetic datasets in the `polygraph` package provide a static `is_valid` function.
If no validity function is available for your dataset, `validity_fn` may be set to `None`. In this case, only the fraction of unique and novel graphs is computed.
