# Overview of Metrics

As noted in the previous tutorial, most metrics provided in this package follow a similar interface:
First, an instance of the metric is initialized by fitting it on the reference set.
Then, it is evaluated against a generated set by calling `.compute(generated)`.
For a technical documentation of this interface, we refer to the [API reference](../api_reference/metrics/interface.md).

For convenience, `polygraph` allows metrics that follow this interface to be bundled and evaluated jointly by using the [`MetricCollection`][polygraph.metrics.MetricCollection] class.

```python
from polygraph.metrics import MetricCollection
from polygraph.metrics.gran import RBFOrbitMMD2, ClassifierOrbitMetric
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset

reference_graphs = PlanarGraphDataset("val").to_nx()
generated_graphs = SBMGraphDataset("val").to_nx()

metrics = MetricCollection(
    metrics={
        "rbf_orbit": RBFOrbitMMD2(reference_graphs=reference_graphs),
        "classifier_orbit": ClassifierOrbitMetric(reference_graphs=reference_graphs),
    }
)
print(metrics.compute(generated_graphs))        # Dictionary of metrics
```

We now proceed to give a high-level overview over the different types of metrics that we implement.

## Maximum Mean Discrepancy

[Maximum Mean Discrepancy (MMD)](../api_reference/metrics/mmd.md) is the most commonly used approach for comparing graph distributions.
The two distributions are embedded in a reproducing kernel Hilbert space (RKHS) and their distance is then computed in this space.

To construct an MMD metric, one must choose two components:

- [Descriptor](../api_reference/utils/graph_descriptors.md) - A function that transforms graphs into vectorial descriptions
- [Kernel](../api_reference/utils/graph_kernels.md) - A kernel function operating on these vectors produced by the descriptor

We implement a large number of different descriptors and kernels in `polygraph`.
For convenience, we provide commonly used combinations of kernels, descriptors, and estimators, based on [classical descriptors](../metrics/gran.md) or [gnn features](../metrics/gin.md).
We recommend using these standardized implementations to ensure fair and comparable evaluations.

However, you may also construct custom MMD metrics, combining kernels and descriptors as you like.
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
In practice the biased estimator is oftentimes used.


!!! warning
    MMD metrics that are computed with different estimators, metrics, or kernels lie on different scales and are not comparable to each other.

## PolyGraphScore

The [PolyGraphScore metric](../api_reference/metrics/polygraphscore.md) operates in a similar fashion as MMD metrics. However, it aims to make metrics comparable across graph descriptors and produces interpretable values between 0 and 1.
The PolyGraphScore is typically computed for several graph descriptors and produces a summary metric for these descriptors.
This summary metric is an estimated lower bound on a probability metric that is intrinsic to the graph distributions and independent of the descriptors

```python
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset
from polygraph.utils.graph_descriptors import OrbitCounts, SparseDegreeHistogram
from polygraph.metrics.base import PolyGraphScore

metric = PolyGraphScore(
    reference_graphs=PlanarGraphDataset("test").to_nx(),
    descriptors={
        "orbit": OrbitCounts(),
        "degree": SparseDegreeHistogram(),
    },
    classifier="tabpfn",
    variant="jsd"
)
metric.compute(SBMGraphDataset("test").to_nx())
```


## Validity, Uniqueness, Novelty

Synthetic datasets oftentimes satisfy structural constraints (e.g. that they contain tree graphs) and thereby have a notion of *validity* of graphs.
Consequently, generative models may be evaluated by the fraction of generated graphs that fulfill these constraints and are thereby valid.
Moreover, one may compute the fraction of graphs that are *unique* within the generated set and the fraction of graphs that are *novel* in the sense
that they do not occur in the training set. The fraction of graphs that satisfy all three properties (validity, uniqueness, and novelty) is referred to as VUN.

We implement these metrics in the [`VUN`][polygraph.metrics.VUN] class.
To determine novelty, this metric must be passed the training set on which the generative model was optimized. The test set, on the other hand, must not be passed to this class:

```python
from polygraph.metrics import VUN
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset

train = PlanarGraphDataset("train").to_nx()
generated = SBMGraphDataset("val").to_nx()

metric = VUN(
    train_graphs=train,         # Pass the training set to determine novelty
    validity_fn=PlanarGraphDataset.is_valid
)
print(metric.compute(generated))        # Dictionary containing fraction of unique/novel/valid graphs (all combinations)
```

All synthetic datasets in the `polygraph` package provide a static `is_valid` function.
If no validity function is available for your dataset, `validity_fn` may be set to `None`. In this case, only the fraction of unique and novel graphs is computed.


## Uncertaingy Quantification
