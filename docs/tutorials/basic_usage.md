# Basic Usage

In this tutorial, we introduce the most basic usage of the `polygraph` package.
We first load two datasets of graphs and then compare how similar these datasets are by computing an MMD metric.

## Loading Datasets of Graphs

Polygraph comes with the most commonly used datasets in graph generation.
We provide documentation for all provided datasets in [here](../datasets/index.md).
Below, we load two synthetic datasets and inspect an element from one of them:

```python
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset

planar = PlanarGraphDataset("train")
sbm = SBMGraphDataset("val")
print(planar[0])            # PyG object: Data(edge_index=[2, 354], num_nodes=64)
```

This will download the dataseets to your device and cache them. You may specify the download location by setting the environment variable `POLYGRAPH_CACHE_DIR`.
All datasets in `polygraph` contain PyTorch-geometric objects.
However, we may also access the graphs as NetworkX objects as follows:

```python
planar_nx = planar.to_nx()
print(planar_nx[0])         # (Networkx) Graph with 64 nodes and 177 edges
```

## Comparing Distributions of Graphs
When evaluating graph generative models, we want to compare a set of *generated* graphs to a set of *reference* graphs (typically the test set).
In `polygraph`, we provide various different metrics to quantify how close these two sets of graphs are.
We usually pass collections of NetworkX graphs to metrics.
Below, we demonstrate how one such metric, [`GRANOrbitMMD2`][polygraph.metrics.gran.GRANOrbitMMD2] may be computed:

```python
from polygraph.metrics.gran import GRANOrbitMMD2

reference = planar.to_nx()
generated = sbm.to_nx()

metric = GRANOrbitMMD2(reference)
print(metric.compute(generated))          # 1.063501049655686
```

We discuss other available metrics [in the next tutorial](metrics_overview.md).

All metrics are evaluated in a similar fashion:

- We first initialize a metric object via `metric = GRANOrbitMMD2(reference)`. This fits the metric to the `reference` set, caching graph descriptors that are required in later computations
- We then compute the metric against the generated set via `metric.compute(generated)`
- We note that we may call `metric.compute` repeatedly with different generated sets, e.g. over the course of training
