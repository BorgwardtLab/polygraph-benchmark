# PolyGraph

PolyGraph is a Python library for evaluating graph generative models by providing standardized datasets and metrics 
(including PolyGraphScore).

## At a glance

Here are a set of datasets and metrics this library provides:
- **Datasets**: ready-to-use splits for procedural and real-world graphs
  - Procedural datasets: `PlanarLGraphDataset`, `SBMLGraphDataset`, `LobsterLGraphDataset`
  - Real-world: `QM9`, `MOSES`, `Guacamol`, `DobsonDoigGraphDataset`, `ModelNet10GraphDataset`
  - Also: `EgoGraphDataset`, `PointCloudGraphDataset`
- **Metrics**: unified, fit-once/compute-many interface with convenience wrappers, avoiding redundant computations.
  - MMD<sup>2</sup>: `GaussianTVMMD2Benchmark`, `RBFMMD2Benchmark`
  - Kernel hyperparameter optimization with `MaxDescriptorMMD2`.
  - PolyGraphScore: `PGS5`.
  - Validation/Uniqueness/Novelty: `VUN`.
  - Uncertainty quantification for benchmarking (`GaussianTVMMD2BenchmarkInterval`, `RBFMMD2Benchmark`, `PGS5Interval`)
- **Interoperability**: works with PyTorch Geometric and NetworkX; caching via `POLYGRAPH_CACHE_DIR`.


<details>
<summary><strong>⚠️ Important - Dataset Usage Warning</strong></summary>

**To help reproduce previous results, we provide the following datasets:**
- `PlanarGraphDataset`
- `SBMGraphDataset` 
- `LobsterGraphDataset`

But they should not be used for benchmarking, due to unreliable metric estimates (see our paper for more details).

We provide larger datasets that should be used instead:
- `PlanarLGraphDataset`
- `SBMLGraphDataset` 
- `LobsterLGraphDataset`

</details>

## Installation

```bash
pip install polygraph-benchmark
```

No manual compilation of ORCA is required. For details on interaction with `graph_tool`, see the more detailed installation instructions in the docs.

## Tutorial

### Datasets
Instantiate a benchmark dataset as follows:
```python
from polygraph.datasets import PlanarGraphDataset

reference = PlanarGraphDataset("test")

reference_nx = reference.to_nx()

# Let's also generate some graphs coming from another distribution.
generated = [nx.erdos_renyi_graph(64, 0.1) for _ in range(40)]
```

### Metrics

#### Maximum Mean Discrepancy
To compute existing MMD2 formulations (e.g. based on the TV pseudokernel), one can use the following:
```python
from polygraph.metrics import GaussianTVMMD2Benchmark # Can also be RBFMMD2Benchmark

gtv_benchmark = GaussianTVMMD2Benchmark(reference)

print(gtv_benchmark.compute(generated))  # {'orbit': ..., 'clustering': ..., 'degree': ..., 'spectral': ...}
```

#### PolyGraphScore
Similarly, you can compute our proposed PolyGraphScore, like so:

```python
from polygraph.metrics import StandardPGS 

pgs = StandardPGS(reference)
print(pgs.compute(generated)) # {'polygraphscore': ..., 'polygraphscore_descriptor': ..., 'subscores': {'orbit': ..., }}
```

`polygraphscore_descriptor` provides the best descriptor used to report the final score.

#### Validity, uniqueness and novelty
VUN values follow a similar interface:
```python
from polygraph.metrics import VUN

pgs = VUN(reference, validity_fn=reference.is_valid, confidence_level=0.95) # if applicable, validity functions are defined as a dataset attribute
print(pgs.compute(generated))  # {'valid': ..., 'valid_unique_novel': ..., 'valid_novel': ..., 'valid_unique': ...}
```

#### Metric uncertainty quantification

For MMD and PGS, uncertainty quantifiation for the metrics are obtained through subsampling. For VUN, a confidence interval is obtained with a binomial test.

For `VUN`, the results can be obtained by specifying a confidence level when instantiating the metric. 

For the others, the `Interval` suffix references the class that implements subsampling.

```python

from polygraph.metrics import GaussianTVMMD2BenchmarkInterval, RBFMMD2BenchmarkInterval, PGS5Interval

metrics = [
	GaussianTVMMD2BenchmarkInterval(), 
	RBFMMD2BenchmarkInterval(), 
	StandardPGSInterval()
]

for metric in metrics:
	metric_results = metric.compute(
		subsample_size=100, # Number of subsamples to consider for each estimate
		num_samples=500,    # Number of estimates to compute
		coverage=0.95       # Optional, coverage of the quantiles (here, 5th and 95th percentile)
	)

metrics_results[0] # "MetricInterval(mean=..., std=..., low=..., high=..., coverage=...)"
```

