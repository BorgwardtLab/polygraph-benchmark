<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/logo_icon_Dark_NordDark.png">
  <source media="(prefers-color-scheme: light)" srcset="logo/logo_icon_Light_NordLight.png">
  <img src="https://raw.githubusercontent.com/BorgwardtLab/polygraph-benchmark/refs/heads/master/logo/logo_Light_NordLight.png" alt="PolyGraph icon" height="128">
  </picture>
  <br>
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/logo_Dark_NordDark.png">
  <source media="(prefers-color-scheme: light)" srcset="logo/logo_Light_NordLight.png">
  <img src="https://raw.githubusercontent.com/BorgwardtLab/polygraph-benchmark/refs/heads/master/logo/logo_icon_Light_NordLight.png" alt="PolyGraph logo" height="100">
  </picture>
</p>

PolyGraph is a Python library for evaluating graph generative models by providing standardized datasets and metrics
(including PolyGraph Discrepancy).

PolyGraph discrepancy is a new metric we introduced, which provides the following advantages over maxmimum mean discrepancy (MMD):

<table>
<thead>
<tr>
  <th>Property</th>
  <th>MMD</th>
  <th>PGD</th>
</tr>
</thead>
<tbody>
<tr>
  <td>Range</td>
  <td>[0, ∞)</td>
  <td>[0, 1]</td>
</tr>
<tr>
  <td>Intrinsic Scale</td>
  <td style="color:red;">❌</td>
  <td style="color:green;">✅</td>
</tr>
<tr>
  <td>Descriptor Comparison</td>
  <td style="color:red;">❌</td>
  <td style="color:green;">✅</td>
</tr>
<tr>
  <td>Multi-Descriptor Aggregation</td>
  <td style="color:red;">❌</td>
  <td style="color:green;">✅</td>
</tr>
<tr>
  <td>Single Ranking</td>
  <td style="color:red;">❌</td>
  <td style="color:green;">✅</td>
</tr>
</tbody>
</table>

It also provides a number of other advantages over MMD which we discuss in our paper.

## Installation

```bash
pip install polygraph-benchmark
```

No manual compilation of ORCA is required. For details on interaction with `graph_tool`, see the more detailed installation instructions in the docs.

If you'd like to use SBM graph dataset validation with graph tools, use a mamba or pixi environment. More information is available in the documentation.

## At a glance

Here are a set of datasets and metrics this library provides:
- 🗂️ **Datasets**: ready-to-use splits for procedural and real-world graphs
  - Procedural datasets: `PlanarLGraphDataset`, `SBMLGraphDataset`, `LobsterLGraphDataset`
  - Real-world: `QM9`, `MOSES`, `Guacamol`, `DobsonDoigGraphDataset`, `ModelNet10GraphDataset`
  - Also: `EgoGraphDataset`, `PointCloudGraphDataset`
- 📊 **Metrics**: unified, fit-once/compute-many interface with convenience wrappers, avoiding redundant computations.
  - MMD<sup>2</sup>: `GaussianTVMMD2Benchmark`, `RBFMMD2Benchmark`
  - Kernel hyperparameter optimization with `MaxDescriptorMMD2`.
  - PolyGraphDiscrepancy: `StandardPGD`, `MolecularPGD` (for molecule descriptors).
  - Validation/Uniqueness/Novelty: `VUN`.
  - Uncertainty quantification for benchmarking (`GaussianTVMMD2BenchmarkInterval`, `RBFMMD2Benchmark`, `PGD5Interval`)
- 🧩 **Extendable**: Users can instantiate custom metrics by specifying descriptors, kernels, or classifiers (`PolyGraphDiscrepancy`, `DescriptorMMD2`). PolyGraph defines all necessary interfaces but imposes no requirements on the data type of graph objects.
- ⚙️ **Interoperability**: Works on Apple Silicon Macs and Linux.
- ✅ **Tested, type checked and documented**

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

## Tutorial

Our [demo script](polygraph_demo.py) showcases some features of our library in action.

### Datasets
Instantiate a benchmark dataset as follows:
```python
import networkx as nx
from polygraph.datasets import PlanarGraphDataset

reference = PlanarGraphDataset("test").to_nx()

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

#### PolyGraphDiscrepancy
Similarly, you can compute our proposed PolyGraphDiscrepancy, like so:

```python
from polygraph.metrics import StandardPGD

pgd = StandardPGD(reference)
print(pgd.compute(generated)) # {'pgd': ..., 'pgd_descriptor': ..., 'subscores': {'orbit': ..., }}
```

`pgd_descriptor` provides the best descriptor used to report the final score.

#### Validity, uniqueness and novelty
VUN values follow a similar interface:
```python
from polygraph.metrics import VUN
reference_ds = PlanarGraphDataset("test")
pgd = VUN(reference, validity_fn=reference_ds.is_valid, confidence_level=0.95) # if applicable, validity functions are defined as a dataset attribute
print(pgd.compute(generated))  # {'valid': ..., 'valid_unique_novel': ..., 'valid_novel': ..., 'valid_unique': ...}
```

#### Metric uncertainty quantification

For MMD and PGD, uncertainty quantifiation for the metrics are obtained through subsampling. For VUN, a confidence interval is obtained with a binomial test.

For `VUN`, the results can be obtained by specifying a confidence level when instantiating the metric.

For the others, the `Interval` suffix references the class that implements subsampling.

```python
from polygraph.metrics import GaussianTVMMD2BenchmarkInterval, RBFMMD2BenchmarkInterval, StandardPGDInterval
from tqdm import tqdm

metrics = [
  GaussianTVMMD2BenchmarkInterval(reference, subsample_size=8, num_samples=10), # specify size of each subsample, and the number of samples
  RBFMMD2BenchmarkInterval(reference, subsample_size=8, num_samples=10),
  StandardPGDInterval(reference, subsample_size=8, num_samples=10)
]

for metric in tqdm(metrics):
	metric_results = metric.compute(
    generated,
  )
```
