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

## Benchmarking snapshot

Here is an example benchmark one can generate with this library using multiple different models and datasets. The details of the generation of this benchmark are given in our paper.

<table>
<thead>
<tr>
  <th rowspan="2">Dataset</th>
  <th rowspan="2">Model</th>
  <th rowspan="2">VUN (↑)</th>
  <th rowspan="2">PGD (↓)</th>
  <th colspan="6">PGD subscores</th>
</tr>
<tr>
  <th>Clust. (↓)</th>
  <th>Deg. (↓)</th>
  <th>GIN (↓)</th>
  <th>Orb5. (↓)</th>
  <th>Orb4. (↓)</th>
  <th>Eig. (↓)</th>
</tr>
</thead>
<tbody>

<!-- Planar-L -->
<tr><td rowspan="4"><b>Planar-L</b></td>
  <td>AutoGraph</td>
  <td><i>85.1</i></td>
  <td><b>34.0 ± 1.8</b></td>
  <td><b>7.0 ± 2.9</b></td>
  <td><b>7.8 ± 3.2</b></td>
  <td><b>8.8 ± 3.0</b></td>
  <td><b>34.0 ± 1.8</b></td>
  <td><b>28.5 ± 1.5</b></td>
  <td><b>26.9 ± 2.3</b></td></tr>
<tr><td>DiGress</td>
  <td>80.1</td>
  <td>45.2 ± 1.8</td>
  <td>24.8 ± 2.0</td>
  <td>23.3 ± 1.2</td>
  <td><i>29.0 ± 1.1</i></td>
  <td>45.2 ± 1.8</td>
  <td><i>40.3 ± 1.8</i></td>
  <td>39.4 ± 2.0</td></tr>
<tr><td>GRAN</td>
  <td>1.6</td>
  <td>99.7 ± 0.2</td>
  <td>99.3 ± 0.2</td>
  <td>98.3 ± 0.3</td>
  <td>98.3 ± 0.3</td>
  <td>99.7 ± 0.1</td>
  <td>99.2 ± 0.2</td>
  <td>98.5 ± 0.4</td></tr>
<tr><td>ESGG</td>
  <td><b>93.9</b></td>
  <td><i>45.0 ± 1.4</i></td>
  <td><i>10.9 ± 3.2</i></td>
  <td><i>21.7 ± 3.0</i></td>
  <td>32.9 ± 2.2</td>
  <td><i>45.0 ± 1.4</i></td>
  <td>42.8 ± 1.9</td>
  <td><i>29.6 ± 1.6</i></td></tr>

<!-- Lobster-L -->
<tr><td rowspan="4"><b>Lobster-L</b></td>
  <td>AutoGraph</td>
  <td><i>83.1</i></td>
  <td><i>18.0 ± 1.6</i></td>
  <td>4.2 ± 1.9</td>
  <td><i>12.1 ± 1.6</i></td>
  <td><i>14.8 ± 1.5</i></td>
  <td><i>18.0 ± 1.6</i></td>
  <td><i>16.1 ± 1.6</i></td>
  <td><i>13.0 ± 1.1</i></td></tr>
<tr><td>DiGress</td>
  <td><b>91.4</b></td>
  <td><b>3.2 ± 2.6</b></td>
  <td><i>2.0 ± 1.3</i></td>
  <td><b>1.2 ± 1.5</b></td>
  <td><b>2.3 ± 2.0</b></td>
  <td><b>3.0 ± 3.1</b></td>
  <td><b>4.5 ± 2.3</b></td>
  <td><b>1.3 ± 1.1</b></td></tr>
<tr><td>GRAN</td>
  <td>41.3</td>
  <td>85.4 ± 0.5</td>
  <td>20.8 ± 1.1</td>
  <td>77.1 ± 1.2</td>
  <td>79.8 ± 0.6</td>
  <td>85.4 ± 0.5</td>
  <td>85.0 ± 0.6</td>
  <td>69.8 ± 1.2</td></tr>
<tr><td>ESGG</td>
  <td>70.9</td>
  <td>69.9 ± 0.6</td>
  <td><b>0.0 ± 0.0</b></td>
  <td>63.4 ± 1.1</td>
  <td>66.8 ± 1.0</td>
  <td>69.9 ± 0.6</td>
  <td>66.0 ± 0.6</td>
  <td>51.7 ± 1.8</td></tr>

<!-- SBM-L -->
<tr><td rowspan="4"><b>SBM-L</b></td>
  <td>AutoGraph</td>
  <td><b>85.6</b></td>
  <td><b>5.6 ± 1.5</b></td>
  <td><b>0.3 ± 0.6</b></td>
  <td><b>6.2 ± 1.4</b></td>
  <td><b>6.3 ± 1.3</b></td>
  <td><b>3.2 ± 2.2</b></td>
  <td><b>4.4 ± 2.0</b></td>
  <td><b>2.5 ± 2.2</b></td></tr>
<tr><td>DiGress</td>
  <td><i>73.0</i></td>
  <td><i>17.4 ± 2.3</i></td>
  <td><i>5.7 ± 2.8</i></td>
  <td><i>8.2 ± 3.3</i></td>
  <td><i>13.8 ± 1.7</i></td>
  <td><i>17.4 ± 2.3</i></td>
  <td><i>14.8 ± 2.5</i></td>
  <td><i>8.7 ± 3.0</i></td></tr>
<tr><td>GRAN</td>
  <td>21.4</td>
  <td>69.1 ± 1.4</td>
  <td>50.2 ± 1.9</td>
  <td>58.6 ± 1.4</td>
  <td>69.1 ± 1.4</td>
  <td>65.7 ± 1.3</td>
  <td>62.8 ± 1.3</td>
  <td>55.9 ± 1.5</td></tr>
<tr><td>ESGG</td>
  <td>10.4</td>
  <td>99.4 ± 0.2</td>
  <td>97.9 ± 0.5</td>
  <td>97.5 ± 0.6</td>
  <td>98.3 ± 0.4</td>
  <td>96.8 ± 0.4</td>
  <td>89.2 ± 0.7</td>
  <td>99.4 ± 0.2</td></tr>

<!-- Proteins -->
<tr><td rowspan="4"><b>Proteins</b></td>
  <td>AutoGraph</td>
  <td>–</td>
  <td><b>67.7 ± 7.4</b></td>
  <td><i>47.7 ± 5.7</i></td>
  <td><i>31.5 ± 8.5</i></td>
  <td><i>45.3 ± 5.1</i></td>
  <td><b>67.7 ± 7.4</b></td>
  <td><b>47.4 ± 7.0</b></td>
  <td>53.2 ± 6.9</td></tr>
<tr><td>DiGress</td>
  <td>–</td>
  <td>88.1 ± 3.1</td>
  <td><b>36.1 ± 4.3</b></td>
  <td><b>29.2 ± 5.0</b></td>
  <td><b>23.2 ± 5.3</b></td>
  <td>88.1 ± 3.1</td>
  <td><i>60.8 ± 3.6</i></td>
  <td><b>23.4 ± 11.8</b></td></tr>
<tr><td>GRAN</td>
  <td>–</td>
  <td>89.7 ± 2.7</td>
  <td>86.0 ± 2.0</td>
  <td>70.6 ± 3.1</td>
  <td>71.5 ± 3.0</td>
  <td>90.4 ± 2.4</td>
  <td>84.4 ± 3.3</td>
  <td>76.7 ± 4.7</td></tr>
<tr><td>ESGG</td>
  <td>–</td>
  <td><i>79.2 ± 4.3</i></td>
  <td>58.2 ± 3.6</td>
  <td>54.0 ± 3.6</td>
  <td>57.4 ± 4.1</td>
  <td><i>80.2 ± 3.1</i></td>
  <td>72.5 ± 3.0</td>
  <td><i>24.3 ± 11.0</i></td></tr>

</tbody>
</table>

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
