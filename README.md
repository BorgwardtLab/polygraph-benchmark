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
## Example Benchmark

The following results mirror the tables from our paper. Bold indicates best, and underlined indicates second-best. Values are multiplied by 100 for legibility. Standard deviations are obtained with subsampling using `StandardPGDInterval` and `MoleculePGDInterval`. Specific parameters are discussed in the paper.

### Procedural and real-world graphs

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th rowspan="2" style="text-align:right;">Model</th>
      <th rowspan="2" style="text-align:right;">VUN (↑)</th>
      <th rowspan="2" style="text-align:right;">PGD (↓)</th>
      <th colspan="6" style="text-align:center;">PGD subscores</th>
    </tr>
    <tr>
      <th style="text-align:right;">Clust. (↓)</th>
      <th style="text-align:right;">Deg. (↓)</th>
      <th style="text-align:right;">GIN (↓)</th>
      <th style="text-align:right;">Orb5. (↓)</th>
      <th style="text-align:right;">Orb4. (↓)</th>
      <th style="text-align:right;">Eig. (↓)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Planar-L</td><td style="text-align:right;">AutoGraph</td><td style="text-align:right;"><u>85.1</u></td><td style="text-align:right;"><strong>34.0 ± 1.8</strong></td><td style="text-align:right;"><strong>7.0 ± 2.9</strong></td><td style="text-align:right;"><strong>7.8 ± 3.2</strong></td><td style="text-align:right;"><strong>8.8 ± 3.0</strong></td><td style="text-align:right;"><strong>34.0 ± 1.8</strong></td><td style="text-align:right;"><strong>28.5 ± 1.5</strong></td><td style="text-align:right;"><strong>26.9 ± 2.3</strong></td></tr>
    <tr><td></td><td style="text-align:right;">DiGress</td><td style="text-align:right;">80.1</td><td style="text-align:right;">45.2 ± 1.8</td><td style="text-align:right;">24.8 ± 2.0</td><td style="text-align:right;">23.3 ± 1.2</td><td style="text-align:right;"><u>29.0 ± 1.1</u></td><td style="text-align:right;">45.2 ± 1.8</td><td style="text-align:right;"><u>40.3 ± 1.8</u></td><td style="text-align:right;">39.4 ± 2.0</td></tr>
    <tr><td></td><td style="text-align:right;">GRAN</td><td style="text-align:right;">1.6</td><td style="text-align:right;">99.7 ± 0.2</td><td style="text-align:right;">99.3 ± 0.2</td><td style="text-align:right;">98.3 ± 0.3</td><td style="text-align:right;">98.3 ± 0.3</td><td style="text-align:right;">99.7 ± 0.1</td><td style="text-align:right;">99.2 ± 0.2</td><td style="text-align:right;">98.5 ± 0.4</td></tr>
    <tr><td></td><td style="text-align:right;">ESGG</td><td style="text-align:right;"><strong>93.9</strong></td><td style="text-align:right;"><u>45.0 ± 1.4</u></td><td style="text-align:right;"><u>10.9 ± 3.2</u></td><td style="text-align:right;"><u>21.7 ± 3.0</u></td><td style="text-align:right;">32.9 ± 2.2</td><td style="text-align:right;"><u>45.0 ± 1.4</u></td><td style="text-align:right;">42.8 ± 1.9</td><td style="text-align:right;"><u>29.6 ± 1.6</u></td></tr>
    <tr><td>Lobster-L</td><td style="text-align:right;">AutoGraph</td><td style="text-align:right;"><u>83.1</u></td><td style="text-align:right;"><u>18.0 ± 1.6</u></td><td style="text-align:right;">4.2 ± 1.9</td><td style="text-align:right;"><u>12.1 ± 1.6</u></td><td style="text-align:right;"><u>14.8 ± 1.5</u></td><td style="text-align:right;"><u>18.0 ± 1.6</u></td><td style="text-align:right;"><u>16.1 ± 1.6</u></td><td style="text-align:right;"><u>13.0 ± 1.1</u></td></tr>
    <tr><td></td><td style="text-align:right;">DiGress</td><td style="text-align:right;"><strong>91.4</strong></td><td style="text-align:right;"><strong>3.2 ± 2.6</strong></td><td style="text-align:right;"><u>2.0 ± 1.3</u></td><td style="text-align:right;"><strong>1.2 ± 1.5</strong></td><td style="text-align:right;"><strong>2.3 ± 2.0</strong></td><td style="text-align:right;"><strong>3.0 ± 3.1</strong></td><td style="text-align:right;"><strong>4.5 ± 2.3</strong></td><td style="text-align:right;"><strong>1.3 ± 1.1</strong></td></tr>
    <tr><td></td><td style="text-align:right;">GRAN</td><td style="text-align:right;">41.3</td><td style="text-align:right;">85.4 ± 0.5</td><td style="text-align:right;">20.8 ± 1.1</td><td style="text-align:right;">77.1 ± 1.2</td><td style="text-align:right;">79.8 ± 0.6</td><td style="text-align:right;">85.4 ± 0.5</td><td style="text-align:right;">85.0 ± 0.6</td><td style="text-align:right;">69.8 ± 1.2</td></tr>
    <tr><td></td><td style="text-align:right;">ESGG</td><td style="text-align:right;">70.9</td><td style="text-align:right;">69.9 ± 0.6</td><td style="text-align:right;"><strong>0.0 ± 0.0</strong></td><td style="text-align:right;">63.4 ± 1.1</td><td style="text-align:right;">66.8 ± 1.0</td><td style="text-align:right;">69.9 ± 0.6</td><td style="text-align:right;">66.0 ± 0.6</td><td style="text-align:right;">51.7 ± 1.8</td></tr>
    <tr><td>SBM-L</td><td style="text-align:right;">AutoGraph</td><td style="text-align:right;"><strong>85.6</strong></td><td style="text-align:right;"><strong>5.6 ± 1.5</strong></td><td style="text-align:right;"><strong>0.3 ± 0.6</strong></td><td style="text-align:right;"><strong>6.2 ± 1.4</strong></td><td style="text-align:right;"><strong>6.3 ± 1.3</strong></td><td style="text-align:right;"><strong>3.2 ± 2.2</strong></td><td style="text-align:right;"><strong>4.4 ± 2.0</strong></td><td style="text-align:right;"><strong>2.5 ± 2.2</strong></td></tr>
    <tr><td></td><td style="text-align:right;">DiGress</td><td style="text-align:right;"><u>73.0</u></td><td style="text-align:right;"><u>17.4 ± 2.3</u></td><td style="text-align:right;"><u>5.7 ± 2.8</u></td><td style="text-align:right;"><u>8.2 ± 3.3</u></td><td style="text-align:right;"><u>13.8 ± 1.7</u></td><td style="text-align:right;"><u>17.4 ± 2.3</u></td><td style="text-align:right;"><u>14.8 ± 2.5</u></td><td style="text-align:right;"><u>8.7 ± 3.0</u></td></tr>
    <tr><td></td><td style="text-align:right;">GRAN</td><td style="text-align:right;">21.4</td><td style="text-align:right;">69.1 ± 1.4</td><td style="text-align:right;">50.2 ± 1.9</td><td style="text-align:right;">58.6 ± 1.4</td><td style="text-align:right;">69.1 ± 1.4</td><td style="text-align:right;">65.7 ± 1.3</td><td style="text-align:right;">62.8 ± 1.3</td><td style="text-align:right;">55.9 ± 1.5</td></tr>
    <tr><td></td><td style="text-align:right;">ESGG</td><td style="text-align:right;">10.4</td><td style="text-align:right;">99.4 ± 0.2</td><td style="text-align:right;">97.9 ± 0.5</td><td style="text-align:right;">97.5 ± 0.6</td><td style="text-align:right;">98.3 ± 0.4</td><td style="text-align:right;">96.8 ± 0.4</td><td style="text-align:right;">89.2 ± 0.7</td><td style="text-align:right;">99.4 ± 0.2</td></tr>
    <tr><td>Proteins</td><td style="text-align:right;">AutoGraph</td><td style="text-align:right;">-</td><td style="text-align:right;"><strong>67.7 ± 7.4</strong></td><td style="text-align:right;"><u>47.7 ± 5.7</u></td><td style="text-align:right;"><u>31.5 ± 8.5</u></td><td style="text-align:right;"><u>45.3 ± 5.1</u></td><td style="text-align:right;"><strong>67.7 ± 7.4</strong></td><td style="text-align:right;"><strong>47.4 ± 7.0</strong></td><td style="text-align:right;">53.2 ± 6.9</td></tr>
    <tr><td></td><td style="text-align:right;">DiGress</td><td style="text-align:right;">-</td><td style="text-align:right;">88.1 ± 3.1</td><td style="text-align:right;"><strong>36.1 ± 4.3</strong></td><td style="text-align:right;"><strong>29.2 ± 5.0</strong></td><td style="text-align:right;"><strong>23.2 ± 5.3</strong></td><td style="text-align:right;">88.1 ± 3.1</td><td style="text-align:right;"><u>60.8 ± 3.6</u></td><td style="text-align:right;"><strong>23.4 ± 11.8</strong></td></tr>
    <tr><td></td><td style="text-align:right;">GRAN</td><td style="text-align:right;">-</td><td style="text-align:right;">89.7 ± 2.7</td><td style="text-align:right;">86.0 ± 2.0</td><td style="text-align:right;">70.6 ± 3.1</td><td style="text-align:right;">71.5 ± 3.0</td><td style="text-align:right;">90.4 ± 2.4</td><td style="text-align:right;">84.4 ± 3.3</td><td style="text-align:right;">76.7 ± 4.7</td></tr>
    <tr><td></td><td style="text-align:right;">ESGG</td><td style="text-align:right;">-</td><td style="text-align:right;"><u>79.2 ± 4.3</u></td><td style="text-align:right;">58.2 ± 3.6</td><td style="text-align:right;">54.0 ± 3.6</td><td style="text-align:right;">57.4 ± 4.1</td><td style="text-align:right;"><u>80.2 ± 3.1</u></td><td style="text-align:right;">72.5 ± 3.0</td><td style="text-align:right;"><u>24.3 ± 11.0</u></td></tr>
  </tbody>
  </table>

### Molecules
We provide here new benchmark values for molecules.

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th rowspan="2" style="text-align:right;">Model</th>
      <th rowspan="2" style="text-align:right;">Valid (↑)</th>
      <th rowspan="2" style="text-align:right;">PGD (↓)</th>
      <th colspan="5" style="text-align:center;">PGD subscores</th>
    </tr>
    <tr>
      <th style="text-align:right;">Topo (↓)</th>
      <th style="text-align:right;">Morgan (↓)</th>
      <th style="text-align:right;">ChemNet (↓)</th>
      <th style="text-align:right;">MolCLR (↓)</th>
      <th style="text-align:right;">Lipinski (↓)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Guacamol</td><td style="text-align:right;">AutoGraph</td><td style="text-align:right;"><u>91.6</u></td><td style="text-align:right;"><u>22.9 ± 0.5</u></td><td style="text-align:right;"><u>8.2 ± 0.7</u></td><td style="text-align:right;"><u>15.7 ± 0.8</u></td><td style="text-align:right;"><u>22.9 ± 0.5</u></td><td style="text-align:right;"><u>16.6 ± 0.4</u></td><td style="text-align:right;"><u>19.4 ± 0.7</u></td></tr>
    <tr><td></td><td style="text-align:right;">AutoGraph*</td><td style="text-align:right;"><strong>95.9</strong></td><td style="text-align:right;"><strong>10.4 ± 1.2</strong></td><td style="text-align:right;"><strong>4.3 ± 0.7</strong></td><td style="text-align:right;"><strong>4.7 ± 1.4</strong></td><td style="text-align:right;"><strong>4.6 ± 0.6</strong></td><td style="text-align:right;"><strong>1.7 ± 1.0</strong></td><td style="text-align:right;"><strong>10.4 ± 1.2</strong></td></tr>
    <tr><td></td><td style="text-align:right;">DiGress</td><td style="text-align:right;">85.2</td><td style="text-align:right;">32.7 ± 0.5</td><td style="text-align:right;">19.6 ± 0.6</td><td style="text-align:right;">20.4 ± 0.5</td><td style="text-align:right;">32.5 ± 0.7</td><td style="text-align:right;">22.9 ± 0.6</td><td style="text-align:right;">32.8 ± 0.5</td></tr>
    <tr><td>Moses</td><td style="text-align:right;">AutoGraph</td><td style="text-align:right;"><strong>87.4</strong></td><td style="text-align:right;"><strong>29.6 ± 0.4</strong></td><td style="text-align:right;"><strong>22.4 ± 0.4</strong></td><td style="text-align:right;"><strong>16.3 ± 1.3</strong></td><td style="text-align:right;"><strong>25.8 ± 0.7</strong></td><td style="text-align:right;"><strong>20.5 ± 0.5</strong></td><td style="text-align:right;"><strong>29.6 ± 0.4</strong></td></tr>
    <tr><td></td><td style="text-align:right;">DiGress</td><td style="text-align:right;"><u>85.7</u></td><td style="text-align:right;"><u>33.4 ± 0.5</u></td><td style="text-align:right;"><u>26.8 ± 0.4</u></td><td style="text-align:right;"><u>24.8 ± 0.8</u></td><td style="text-align:right;"><u>29.1 ± 0.6</u></td><td style="text-align:right;"><u>24.3 ± 0.7</u></td><td style="text-align:right;"><u>33.4 ± 0.5</u></td></tr>
  </tbody>
  </table>

<sub>* AutoGraph* denotes a variant that leverages additional training heuristics as described in the paper.</sub>
