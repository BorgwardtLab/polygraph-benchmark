# PolyGraph

PolyGraph is a Python library for evaluating graph generative
models.
With `polygraph`, evaluating a generative model becomes as easy as this:

```python
import networkx as nx
from polygraph.datasets import PlanarGraphDataset
from polygraph.metrics import GaussianTVMMD2Benchmark

reference = PlanarGraphDataset("test").to_nx()
benchmark = GaussianTVMMD2Benchmark(reference)

generated = [nx.erdos_renyi_graph(64, 0.1) for _ in range(40)]
print(benchmark.compute(generated))     # {'orbit': 1.3305546735190608, 'clustering': 0.2799915534527712, 'degree': 0.07563928348299709, 'spectral': 0.07841922146118052}
```

## Installation

You may install this package via:
```
pip install polygraph-benchmark
```
No manual compilation of [ORCA](https://github.com/thocevar/orca) is required.
For details on the interaction with the `graph_tool` package, see the more detailed [installation instructions](installation.md).

## Usage

We provide a few basic tutorials:

- [Basic Usage](tutorials/basic_usage.md) - How to load datasets and compute metrics
- [Metrics Overview](tutorials/metrics_overview.md) - An overview of which metrics are implemented in `polygraph` (MMD, PGD, VUN, Frechet Distance)
- [Custom Datasets](tutorials/custom_datasets.md) - How to build custom datasets and share them

### PGD vs MMD overview

PolyGraph Discrepancy (PGD) is our proposed metric for graph generative model evaluation. Compared to maximum mean discrepancy (MMD), PGD provides a bounded range, an intrinsic scale, and a principled way to compare and aggregate across descriptors.

<style>
table {
  font-size: 90%;
  margin: 0 auto;
}
th, td {
  text-align: center;
  padding: 4px 8px;
}
th:first-child, td:first-child {
  text-align: left;
}
</style>

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

PGD and its motivation are described in more detail in the paper and API docs.

### Benchmarking snapshot

The table below shows an example benchmark generated with this library across multiple datasets and models. Values illustrate typical outputs from the implemented metrics (VUN, PGD, and PGD subscores) and are for demonstration and reproduction of results discussed in our paper.

<style>
table {
  font-size: 85%;
  border-collapse: collapse;
}
th, td {
  text-align: center;
  padding: 4px 6px;
  border: 1px solid #ddd;
}
th:first-child, td:first-child {
  text-align: left;
}
th:nth-child(2), td:nth-child(2) {
  text-align: left;
}
</style>

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
  <td><i>14.8 ± 1.5</td>
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
  <td><b>6.3 ± 1.3</td>
  <td><b>3.2 ± 2.2</td>
  <td><b>4.4 ± 2.0</td>
  <td><b>2.5 ± 2.2</td></tr>
<tr><td>DiGress</td>
  <td><i>73.0</i></td>
  <td><i>17.4 ± 2.3</td>
  <td><i>5.7 ± 2.8</td>
  <td><i>8.2 ± 3.3</td>
  <td><i>13.8 ± 1.7</td>
  <td><i>17.4 ± 2.3</td>
  <td><i>14.8 ± 2.5</td>
  <td><i>8.7 ± 3.0</td></tr>
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
  <td><i>31.5 ± 8.5</td>
  <td><i>45.3 ± 5.1</td>
  <td><b>67.7 ± 7.4</td>
  <td><b>47.4 ± 7.0</td>
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
  <td><i>79.2 ± 4.3</td>
  <td>58.2 ± 3.6</td>
  <td>54.0 ± 3.6</td>
  <td>57.4 ± 4.1</td>
  <td><i>80.2 ± 3.1</td>
  <td>72.5 ± 3.0</td>
  <td><i>24.3 ± 11.0</td></tr>

</tbody>
</table>
