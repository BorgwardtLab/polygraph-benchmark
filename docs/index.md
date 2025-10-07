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

The table below shows an example benchmark generated with this library across multiple datasets and models. Values illustrate typical outputs from the newly proposed PolyGraph Discrepancy. For completeness, this library and our paper also implements and provides various MMD estimates on the datasets below. Values are scaled by 100 for legibility and subsampling is used to obtain standard deviations (using `StandardPGDInterval` and `MoleculePGDInterval`). More details are provided in our paper.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th style="text-align:right;">Planar-L</th>
      <th style="text-align:right;">Lobster-L</th>
      <th style="text-align:right;">SBM-L</th>
      <th style="text-align:right;">Proteins</th>
      <th style="text-align:right;">Guacamol</th>
      <th style="text-align:right;">Moses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AutoGraph</td>
      <td style="text-align:right;"><strong>34.0 ± 1.8</strong></td>
      <td style="text-align:right;"><u>18.0 ± 1.6</u></td>
      <td style="text-align:right;"><strong>5.6 ± 1.5</strong></td>
      <td style="text-align:right;"><strong>67.7 ± 7.4</strong></td>
      <td style="text-align:right;"><u>22.9 ± 0.5</u></td>
      <td style="text-align:right;"><strong>29.6 ± 0.4</strong></td>
    </tr>
    <tr>
      <td>AutoGraph*</td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;"><strong>10.4 ± 1.2</strong></td>
      <td style="text-align:right;">—</td>
    </tr>
    <tr>
      <td>DiGress</td>
      <td style="text-align:right;">45.2 ± 1.8</td>
      <td style="text-align:right;"><strong>3.2 ± 2.6</strong></td>
      <td style="text-align:right;"><u>17.4 ± 2.3</u></td>
      <td style="text-align:right;">88.1 ± 3.1</td>
      <td style="text-align:right;">32.7 ± 0.5</td>
      <td style="text-align:right;"><u>33.4 ± 0.5</u></td>
    </tr>
    <tr>
      <td>GRAN</td>
      <td style="text-align:right;">99.7 ± 0.2</td>
      <td style="text-align:right;">85.4 ± 0.5</td>
      <td style="text-align:right;">69.1 ± 1.4</td>
      <td style="text-align:right;">89.7 ± 2.7</td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;">—</td>
    </tr>
    <tr>
      <td>ESGG</td>
      <td style="text-align:right;"><u>45.0 ± 1.4</u></td>
      <td style="text-align:right;">69.9 ± 0.6</td>
      <td style="text-align:right;">99.4 ± 0.2</td>
      <td style="text-align:right;"><u>79.2 ± 4.3</u></td>
      <td style="text-align:right;">—</td>
      <td style="text-align:right;">—</td>
    </tr>
  </tbody>
  </table>

<sub>* AutoGraph* denotes a variant that leverages additional training heuristics as described in the paper.</sub>
