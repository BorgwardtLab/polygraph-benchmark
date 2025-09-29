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
