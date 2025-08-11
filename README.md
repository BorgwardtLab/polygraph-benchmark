# PolyGraph

PolyGraph is a Python library for evaluating graph generative models.

## Installation

```bash
pip install polygraph-benchmark
```

No manual compilation of ORCA is required. For details on interaction with `graph_tool`, see the more detailed installation instructions in the docs.

## Quickstart

```python
import networkx as nx
from polygraph.datasets import PlanarGraphDataset
from polygraph.metrics import GaussianTVMMD2Benchmark

reference = PlanarGraphDataset("test").to_nx()
benchmark = GaussianTVMMD2Benchmark(reference)

generated = [nx.erdos_renyi_graph(64, 0.1) for _ in range(40)]
print(benchmark.compute(generated))  # {'orbit': ..., 'clustering': ..., 'degree': ..., 'spectral': ...}
```
