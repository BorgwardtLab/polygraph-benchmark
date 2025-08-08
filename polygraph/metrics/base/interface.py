"""
PolyGraph implements metrics that provide either a single estimate or an interval to quantify uncertainty.
We provide a minimal common interface for metrics as the protocol [`GenerationMetric`][polygraph.metrics.base.GenerationMetric].
The only requirement to satisfy this interface is to implement a `compute` method that accepts a collection of NetworkX graphs.

Metrics that implement this interface may be evaluated jointly using the [`MetricCollection`][polygraph.metrics.base.MetricCollection] class.

```python
from polygraph.metrics import MetricCollection
from polygraph.metrics.rbf_mmd import RBFOrbitMMD2
from polygraph.metrics.polygraphscore import PGS5
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset

reference_graphs = PlanarGraphDataset("val").to_nx()
generated_graphs = SBMGraphDataset("test").to_nx()

metrics = MetricCollection(
    metrics={
        "rbf_orbit": RBFOrbitMMD2(reference_graphs=reference_graphs),
        "pgs5": PGS5(reference_graphs=reference_graphs),
    }
)
print(metrics.compute(generated_graphs))
```
"""

from typing import Protocol, Collection, Any, Dict
import networkx as nx


class GenerationMetric(Protocol):
    """Interface for metrics that provide a single estimate."""

    def compute(self, generated_graphs: Collection[nx.Graph]) -> Any:
        """Compute the metric on the generated graphs.

        Args:
            generated_graphs: Collection of generated graphs to evaluate.
        """
        ...


class MetricCollection(GenerationMetric):
    """Collection of metrics that provide a single estimate."""

    def __init__(self, metrics: Dict[str, GenerationMetric]):
        self._metrics = metrics

    def compute(
        self,
        generated_graphs: Collection[nx.Graph],
    ) -> Dict[str, Any]:
        """Compute the metrics on the generated graphs.

        Args:
            generated_graphs: Collection of generated graphs.
        """
        return {
            name: metric.compute(generated_graphs)
            for name, metric in self._metrics.items()
        }
