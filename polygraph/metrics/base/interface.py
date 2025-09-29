"""
PolyGraph implements metrics that provide either a single estimate or an interval to quantify uncertainty.
We provide a minimal common interface for metrics as the protocol [`GenerationMetric`][polygraph.metrics.base.GenerationMetric].
The only requirement to satisfy this interface is to implement a `compute` method that accepts a collection of graphs.
In practice, these graphs may either be `nx.Graph` or `rdkit.Chem.Mol` objects, as determined by the `GraphType` generic parameter.

Metrics that implement this interface may be evaluated jointly using the [`MetricCollection`][polygraph.metrics.base.MetricCollection] class.

```python
from polygraph.metrics import MetricCollection, StandardPGD
from polygraph.metrics.rbf_mmd import RBFOrbitMMD2
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset

reference_graphs = PlanarGraphDataset("val").to_nx()
generated_graphs = SBMGraphDataset("val").to_nx()

metrics = MetricCollection(
    metrics={
        "rbf_orbit": RBFOrbitMMD2(reference_graphs=reference_graphs),
        "pgd": StandardPGD(reference_graphs=reference_graphs),
    }
)
print(metrics.compute(generated_graphs))
```
"""

from polygraph import GraphType

from typing import Protocol, Collection, Any, Dict, Generic


class GenerationMetric(Protocol, Generic[GraphType]):
    """Interface for all graph generation metrics."""

    def compute(self, generated_graphs: Collection[GraphType]) -> Any:
        """Compute the metric on the generated graphs.

        Args:
            generated_graphs: Collection of generated graphs to evaluate.
        """
        ...


class MetricCollection(GenerationMetric[GraphType], Generic[GraphType]):
    """Collection of metrics that are evaluated jointly."""

    def __init__(self, metrics: Dict[str, GenerationMetric[GraphType]]):
        self._metrics = metrics

    def compute(
        self,
        generated_graphs: Collection[GraphType],
    ) -> Dict[str, Any]:
        """Compute the metrics on the generated graphs.

        Args:
            generated_graphs: Collection of generated graphs.
        """
        return {
            name: metric.compute(generated_graphs)
            for name, metric in self._metrics.items()
        }
