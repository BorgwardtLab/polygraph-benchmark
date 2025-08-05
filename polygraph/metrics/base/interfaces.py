"""
PolyGraph implements metrics that provide either a single estimate or an interval to quantify uncertainty.
While the arguments to these metrics and their return types vary, we provide a minimal common interface here.

- [`GenerationMetric`][polygraph.metrics.base.interfaces.GenerationMetric] provides the interface for metrics that provide a single estimate.
- [`GenerationMetricInterval`][polygraph.metrics.base.interfaces.GenerationMetricInterval] provides the interface for metrics that provide an interval to quantify uncertainty.


Metrics that implement these interfaces may be evaluated jointly using the [`MetricCollection`][polygraph.metrics.base.interfaces.MetricCollection] and \
[`MetricIntervalCollection`][polygraph.metrics.base.interfaces.MetricIntervalCollection] classes.

```python
from polygraph.metrics import MetricCollection, MetricIntervalCollection
from polygraph.metrics.gran import RBFOrbitMMD2, ClassifierOrbitMetric
from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset

reference_graphs = PlanarGraphDataset("val").to_nx()
generated_graphs = SBMGraphDataset("test").to_nx()

metrics = MetricCollection(
    metrics={
        "rbf_orbit": RBFOrbitMMD2(reference_graphs=reference_graphs),
        "classifier_orbit": ClassifierOrbitMetric(reference_graphs=reference_graphs),
    }
)
print(metrics.compute(generated_graphs))
```
"""
from typing import Protocol, Collection, Any, Dict, Optional
import networkx as nx


class GenerationMetric(Protocol):
    """Interface for metrics that provide a single estimate."""
    def compute(self, generated_graphs: Collection[nx.Graph], **kwargs: Any) -> Any:
        """Compute the metric on the generated graphs.

        Args:
            generated_graphs: Collection of generated graphs.
            kwargs: Additional keyword arguments to pass to the metric.
        """
        ...


class GenerationMetricInterval(Protocol):
    """Interface for metrics that provide an interval to quantify uncertainty."""
    def compute(
        self, 
        generated_graphs: Collection[nx.Graph], 
        subsample_size: int, 
        num_samples: int = 100, 
        **kwargs: Any
    ) -> Any:
        """Compute the metric on the generated graphs.

        Args:
            generated_graphs: Collection of generated graphs.
            subsample_size: Size of the subsample to use for each point estimate.
            num_samples: Number of samples to use for quantifying uncertainty.
            kwargs: Additional keyword arguments to pass to the metric.
        """
        ...


class MetricCollection(GenerationMetric):
    """Collection of metrics that provide a single estimate."""
    def __init__(self, metrics: Dict[str, GenerationMetric]):
        self._metrics = metrics
    
    def compute(
        self, 
        generated_graphs: Collection[nx.Graph], 
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute the metrics on the generated graphs.

        Args:
            generated_graphs: Collection of generated graphs.
            kwargs: Dictionary mapping metric names to keyword arguments to pass to the metrics.
                By default, no additional arguments are passed to the metrics.
        """
        kwargs = kwargs or {}
        return {
            name: metric.compute(generated_graphs, **kwargs.get(name, {})) 
            for name, metric in self._metrics.items()
        }
    

class MetricIntervalCollection(GenerationMetricInterval):
    """Collection of metrics that provide an interval to quantify uncertainty."""
    def __init__(self, metrics: Dict[str, GenerationMetricInterval]):
        self._metrics = metrics
    
    def compute(
        self, 
        generated_graphs: Collection[nx.Graph], 
        subsample_size: int, 
        num_samples: int = 100, 
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute the metrics on the generated graphs.

        Args:
            generated_graphs: Collection of generated graphs.
            subsample_size: Size of the subsample to use for each point estimate.
            num_samples: Number of samples to use for quantifying uncertainty.
            kwargs: Dictionary mapping metric names to keyword arguments to pass to the metrics.
                By default, no additional arguments are passed to the metrics.
        """
        kwargs = kwargs or {}
        return {
            name: metric.compute(
                generated_graphs, subsample_size, num_samples, **kwargs.get(name, {})
            ) 
            for name, metric in self._metrics.items()
        }