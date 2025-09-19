"""Metrics mainly used for evaluating graph generative models on synthetic data.

This module provides [`VUN`][polygraph.metrics.VUN], a class for computing the Valid-Unique-Novel (VUN) metrics, which
measure what fraction of generated graphs are:

- Valid: Satisfy domain-specific constraints
- Unique: Not isomorphic to other generated graphs
- Novel: Not isomorphic to training graphs

By passing `confidence_level` to the constructor, you may also compute Binomial confidence intervals for the proportions.

Example:
    ```python
    from polygraph.datasets import PlanarGraphDataset, SBMGraphDataset
    from polygraph.metrics import VUN

    train = PlanarGraphDataset("val")
    generated = SBMGraphDataset("val")

    # Without uncertainty quantification
    vun = VUN(train.to_nx(), validity_fn=train.is_valid)
    print(vun.compute(generated.to_nx()))           # {'unique': 1.0, 'novel': 1.0, 'unique_novel': 1.0, 'valid': 0.0, 'valid_unique_novel': 0.0, 'valid_novel': 0.0, 'valid_unique': 0.0}

    # With uncertainty quantification
    vun = VUN(train.to_nx(), validity_fn=train.is_valid, confidence_level=0.95)
    print(vun.compute(generated.to_nx()))           # {'unique': ConfidenceInterval(mle=1.0, low=None, high=None), 'novel': ConfidenceInterval(mle=1.0, low=0.8911188393205571, high=1.0), 'unique_novel': ConfidenceInterval(mle=1.0, low=None, high=None), 'valid': ConfidenceInterval(mle=0.0, low=0.0, high=0.10888116067944287), 'valid_unique_novel': ConfidenceInterval(mle=0.0, low=None, high=None), 'valid_novel': ConfidenceInterval(mle=0.0, low=0.0, high=0.10888116067944287), 'valid_unique': ConfidenceInterval(mle=0.0, low=None, high=None
    ```
"""

from collections import defaultdict, namedtuple
from typing import (
    Any,
    Callable,
    Collection,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

import networkx as nx
from scipy.stats import binomtest

from polygraph.metrics.base.interface import GenerationMetric

__all__ = ["VUN"]

BinomConfidenceInterval = namedtuple(
    "ConfidenceInterval", ["mle", "low", "high"]
)


class _GraphSet:
    """Internal class for efficiently checking graph isomorphism.

    Uses Weisfeiler-Lehman hashing as a fast pre-filter before running exact
    isomorphism checks. Supports checking isomorphism with node and edge attributes.

    Args:
        nx_graphs: Initial collection of graphs
        node_attrs: Node attributes to consider for isomorphism
        edge_attrs: Edge attributes to consider for isomorphism
    """

    def __init__(
        self,
        nx_graphs: Optional[Iterable[nx.Graph]] = None,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
    ):
        self.nx_graphs = [] if nx_graphs is None else list(nx_graphs)
        self._node_attrs = node_attrs if node_attrs is not None else []
        self._edge_attrs = edge_attrs if edge_attrs is not None else []
        self._hash_set = self._compute_hash_set(self.nx_graphs)

    def add_from(self, graph_iter: Iterable[nx.Graph]) -> None:
        """Adds multiple graphs to the set."""
        for g in graph_iter:
            self.add(g)

    def add(self, g: nx.Graph) -> None:
        """Adds a single graph to the set."""
        self.nx_graphs.append(g)
        self._hash_set[self._graph_fingerprint(g)].append(
            len(self.nx_graphs) - 1
        )

    def __contains__(self, g: nx.Graph) -> bool:
        """Checks if a graph is isomorphic to any graph in the set."""
        fingerprint = self._graph_fingerprint(g)
        if fingerprint not in self._hash_set:
            return False
        potentially_isomorphic = [
            self.nx_graphs[idx] for idx in self._hash_set[fingerprint]
        ]
        for h in potentially_isomorphic:
            if nx.is_isomorphic(
                g,
                h,
                node_match=self._node_match
                if len(self._node_attrs) > 0
                else None,
                edge_match=self._edge_match
                if len(self._edge_attrs) > 0
                else None,
            ):
                return True
        return False

    def __add__(self, other: "_GraphSet") -> "_GraphSet":
        return _GraphSet(self.nx_graphs + other.nx_graphs)

    def _node_match(self, n1: Dict[str, Any], n2: Dict[str, Any]) -> bool:
        return all(n1[key] == n2[key] for key in self._node_attrs)

    def _edge_match(self, e1: Dict[str, Any], e2: Dict[str, Any]) -> bool:
        return all(e1[key] == e2[key] for key in self._edge_attrs)

    def _graph_fingerprint(self, g: nx.Graph) -> str:
        return nx.weisfeiler_lehman_graph_hash(
            g,
            edge_attr=self._edge_attrs[0]
            if len(self._edge_attrs) > 0
            else None,
            node_attr=self._node_attrs[0]
            if len(self._node_attrs) > 0
            else None,
        )

    def _compute_hash_set(
        self, nx_graphs: List[nx.Graph]
    ) -> DefaultDict[str, List[int]]:
        hash_set = defaultdict(list)
        for idx, g in enumerate(nx_graphs):
            hash_set[self._graph_fingerprint(g)].append(idx)
        return hash_set


class VUN(GenerationMetric[nx.Graph]):
    """Computes Valid-Unique-Novel metrics for generated graphs.

    Measures the fraction of generated graphs that are valid (optional), unique
    (not isomorphic to other generated graphs), and novel (not isomorphic to
    training graphs). Also computes confidence intervals for these proportions.

    Args:
        train_graphs: Collection of training graphs to check novelty against
        validity_fn: Optional function that takes a graph and returns `True` if the given graph is valid.
            If `None`, only uniqueness and novelty are computed.
        confidence_level: Confidence level for binomial proportion intervals. If `None`, only the point estimates are returned.
    """

    def __init__(
        self,
        train_graphs: Iterable[nx.Graph],
        validity_fn: Optional[Callable] = None,
        confidence_level: Optional[float] = None,
    ):
        self._train_set = _GraphSet()
        self._train_set.add_from(train_graphs)
        self._validity_fn = validity_fn
        self._confidence_level = confidence_level
        self._compute_ci = self._confidence_level is not None

    def compute(
        self,
        generated_graphs: Collection[nx.Graph],
    ) -> Union[Dict[str, BinomConfidenceInterval], Dict[str, float]]:
        """Computes VUN metrics for a collection of generated graphs.

        Args:
            generated_graphs: Collection of networkx graphs to evaluate

        Returns:
            Dictionary containing metrics. If `confidence_level` was provided, it contains
                confidence intervals as tuples (estimate, lower bound, upper bound).
                Otherwise returns only the point estimates.
        Raises:
            ValueError: If generated_samples is empty
        """
        n_graphs = len(generated_graphs)

        if n_graphs == 0:
            raise ValueError("Generated samples must not be empty")

        novel = [graph not in self._train_set for graph in generated_graphs]

        unique = []
        generated_set = _GraphSet()
        for graph in generated_graphs:
            unique.append(graph not in generated_set)
            generated_set.add(graph)

        unique_novel = [u and n for u, n in zip(unique, novel)]

        result = {
            "unique": sum(unique),
            "novel": sum(novel),
            "unique_novel": sum(unique_novel),
        }

        if self._validity_fn is not None:
            valid = [self._validity_fn(graph) for graph in generated_graphs]
            unique_novel_valid = [
                un and v for un, v in zip(unique_novel, valid)
            ]
            valid_novel = [v and n for v, n in zip(valid, novel)]
            valid_unique = [v and u for v, u in zip(valid, unique)]
            result.update(
                {
                    "valid": sum(valid),
                    "valid_unique_novel": sum(unique_novel_valid),
                    "valid_novel": sum(valid_novel),
                    "valid_unique": sum(valid_unique),
                }
            )

        if self._compute_ci:
            assert self._confidence_level is not None
            result_w_ci = {}
            for key, val in result.items():
                if "unique" not in key:
                    interval = binomtest(k=val, n=n_graphs).proportion_ci(
                        confidence_level=self._confidence_level
                    )
                    low, high = interval.low, interval.high
                else:
                    low, high = None, None
                result_w_ci[key] = BinomConfidenceInterval(
                    mle=val / n_graphs, low=low, high=high
                )
            return result_w_ci
        else:
            return {key: val / n_graphs for key, val in result.items()}
