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

import json
import signal
from collections import defaultdict, namedtuple
from functools import partial
from multiprocessing import Pool
from typing import (
    Callable,
    Collection,
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


class _IsomorphismTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _IsomorphismTimeout()


def _is_isomorphic_with_timeout(
    g: nx.Graph,
    h: nx.Graph,
    timeout: int,
) -> bool:
    """Run isomorphism check with a SIGALRM timeout.

    If the check exceeds ``timeout`` seconds, conservatively returns True
    (i.e. treats the pair as isomorphic).
    """
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        result = nx.is_isomorphic(g, h)
    except _IsomorphismTimeout:
        result = True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return result


class _GraphSet:
    """Graph set with WL hash pre-filter and per-pair isomorphism timeout."""

    def __init__(
        self,
        nx_graphs: Optional[Iterable[nx.Graph]] = None,
        iso_timeout: int = 10,
    ):
        self.nx_graphs = [] if nx_graphs is None else list(nx_graphs)
        self._iso_timeout = iso_timeout
        self._hash_set: Dict[str, List[int]] = defaultdict(list)
        for idx, g in enumerate(self.nx_graphs):
            self._hash_set[nx.weisfeiler_lehman_graph_hash(g)].append(idx)

    def add(self, g: nx.Graph) -> None:
        self.nx_graphs.append(g)
        self._hash_set[nx.weisfeiler_lehman_graph_hash(g)].append(
            len(self.nx_graphs) - 1
        )

    def __contains__(self, g: nx.Graph) -> bool:
        fp = nx.weisfeiler_lehman_graph_hash(g)
        if fp not in self._hash_set:
            return False
        for idx in self._hash_set[fp]:
            if _is_isomorphic_with_timeout(
                g, self.nx_graphs[idx], self._iso_timeout
            ):
                return True
        return False


# ---------------------------------------------------------------------------
# Parallel worker functions (must be importable, not in __main__)
# ---------------------------------------------------------------------------


def _check_novel_worker(
    gen_graph_json: str,
    train_graphs_json: List[str],
    train_hashes: Dict[str, List[int]],
    iso_timeout: int,
) -> bool:
    """Check if a single generated graph is novel (not in training set)."""
    g = nx.node_link_graph(json.loads(gen_graph_json))
    fp = nx.weisfeiler_lehman_graph_hash(g)
    if fp not in train_hashes:
        return True
    for idx in train_hashes[fp]:
        h = nx.node_link_graph(json.loads(train_graphs_json[idx]))
        if _is_isomorphic_with_timeout(g, h, iso_timeout):
            return False
    return True


def _check_validity_worker(graph_json: str, validity_fn: Callable) -> bool:
    """Check validity of a single graph in a worker process."""
    g = nx.node_link_graph(json.loads(graph_json))
    return validity_fn(g)


class VUN(GenerationMetric[nx.Graph]):
    """Computes Valid-Unique-Novel metrics for generated graphs.

    Measures the fraction of generated graphs that are valid (optional), unique
    (not isomorphic to other generated graphs), and novel (not isomorphic to
    training graphs). Supports parallel novelty and validity checking via
    multiprocessing.

    Args:
        train_graphs: Collection of training graphs to check novelty against
        validity_fn: Optional function that takes a graph and returns `True` if valid.
            If `None`, only uniqueness and novelty are computed.
        confidence_level: Confidence level for binomial proportion intervals.
            If `None`, only point estimates are returned.
        iso_timeout: Per-pair isomorphism timeout in seconds. Pairs exceeding
            the timeout are conservatively treated as isomorphic.
        n_jobs: Number of parallel workers for novelty and validity checks.
            Use 1 for sequential execution.
    """

    def __init__(
        self,
        train_graphs: Iterable[nx.Graph],
        validity_fn: Optional[Callable] = None,
        confidence_level: Optional[float] = None,
        iso_timeout: int = 10,
        n_jobs: int = 1,
    ):
        self._train_graphs = list(train_graphs)
        self._train_set = _GraphSet(iso_timeout=iso_timeout)
        self._validity_fn = validity_fn
        self._confidence_level = confidence_level
        self._compute_ci = self._confidence_level is not None
        self._iso_timeout = iso_timeout
        self._n_jobs = n_jobs

        # Build WL hash index for the training set
        self._train_hashes: Dict[str, List[int]] = defaultdict(list)
        for idx, g in enumerate(self._train_graphs):
            h = nx.weisfeiler_lehman_graph_hash(g)
            self._train_set._hash_set[h].append(idx)
            self._train_hashes[h].append(idx)
        self._train_set.nx_graphs = self._train_graphs

    def _compute_novel_parallel(
        self, generated_graphs: List[nx.Graph]
    ) -> List[bool]:
        train_json = [
            json.dumps(nx.node_link_data(g)) for g in self._train_graphs
        ]
        gen_json = [json.dumps(nx.node_link_data(g)) for g in generated_graphs]
        worker_fn = partial(
            _check_novel_worker,
            train_graphs_json=train_json,
            train_hashes=dict(self._train_hashes),
            iso_timeout=self._iso_timeout,
        )
        with Pool(processes=self._n_jobs) as pool:
            return pool.map(worker_fn, gen_json, chunksize=64)

    def _compute_valid_parallel(
        self, generated_graphs: List[nx.Graph]
    ) -> List[bool]:
        gen_json = [json.dumps(nx.node_link_data(g)) for g in generated_graphs]
        assert self._validity_fn is not None
        worker_fn = partial(
            _check_validity_worker, validity_fn=self._validity_fn
        )
        with Pool(processes=self._n_jobs) as pool:
            return pool.map(worker_fn, gen_json, chunksize=32)

    def compute(
        self,
        generated_graphs: Collection[nx.Graph],
    ) -> Union[Dict[str, BinomConfidenceInterval], Dict[str, float]]:
        """Computes VUN metrics for a collection of generated graphs.

        Args:
            generated_graphs: Collection of networkx graphs to evaluate

        Returns:
            Dictionary containing metrics. If ``confidence_level`` was provided,
                values are ``BinomConfidenceInterval`` namedtuples.
                Otherwise values are plain floats.

        Raises:
            ValueError: If generated_samples is empty
        """
        generated_graphs = list(generated_graphs)
        n_graphs = len(generated_graphs)

        if n_graphs == 0:
            raise ValueError("Generated samples must not be empty")

        # Novelty: parallel when n_jobs > 1
        if self._n_jobs > 1:
            novel = self._compute_novel_parallel(generated_graphs)
        else:
            novel = [graph not in self._train_set for graph in generated_graphs]

        # Uniqueness: always sequential (inherently order-dependent)
        unique = []
        generated_set = _GraphSet(iso_timeout=self._iso_timeout)
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
            # Validity: parallel when n_jobs > 1
            if self._n_jobs > 1:
                valid = self._compute_valid_parallel(generated_graphs)
            else:
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
