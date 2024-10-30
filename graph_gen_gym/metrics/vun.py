from collections import defaultdict, namedtuple
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional

import joblib
import networkx as nx
from scipy.stats import binomtest

from graph_gen_gym.datasets.abstract_dataset import AbstractDataset


class _GraphSet:
    def __init__(self, nx_graphs: Optional[Iterable[nx.Graph]] = None):
        self.nx_graphs = [] if nx_graphs is None else list(nx_graphs)
        self._hash_set = self._compute_hash_set(self.nx_graphs)

    def add_from(self, graph_iter: Iterable[nx.Graph]) -> None:
        for g in graph_iter:
            self.add(g)

    def add(self, g: nx.Graph) -> None:
        self.nx_graphs.append(g)
        self._hash_set[_GraphSet._graph_fingerprint(g)].append(len(self.nx_graphs) - 1)

    def __contains__(self, g: nx.Graph) -> bool:
        fingerprint = self._graph_fingerprint(g)
        if fingerprint not in self._hash_set:
            return False
        potentially_isomorphic = [
            self.nx_graphs[idx] for idx in self._hash_set[fingerprint]
        ]
        for h in potentially_isomorphic:
            if nx.is_isomorphic(g, h):
                return True
        return False

    def __add__(self, other: "_GraphSet") -> "_GraphSet":
        return _GraphSet(self.nx_graphs + other.nx_graphs)

    @staticmethod
    def _graph_fingerprint(g: nx.Graph) -> str:
        nodes = list(g.nodes)
        triangle_counts = nx.triangles(g, nodes)
        degrees = [item[1] for item in g.degree(nodes)]
        fingerprint = tuple(
            sorted(
                [
                    joblib.hash((deg, triangle))
                    for deg, triangle in zip(degrees, triangle_counts)
                ]
            )
        )
        return fingerprint

    @staticmethod
    def _compute_hash_set(nx_graphs: List[nx.Graph]) -> DefaultDict[str, List[int]]:
        hash_set = defaultdict(list)
        for idx, g in enumerate(nx_graphs):
            hash_set[_GraphSet._graph_fingerprint(g)].append(idx)
        return hash_set


BinomConfidenceInterval = namedtuple("ConfidenceInterval", ["mle", "low", "high"])


class VUN:
    def __init__(
        self, train_graphs: AbstractDataset, validity_fn: Optional[Callable] = None
    ):
        self._train_set = _GraphSet()
        self._train_set.add_from(train_graphs.to_nx())
        self._validity_fn = validity_fn

    def compute(
        self, generated_samples: Iterable[nx.Graph], confidence_level: float = 0.95
    ) -> Dict[str, BinomConfidenceInterval]:
        n_graphs = len(generated_samples)
        novel = [graph not in self._train_set for graph in generated_samples]
        unique = []
        generated_set = _GraphSet()
        for graph in generated_samples:
            unique.append(graph not in generated_set)
            generated_set.add(graph)
        unique_novel = [u and n for u, n in zip(unique, novel)]

        result = {
            "unique": sum(unique),
            "novel": sum(novel),
            "unique_novel": sum(unique_novel),
        }

        if self._validity_fn is not None:
            valid = [self._validity_fn(graph) for graph in generated_samples]
            unique_novel_valid = [un and v for un, v in zip(unique_novel, valid)]
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

        result_w_ci = {}
        for key, val in result.items():
            if "unique" not in key:
                interval = binomtest(k=val, n=n_graphs).proportion_ci(
                    confidence_level=confidence_level
                )
                low, high = interval.low, interval.high
            else:
                low, high = None, None
            result_w_ci[key] = BinomConfidenceInterval(
                mle=val / n_graphs, low=low, high=high
            )
        return result_w_ci
