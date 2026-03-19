"""VUN (Valid-Unique-Novel) computation helpers for multiprocessing.

Worker functions live here (not in __main__) so they can be pickled
by multiprocessing when submitit wraps the calling script.
"""

import json
import signal
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional

import networkx as nx
from loguru import logger


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError()


def _is_isomorphic_with_timeout(g: nx.Graph, h: nx.Graph, timeout: int) -> bool:
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        result = nx.is_isomorphic(g, h)
    except _TimeoutError:
        result = True  # Conservative: treat timeout as isomorphic
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return result


class GraphSet:
    """Graph set with WL hash pre-filter and per-pair isomorphism timeout."""

    def __init__(self, nx_graphs: Optional[list] = None, iso_timeout: int = 10):
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


def _check_validity_worker(graph_json: str, dataset: str) -> bool:
    """Check validity of a single graph in a worker process."""
    g = nx.node_link_graph(json.loads(graph_json))
    if dataset == "planar":
        from polygraph.datasets.planar import is_planar_graph

        return is_planar_graph(g)
    elif dataset == "lobster":
        from polygraph.datasets.lobster import is_lobster_graph

        return is_lobster_graph(g)
    elif dataset == "sbm":
        from polygraph.datasets.sbm import is_sbm_graph

        return is_sbm_graph(g)
    return True


def compute_vun_parallel(
    train_graphs: List[nx.Graph],
    generated_graphs: List[nx.Graph],
    dataset: str,
    iso_timeout: int = 10,
    n_workers: int = 8,
) -> Dict[str, float]:
    """Compute VUN metrics with parallel validity and novelty checking."""
    n = len(generated_graphs)

    logger.info("  Validity check ({} workers)...", n_workers)
    gen_json_for_validity = [
        json.dumps(nx.node_link_data(g)) for g in generated_graphs
    ]
    worker_fn = partial(_check_validity_worker, dataset=dataset)
    with Pool(processes=n_workers) as pool:
        valid = pool.map(worker_fn, gen_json_for_validity, chunksize=32)
    logger.info("  Valid: {}/{}", sum(valid), n)

    logger.info("  Uniqueness check (sequential)...")
    gen_set = GraphSet(iso_timeout=iso_timeout)
    unique = []
    for g in generated_graphs:
        unique.append(g not in gen_set)
        gen_set.add(g)
    logger.info("  Unique: {}/{}", sum(unique), n)

    logger.info("  Novelty check ({} workers)...", n_workers)
    train_hashes: Dict[str, List[int]] = defaultdict(list)
    for idx, g in enumerate(train_graphs):
        train_hashes[nx.weisfeiler_lehman_graph_hash(g)].append(idx)

    train_json = [json.dumps(nx.node_link_data(g)) for g in train_graphs]
    gen_json = [json.dumps(nx.node_link_data(g)) for g in generated_graphs]

    worker_fn = partial(
        _check_novel_worker,
        train_graphs_json=train_json,
        train_hashes=dict(train_hashes),
        iso_timeout=iso_timeout,
    )

    with Pool(processes=n_workers) as pool:
        novel = pool.map(worker_fn, gen_json, chunksize=64)
    logger.info("  Novel: {}/{}", sum(novel), n)

    unique_novel = [u and nv for u, nv in zip(unique, novel)]
    valid_unique_novel = [un and v for un, v in zip(unique_novel, valid)]

    return {
        "valid": sum(valid) / n,
        "unique": sum(unique) / n,
        "novel": sum(novel) / n,
        "unique_novel": sum(unique_novel) / n,
        "valid_unique_novel": sum(valid_unique_novel) / n,
    }
