#!/usr/bin/env python3
"""Compute VUN metrics for a single (dataset, model) combo and patch JSON results.

Uses a 10s per-pair timeout on isomorphism checks with parallel novelty
checking across generated graphs.

Usage:
    # Single combo
    python compute_vun.py --dataset sbm --model AUTOGRAPH

    # All missing combos (for local use)
    python compute_vun.py --all-missing
"""

import json
import signal
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional

import networkx as nx
import typer
from loguru import logger
from pyprojroot import here

sys.path.insert(0, str(here() / "reproducibility"))
from utils.data import get_reference_dataset as _get_ref
from utils.data import load_graphs as _load

REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
RESULTS_BASE = REPO_ROOT / "reproducibility" / "tables" / "results"

DATASETS = ["planar", "lobster", "sbm"]
MODELS = ["AUTOGRAPH", "DIGRESS", "GRAN", "ESGG"]

app = typer.Typer()


# ---------------------------------------------------------------------------
# Per-pair isomorphism with SIGALRM timeout
# ---------------------------------------------------------------------------

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


class _GraphSet:
    """Graph set with WL hash pre-filter and per-pair isomorphism timeout."""

    def __init__(self, nx_graphs: Optional[Iterable[nx.Graph]] = None, iso_timeout: int = 10):
        self.nx_graphs = [] if nx_graphs is None else list(nx_graphs)
        self._iso_timeout = iso_timeout
        self._hash_set: Dict[str, List[int]] = defaultdict(list)
        for idx, g in enumerate(self.nx_graphs):
            self._hash_set[nx.weisfeiler_lehman_graph_hash(g)].append(idx)

    def add(self, g: nx.Graph) -> None:
        self.nx_graphs.append(g)
        self._hash_set[nx.weisfeiler_lehman_graph_hash(g)].append(len(self.nx_graphs) - 1)

    def __contains__(self, g: nx.Graph) -> bool:
        fp = nx.weisfeiler_lehman_graph_hash(g)
        if fp not in self._hash_set:
            return False
        for idx in self._hash_set[fp]:
            if _is_isomorphic_with_timeout(g, self.nx_graphs[idx], self._iso_timeout):
                return True
        return False


# ---------------------------------------------------------------------------
# Parallel novelty worker
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
    validity_fn: Callable,
    iso_timeout: int = 10,
    n_workers: int = 8,
    dataset: str = "",
) -> Dict[str, float]:
    n = len(generated_graphs)

    logger.info("  Validity check ({} workers)...", n_workers)
    gen_json_for_validity = [json.dumps(nx.node_link_data(g)) for g in generated_graphs]
    worker_fn = partial(_check_validity_worker, dataset=dataset)
    valid = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for is_valid in executor.map(worker_fn, gen_json_for_validity, chunksize=32):
            valid.append(is_valid)
    logger.info("  Valid: {}/{}", sum(valid), n)

    logger.info("  Uniqueness check (sequential)...")
    gen_set = _GraphSet(iso_timeout=iso_timeout)
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

    novel = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for is_novel in executor.map(worker_fn, gen_json, chunksize=64):
            novel.append(is_novel)
    logger.info("  Novel: {}/{}", sum(novel), n)

    unique_novel = [u and nv for u, nv in zip(unique, novel)]
    valid_unique_novel = [un and v for un, v in zip(unique_novel, valid)]

    return {
        "unique": sum(unique) / n,
        "novel": sum(novel) / n,
        "valid": sum(valid) / n,
        "unique_novel": sum(unique_novel) / n,
        "valid_unique_novel": sum(valid_unique_novel) / n,
    }


def _get_validity_fn(dataset: str) -> Optional[Callable]:
    from polygraph.datasets.lobster import is_lobster_graph
    from polygraph.datasets.planar import is_planar_graph
    from polygraph.datasets.sbm import is_sbm_graph

    return {"planar": is_planar_graph, "lobster": is_lobster_graph, "sbm": is_sbm_graph}.get(dataset)


def _get_result_dirs() -> List:
    dirs = []
    for d in RESULTS_BASE.iterdir():
        if d.is_dir() and d.name.startswith("benchmark"):
            dirs.append(d)
    return dirs


def _has_vun(result_dirs, dataset, model) -> bool:
    for rdir in result_dirs:
        p = rdir / f"{dataset}_{model}.json"
        if p.exists():
            data = json.loads(p.read_text())
            if "vun" in data and data["vun"] is not None:
                return True
    return False


def _run_one(dataset: str, model: str, iso_timeout: int, n_workers: int, force: bool):
    """Compute VUN for a single (dataset, model) and patch all result JSONs."""
    result_dirs = _get_result_dirs()
    if not force and _has_vun(result_dirs, dataset, model):
        logger.info("Skipping {}/{} (VUN already present)", model, dataset)
        return

    validity_fn = _get_validity_fn(dataset)
    if validity_fn is None:
        logger.info("No validity function for {}, skipping", dataset)
        return

    logger.info("Loading data for {}/{}...", model, dataset)
    train_graphs = _get_ref(dataset, split="train", num_graphs=8192)
    ref_graphs = _get_ref(dataset, split="test", num_graphs=4096)
    generated_graphs = _load(DATA_DIR, model, dataset)
    if not generated_graphs:
        logger.warning("No graphs for {}/{}", model, dataset)
        return
    generated_graphs = generated_graphs[:len(ref_graphs)]

    logger.info("Computing VUN for {}/{} ({} gen, {} train)...",
                model, dataset, len(generated_graphs), len(train_graphs))

    result = compute_vun_parallel(
        train_graphs, generated_graphs, validity_fn,
        iso_timeout=iso_timeout, n_workers=n_workers, dataset=dataset,
    )

    vun_value = result["valid_unique_novel"]
    logger.success("{}/{}: VUN = {:.4f}", model, dataset, vun_value)

    for rdir in result_dirs:
        json_path = rdir / f"{dataset}_{model}.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())
        data["vun"] = vun_value
        json_path.write_text(json.dumps(data, indent=2))
        logger.info("Patched {}", json_path)


@app.command()
def main(
    dataset: Optional[str] = typer.Option(None, help="Dataset name (planar/lobster/sbm)"),
    model: Optional[str] = typer.Option(None, help="Model name (AUTOGRAPH/DIGRESS/GRAN/ESGG)"),
    all_missing: bool = typer.Option(False, "--all-missing", help="Run all combos missing VUN"),
    iso_timeout: int = typer.Option(10, help="Per-pair isomorphism timeout in seconds"),
    n_workers: int = typer.Option(8, help="Number of parallel workers for novelty checks"),
    force: bool = typer.Option(False, help="Recompute even if VUN already present"),
):
    """Compute VUN for one or all dataset/model combos and patch JSON results."""
    if all_missing:
        for ds in DATASETS:
            for m in MODELS:
                _run_one(ds, m, iso_timeout, n_workers, force)
    elif dataset and model:
        _run_one(dataset, model, iso_timeout, n_workers, force)
    else:
        logger.error("Provide --dataset + --model, or --all-missing")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
