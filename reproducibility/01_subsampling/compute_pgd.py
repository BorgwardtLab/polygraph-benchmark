#!/usr/bin/env python3
"""Compute PGD subsampling metrics showing bias/variance tradeoff.

Functionally equivalent to polygraph/experiments/subsampling/subsampling_pgs.py.
Computes StandardPGDInterval across subsample sizes for each (dataset, model)
combination.

Usage:
    python compute_pgd.py                                                   # Single run
    python compute_pgd.py --multirun                                        # All combos (default: SLURM GPU)
    python compute_pgd.py --multirun hydra/launcher=slurm_gpu               # Explicit SLURM GPU
    python compute_pgd.py --multirun hydra/launcher=basic                   # Force local execution
    python compute_pgd.py --multirun subset=true subsample_size=8,16,32     # Quick test
"""

import json
import pickle
import time
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Dict, List

import hydra
import networkx as nx
import torch
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from polygraph.metrics import StandardPGDInterval
from polygraph.utils.io import (
    maybe_append_reproducibility_jsonl as maybe_append_jsonl,
)


def _make_tabpfn_classifier(weights_version: str):
    """Create a TabPFN classifier for the given weights version."""
    from tabpfn import TabPFNClassifier
    from tabpfn.classifier import ModelVersion

    version_map = {
        "v2": ModelVersion.V2,
        "v2.5": ModelVersion.V2_5,
    }
    if weights_version not in version_map:
        raise ValueError(f"Unknown weights_version: {weights_version!r}. Must be one of {list(version_map)}")
    return TabPFNClassifier.create_default_for_version(
        version_map[weights_version], device="auto", n_estimators=4,
    )

# ---------------------------------------------------------------------------
# Paths (resolved before Hydra touches CWD; we disable chdir in the config)
# ---------------------------------------------------------------------------
REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
EXPERIMENT_RESULTS_DIR = (
    REPO_ROOT / "reproducibility" / "figures" / "01_subsampling" / "results"
)


# ---------------------------------------------------------------------------
# Graph loading helpers
# ---------------------------------------------------------------------------
def load_graphs(model: str, dataset: str) -> List[nx.Graph]:
    """Load model-generated graphs from ``data/{model}/{dataset}.pkl``."""
    pkl_path = DATA_DIR / model / f"{dataset}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} not found")
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)

    cleaned: List[nx.Graph] = []
    for g in graphs:
        if isinstance(g, nx.Graph):
            simple = nx.Graph(g)
        elif isinstance(g, (list, tuple)) and len(g) == 2:
            try:
                _node_feat, adj = g
                if isinstance(adj, torch.Tensor):
                    adj = adj.numpy()
                simple = nx.from_numpy_array(adj)
            except Exception as e:
                logger.warning("Could not convert graph: {}", e)
                continue
        else:
            logger.warning("Unknown graph format: {}", type(g))
            continue
        simple.remove_edges_from(nx.selfloop_edges(simple))
        cleaned.append(simple)
    return cleaned


def get_reference_dataset(
    dataset: str, split: str = "train", num_graphs: int = 4096
) -> List[nx.Graph]:
    """Get reference dataset from polygraph procedural generators."""
    from polygraph.datasets.lobster import ProceduralLobsterGraphDataset
    from polygraph.datasets.planar import ProceduralPlanarGraphDataset
    from polygraph.datasets.sbm import ProceduralSBMGraphDataset

    classes = {
        "planar": ProceduralPlanarGraphDataset,
        "lobster": ProceduralLobsterGraphDataset,
        "sbm": ProceduralSBMGraphDataset,
    }
    if dataset not in classes:
        raise ValueError(f"Unknown dataset: {dataset}")
    return list(classes[dataset](split=split, num_graphs=num_graphs).to_nx())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra.main(config_path="../configs", config_name="01_subsampling_pgd", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compute PGD for one (dataset, model, subsample_size) cell."""
    results_suffix: str = cfg.get("results_suffix", "")
    tabpfn_weights_version: str = cfg.get("tabpfn_weights_version", "v2.5")
    RESULTS_DIR = EXPERIMENT_RESULTS_DIR / f"{Path(__file__).stem}{results_suffix}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset: str = cfg.dataset
    model: str = cfg.model
    subsample_size: int = cfg.subsample_size
    num_bootstrap: int = 3 if cfg.subset else cfg.num_bootstrap
    classifier = _make_tabpfn_classifier(tabpfn_weights_version)

    logger.info(
        "PGD subsampling: dataset={}, model={}, n={}, bootstraps={}",
        dataset, model, subsample_size, num_bootstrap,
    )

    # -- Load reference graphs (train split, 10x subsample size as in original) --
    try:
        reference_graphs = get_reference_dataset(
            dataset, split="train", num_graphs=subsample_size * 10
        )
    except Exception as e:
        logger.error("Error loading reference dataset: {}", e)
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute_pgd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "status": "error",
                "error": str(e),
            }
        )
        return

    # -- Load test/generated graphs --
    try:
        if model == "test":
            generated_graphs = get_reference_dataset(
                dataset, split="test", num_graphs=max(subsample_size * 10, 512)
            )
        else:
            generated_graphs = load_graphs(model, dataset)
    except Exception as e:
        logger.error("Error loading graphs for {}/{}: {}", model, dataset, e)
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute_pgd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "status": "error",
                "error": str(e),
            }
        )
        return

    if not generated_graphs:
        logger.warning("No graphs found for {}/{}", model, dataset)
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute_pgd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "status": "skipped",
                "reason": "no_generated_graphs",
            }
        )
        return

    if subsample_size > min(len(reference_graphs), len(generated_graphs)):
        logger.warning(
            "Subsample size {} exceeds available graphs (ref={}, gen={}), skipping",
            subsample_size, len(reference_graphs), len(generated_graphs),
        )
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute_pgd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "status": "skipped",
                "reason": "subsample_size_exceeds_available_graphs",
            }
        )
        return

    # -- Compute PGD --
    metric_start = time.perf_counter()
    try:
        metric = StandardPGDInterval(
            reference_graphs=reference_graphs,
            subsample_size=min(subsample_size, len(generated_graphs)),
            num_samples=num_bootstrap,
            classifier=classifier,
        )
        result = metric.compute(generated_graphs)
        pgd_runtime_perf_seconds = round(time.perf_counter() - metric_start, 6)

        output: Dict = {
            "dataset": dataset,
            "model": model,
            "subsample_size": subsample_size,
            "num_bootstrap": num_bootstrap,
            "pgd_mean": float(result["pgd"].mean),
            "pgd_std": float(result["pgd"].std),
            "pgd_runtime_seconds": pgd_runtime_perf_seconds,
            "pgd_runtime_perf_seconds": pgd_runtime_perf_seconds,
            "tabpfn_package_version": pkg_version("tabpfn"),
            "tabpfn_weights_version": tabpfn_weights_version,
        }
        if result["pgd"].low is not None:
            output["pgd_low"] = float(result["pgd"].low)
        if result["pgd"].high is not None:
            output["pgd_high"] = float(result["pgd"].high)

        # Include per-descriptor subscores
        for name, interval in result.get("subscores", {}).items():
            output[f"{name}_mean"] = float(interval.mean)
            output[f"{name}_std"] = float(interval.std)
            if interval.low is not None:
                output[f"{name}_low"] = float(interval.low)
            if interval.high is not None:
                output[f"{name}_high"] = float(interval.high)

        fname = f"pgd_{dataset}_{model}_{subsample_size}.json"
        out_path = RESULTS_DIR / fname
        out_path.write_text(json.dumps(output, indent=2))
        logger.success("Result saved to {}", out_path)
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute_pgd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "status": "ok",
                "output_path": str(out_path),
                "result": output,
                "pgd_runtime_seconds": pgd_runtime_perf_seconds,
                "pgd_runtime_perf_seconds": pgd_runtime_perf_seconds,
            }
        )
    except Exception as e:
        metric_runtime_perf_seconds = round(time.perf_counter() - metric_start, 6)
        logger.error(
            "Error computing PGD for {}/{}/n={}: {}",
            dataset, model, subsample_size, e,
        )
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute_pgd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "status": "error",
                "error": str(e),
                "pgd_runtime_seconds": metric_runtime_perf_seconds,
                "pgd_runtime_perf_seconds": metric_runtime_perf_seconds,
            }
        )


if __name__ == "__main__":
    main()
