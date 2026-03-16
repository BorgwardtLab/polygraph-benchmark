#!/usr/bin/env python3
"""Compute MMD subsampling metrics showing bias/variance tradeoff.

Functionally equivalent to polygraph/experiments/subsampling/subsampling.py.
Computes MaxDescriptorMMD2 (with AdaptiveRBFKernel) across subsample sizes
for each (dataset, model, descriptor, variant) combination.

Usage:
    python compute_mmd.py                                                   # Single run
    python compute_mmd.py --multirun                                        # All combos (default: SLURM CPU)
    python compute_mmd.py --multirun hydra/launcher=slurm_cpu               # Explicit SLURM CPU
    python compute_mmd.py --multirun hydra/launcher=basic                   # Force local execution
    python compute_mmd.py --multirun subset=true subsample_size=8,16,32     # Quick test
"""

import json
import pickle
import time
from pathlib import Path
from typing import List, Literal, cast

import hydra
import networkx as nx
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from polygraph.metrics.base import MaxDescriptorMMD2Interval
from polygraph.utils.io import (
    maybe_append_reproducibility_jsonl as maybe_append_jsonl,
)
from polygraph.utils.descriptors import (
    ClusteringHistogram,
    EigenvalueHistogram,
    NormalizedDescriptor,
    OrbitCounts,
    RandomGIN,
    SparseDegreeHistogram,
)
from polygraph.utils.kernels import AdaptiveRBFKernel

# ---------------------------------------------------------------------------
# Paths (resolved before Hydra touches CWD; we disable chdir in the config)
# ---------------------------------------------------------------------------
REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
EXPERIMENT_RESULTS_DIR = (
    REPO_ROOT / "reproducibility" / "figures" / "01_subsampling" / "results"
)
RESULTS_DIR = EXPERIMENT_RESULTS_DIR / Path(__file__).stem

# Bandwidths used in the original subsampling.py experiment
BANDWIDTHS = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])

# Descriptor names matching the original experiment
DESCRIPTOR_NAMES = [
    "orbit4",
    "orbit5",
    "degree",
    "spectral",
    "clustering",
    "gin",
]


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
# Descriptor factory
# ---------------------------------------------------------------------------
def make_descriptor(name: str, reference_graphs: List[nx.Graph]):
    """Instantiate a descriptor by name, matching the original experiment."""
    factories = {
        "orbit4": lambda: OrbitCounts(graphlet_size=4),
        "orbit5": lambda: OrbitCounts(graphlet_size=5),
        "degree": lambda: SparseDegreeHistogram(),
        "spectral": lambda: EigenvalueHistogram(),
        "clustering": lambda: ClusteringHistogram(bins=100),
        "gin": lambda: NormalizedDescriptor(
            RandomGIN(seed=42), ref_graphs=reference_graphs
        ),
    }
    if name not in factories:
        raise ValueError(
            f"Unknown descriptor: {name}. Choose from {list(factories.keys())}"
        )
    return factories[name]()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra.main(
    config_path="../configs",
    config_name="01_subsampling_mmd",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Compute MMD for one (dataset, model, subsample_size, descriptor, variant) cell."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset: str = cfg.dataset
    model: str = cfg.model
    subsample_size: int = cfg.subsample_size
    descriptor_name: str = cfg.descriptor
    variant: str = cfg.variant
    num_bootstrap: int = 3 if cfg.subset else cfg.num_bootstrap

    logger.info(
        "MMD subsampling: dataset={}, model={}, n={}, descriptor={}, variant={}, bootstraps={}",
        dataset,
        model,
        subsample_size,
        descriptor_name,
        variant,
        num_bootstrap,
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
                "script": "compute_mmd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "descriptor": descriptor_name,
                "variant": variant,
                "status": "error",
                "error": str(e),
            }
        )
        return

    # -- Load test/generated graphs --
    try:
        if model == "test":
            generated_graphs = get_reference_dataset(
                dataset, split="test", num_graphs=subsample_size
            )
        else:
            generated_graphs = load_graphs(model, dataset)
    except Exception as e:
        logger.error("Error loading graphs for {}/{}: {}", model, dataset, e)
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute_mmd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "descriptor": descriptor_name,
                "variant": variant,
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
                "script": "compute_mmd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "descriptor": descriptor_name,
                "variant": variant,
                "status": "skipped",
                "reason": "no_generated_graphs",
            }
        )
        return

    if subsample_size > min(len(reference_graphs), len(generated_graphs)):
        logger.warning(
            "Subsample size {} exceeds available graphs (ref={}, gen={}), skipping",
            subsample_size,
            len(reference_graphs),
            len(generated_graphs),
        )
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute_mmd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "descriptor": descriptor_name,
                "variant": variant,
                "status": "skipped",
                "reason": "subsample_size_exceeds_available_graphs",
            }
        )
        return

    # -- Compute MMD --
    metric_start = time.perf_counter()
    try:
        descriptor = make_descriptor(descriptor_name, reference_graphs)
        kernel = AdaptiveRBFKernel(descriptor_fn=descriptor, bw=BANDWIDTHS)
        metric = MaxDescriptorMMD2Interval(
            reference_graphs=reference_graphs,
            kernel=kernel,
            subsample_size=min(subsample_size, len(generated_graphs)),
            num_samples=num_bootstrap,
            variant=cast(Literal["biased", "umve", "ustat"], variant),
        )
        result = metric.compute(generated_graphs)
        mmd_runtime_perf_seconds = round(time.perf_counter() - metric_start, 6)

        output = {
            "dataset": dataset,
            "model": model,
            "subsample_size": subsample_size,
            "descriptor": descriptor_name,
            "variant": variant,
            "num_bootstrap": num_bootstrap,
            "mmd_mean": float(result.mean),
            "mmd_std": float(result.std),
            "mmd_low": float(result.low) if result.low is not None else None,
            "mmd_high": float(result.high) if result.high is not None else None,
            "mmd_runtime_seconds": mmd_runtime_perf_seconds,
            "mmd_runtime_perf_seconds": mmd_runtime_perf_seconds,
        }

        fname = (
            f"mmd_{dataset}_{model}_{subsample_size}"
            f"_{descriptor_name}_{variant}.json"
        )
        out_path = RESULTS_DIR / fname
        out_path.write_text(json.dumps(output, indent=2))
        logger.success("Result saved to {}", out_path)
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute_mmd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "descriptor": descriptor_name,
                "variant": variant,
                "status": "ok",
                "output_path": str(out_path),
                "result": output,
                "mmd_runtime_seconds": mmd_runtime_perf_seconds,
                "mmd_runtime_perf_seconds": mmd_runtime_perf_seconds,
            }
        )
    except Exception as e:
        metric_runtime_perf_seconds = round(
            time.perf_counter() - metric_start, 6
        )
        logger.error(
            "Error computing MMD for {}/{}/n={}/{}/{}: {}",
            dataset,
            model,
            subsample_size,
            descriptor_name,
            variant,
            e,
        )
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute_mmd.py",
                "dataset": dataset,
                "model": model,
                "subsample_size": subsample_size,
                "descriptor": descriptor_name,
                "variant": variant,
                "status": "error",
                "error": str(e),
                "mmd_runtime_seconds": metric_runtime_perf_seconds,
                "mmd_runtime_perf_seconds": metric_runtime_perf_seconds,
            }
        )


if __name__ == "__main__":
    main()
