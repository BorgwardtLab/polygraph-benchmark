#!/usr/bin/env python3
"""Compute subsampling metrics showing bias/variance tradeoff.

Usage:
    python compute.py                                              # Single run (default config)
    python compute.py --multirun                                   # All subsample sizes (default: SLURM GPU)
    python compute.py --multirun hydra/launcher=slurm_gpu          # Explicit SLURM GPU
    python compute.py --multirun hydra/launcher=basic              # Force local execution
    python compute.py --multirun subset=true subsample_size=25,50,100  # Quick test
"""

import json
import time
from pathlib import Path
from typing import List

import hydra
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from polygraph.utils.io import (
    maybe_append_reproducibility_jsonl as maybe_append_jsonl,
)

# Paths (resolved before Hydra touches CWD, though we disable chdir)
REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
EXPERIMENT_RESULTS_DIR = (
    REPO_ROOT / "reproducibility" / "figures" / "01_subsampling" / "results"
)
RESULTS_DIR = EXPERIMENT_RESULTS_DIR / Path(__file__).stem


def get_reference_dataset(dataset: str = "planar", split: str = "test", num_graphs: int = 4096):
    """Get reference dataset from polygraph library."""
    from polygraph.datasets.lobster import ProceduralLobsterGraphDataset
    from polygraph.datasets.planar import ProceduralPlanarGraphDataset
    from polygraph.datasets.sbm import ProceduralSBMGraphDataset

    if dataset == "planar":
        return list(ProceduralPlanarGraphDataset(split=split, num_graphs=num_graphs).to_nx())
    elif dataset == "lobster":
        return list(ProceduralLobsterGraphDataset(split=split, num_graphs=num_graphs).to_nx())
    elif dataset == "sbm":
        return list(ProceduralSBMGraphDataset(split=split, num_graphs=num_graphs).to_nx())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def compute_pgs_for_subsample_size(
    reference_graphs: List,
    generated_graphs: List,
    subsample_size: int,
    num_bootstrap: int = 10,
) -> dict:
    """Compute PGD metrics with a specific subsample size."""
    from polygraph.metrics import StandardPGDInterval

    metric = StandardPGDInterval(
        reference_graphs,
        subsample_size=subsample_size,
        num_samples=num_bootstrap,
    )
    result = metric.compute(generated_graphs)
    return {
        "mean": result["pgd"].mean,
        "std": result["pgd"].std,
    }


@hydra.main(config_path="../configs", config_name="01_subsampling_pgd", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compute PGD metrics for one subsample size and save result as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset
    subsample_size = cfg.subsample_size
    num_bootstrap = 3 if cfg.subset else cfg.num_bootstrap

    logger.info("Computing PGD for dataset={}, subsample_size={}", dataset, subsample_size)

    try:
        reference_graphs = get_reference_dataset(dataset, split="test")
        train_graphs = get_reference_dataset(dataset, split="train")
    except Exception as e:
        logger.error("Error loading dataset: {}", e)
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute.py",
                "dataset": dataset,
                "subsample_size": subsample_size,
                "status": "error",
                "error": str(e),
            }
        )
        return

    if subsample_size > min(len(reference_graphs), len(train_graphs)):
        logger.warning("Subsample size {} exceeds available graphs, skipping", subsample_size)
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute.py",
                "dataset": dataset,
                "subsample_size": subsample_size,
                "status": "skipped",
                "reason": "subsample_size_exceeds_available_graphs",
            }
        )
        return

    try:
        metric_start = time.perf_counter()
        result = compute_pgs_for_subsample_size(
            reference_graphs, train_graphs,
            subsample_size=subsample_size,
            num_bootstrap=num_bootstrap,
        )
        pgs_runtime_perf_seconds = round(time.perf_counter() - metric_start, 6)
        output = {
            "dataset": dataset,
            "subsample_size": subsample_size,
            "pgs_mean": result["mean"],
            "pgs_std": result["std"],
            "pgs_runtime_seconds": pgs_runtime_perf_seconds,
            "pgs_runtime_perf_seconds": pgs_runtime_perf_seconds,
        }

        out_path = RESULTS_DIR / f"{dataset}_{subsample_size}.json"
        out_path.write_text(json.dumps(output, indent=2))
        logger.success("Result saved to {}", out_path)
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute.py",
                "dataset": dataset,
                "subsample_size": subsample_size,
                "status": "ok",
                "output_path": str(out_path),
                "result": output,
                "pgs_runtime_seconds": pgs_runtime_perf_seconds,
                "pgs_runtime_perf_seconds": pgs_runtime_perf_seconds,
            }
        )
    except Exception as e:
        metric_runtime_perf_seconds = round(time.perf_counter() - metric_start, 6)
        logger.error("Error computing PGD for subsample_size={}: {}", subsample_size, e)
        maybe_append_jsonl(
            {
                "experiment": "01_subsampling",
                "script": "compute.py",
                "dataset": dataset,
                "subsample_size": subsample_size,
                "status": "error",
                "error": str(e),
                "pgs_runtime_seconds": metric_runtime_perf_seconds,
                "pgs_runtime_perf_seconds": metric_runtime_perf_seconds,
            }
        )


if __name__ == "__main__":
    main()
