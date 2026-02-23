#!/usr/bin/env python3
"""Compute GKLR (Graph Kernel Logistic Regression) metrics from pre-generated graphs.

Usage:
    python compute.py                                              # Single run (default config)
    python compute.py --multirun                                   # All dataset x model combos
    python compute.py --multirun hydra/launcher=slurm_cpu          # On SLURM
    python compute.py --multirun subset=true                       # Quick test
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import hydra
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from polygraph.utils.io import maybe_append_reproducibility_jsonl as maybe_append_jsonl

sys.path.insert(0, str(here() / "reproducibility"))
from utils.data import get_reference_dataset
from utils.data import load_graphs as _load

# ---------------------------------------------------------------------------
# Paths (resolved before Hydra touches CWD)
# ---------------------------------------------------------------------------
REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "reproducibility" / "tables" / "results" / "gklr"


def load_graphs(model: str, dataset: str) -> List:
    return _load(DATA_DIR, model, dataset)


def compute_gklr_metrics(reference_graphs: List, generated_graphs: List, subset: bool = False) -> Dict:
    """Compute PGD metrics using graph kernel descriptors with kernel logistic regression."""
    from polygraph.metrics.base import KernelLogisticRegression, PolyGraphDiscrepancyInterval
    from polygraph.utils.descriptors import (
        WeisfeilerLehmanDescriptor,
        ShortestPathHistogramDescriptor,
        PyramidMatchDescriptor,
    )

    if subset:
        reference_graphs = reference_graphs[:50]
        generated_graphs = generated_graphs[:50]
        subsample_size = 20
        num_samples = 5
    else:
        subsample_size = min(len(reference_graphs), len(generated_graphs)) // 4
        num_samples = 10

    descriptors = {
        "wl": WeisfeilerLehmanDescriptor(),
        "shortest_path": ShortestPathHistogramDescriptor(),
        "pyramid_match": PyramidMatchDescriptor(),
    }

    classifier = KernelLogisticRegression(max_iter=1000)

    metric = PolyGraphDiscrepancyInterval(
        reference_graphs,
        descriptors=descriptors,
        subsample_size=subsample_size,
        num_samples=num_samples,
        variant="jsd",
        classifier=classifier,
        scale=False,
    )

    result = metric.compute(generated_graphs)

    return {
        "pgd_mean": result["pgd"].mean,
        "pgd_std": result["pgd"].std,
        "subscores": {
            name: {"mean": interval.mean, "std": interval.std}
            for name, interval in result["subscores"].items()
        }
    }


@hydra.main(config_path="../configs", config_name="08_gklr", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compute GKLR metrics for one (dataset, model) pair and save as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset
    model = cfg.model
    subset = cfg.subset

    logger.info("Computing GKLR for {}/{}", model, dataset)

    try:
        reference_graphs = get_reference_dataset(dataset, split="test", num_graphs=512)
    except Exception as e:
        logger.error("Error loading reference dataset: {}", e)
        maybe_append_jsonl(
            {
                "experiment": "08_gklr",
                "script": "compute.py",
                "dataset": dataset,
                "model": model,
                "status": "error",
                "error": str(e),
            }
        )
        return

    generated_graphs = load_graphs(model, dataset)
    if not generated_graphs:
        logger.warning("No graphs found for {}/{}", model, dataset)
        maybe_append_jsonl(
            {
                "experiment": "08_gklr",
                "script": "compute.py",
                "dataset": dataset,
                "model": model,
                "status": "skipped",
                "reason": "no_generated_graphs",
            }
        )
        return

    generated_graphs = generated_graphs[:len(reference_graphs)]
    result: Dict = {"dataset": dataset, "model": model}

    try:
        gklr_results = compute_gklr_metrics(reference_graphs, generated_graphs, subset=subset)
        result["pgs_mean"] = gklr_results.get("pgd_mean", float("nan"))
        result["pgs_std"] = gklr_results.get("pgd_std", float("nan"))
        for key, value in gklr_results.get("subscores", {}).items():
            if isinstance(value, dict):
                result[f"{key}_mean"] = value.get("mean", float("nan"))
                result[f"{key}_std"] = value.get("std", float("nan"))
    except Exception as e:
        logger.error("Error computing GKLR for {}/{}: {}", model, dataset, e)

    out_path = RESULTS_DIR / f"{dataset}_{model}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.success("Result saved to {}", out_path)
    maybe_append_jsonl(
        {
            "experiment": "08_gklr",
            "script": "compute.py",
            "dataset": dataset,
            "model": model,
            "status": "ok",
            "output_path": str(out_path),
            "result": result,
        }
    )


if __name__ == "__main__":
    main()
