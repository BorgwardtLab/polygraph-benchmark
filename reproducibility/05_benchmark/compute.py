#!/usr/bin/env python3
"""Compute benchmark metrics (Table 1) from pre-generated graphs.

Usage:
    python compute.py                                              # Single run (default config)
    python compute.py --multirun                                   # All dataset x model combos
    python compute.py --multirun hydra/launcher=slurm_cpu          # On SLURM
    python compute.py --multirun subset=true                       # Quick test
"""

import json
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from polygraph.utils.io import maybe_append_reproducibility_jsonl as maybe_append_jsonl

sys.path.insert(0, str(here() / "reproducibility"))
from utils.data import get_reference_dataset as _get_ref
from utils.data import load_graphs as _load

# ---------------------------------------------------------------------------
# Paths (resolved before Hydra touches CWD)
# ---------------------------------------------------------------------------
REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
_RESULTS_DIR_BASE = REPO_ROOT / "reproducibility" / "tables" / "results"


def load_graphs(model: str, dataset: str) -> List:
    return _load(DATA_DIR, model, dataset)


def get_reference_dataset(dataset: str, split: str = "test"):
    num = 8192 if split == "train" else 4096
    return _get_ref(dataset, split=split, num_graphs=num)


def compute_pgs_metrics(reference_graphs: List, generated_graphs: List, dataset: str = "", subset: bool = False) -> Dict:
    """Compute PGD metrics using the polygraph library."""
    from polygraph.metrics import StandardPGDInterval

    if subset:
        reference_graphs = reference_graphs[:50]
        generated_graphs = generated_graphs[:50]
        subsample_size = 20
        num_samples = 5
    else:
        min_subset = min(len(reference_graphs), len(generated_graphs))
        # subsample_size must be <= min_subset / 2 for PolyGraphDiscrepancyInterval
        subsample_size = min(int(min_subset * 0.5), 2048)
        num_samples = 10

    metric = StandardPGDInterval(reference_graphs, subsample_size=subsample_size, num_samples=num_samples)
    result = metric.compute(generated_graphs)

    return {
        "polyscore_mean": result["pgd"].mean,
        "polyscore_std": result["pgd"].std,
        "subscores": {
            name: {"mean": interval.mean, "std": interval.std}
            for name, interval in result["subscores"].items()
        }
    }


def compute_vun_metrics(train_graphs: List, generated_graphs: List, dataset: str, subset: bool = False) -> Optional[Dict]:
    """Compute VUN metrics for datasets that support validity checking."""
    from polygraph.datasets.lobster import is_lobster_graph
    from polygraph.datasets.planar import is_planar_graph
    from polygraph.datasets.sbm import is_sbm_graph
    from polygraph.metrics import VUN

    validity_fns = {
        "planar": is_planar_graph,
        "lobster": is_lobster_graph,
        "sbm": is_sbm_graph,
    }

    if dataset not in validity_fns:
        return None

    if subset:
        train_graphs = train_graphs[:50]
        generated_graphs = generated_graphs[:50]

    vun_metric = VUN(
        train_graphs=train_graphs,
        validity_fn=validity_fns[dataset],
        iso_timeout=10,
        n_jobs=10,
    )

    results = vun_metric.compute(generated_graphs)
    return results


@hydra.main(config_path="../configs", config_name="05_benchmark", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compute benchmark metrics for one (dataset, model) pair and save as JSON."""
    results_suffix: str = cfg.get("results_suffix", "")
    RESULTS_DIR = _RESULTS_DIR_BASE / f"benchmark{results_suffix}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset
    model = cfg.model
    subset = cfg.subset

    logger.info("Computing benchmark for {}/{}", model, dataset)

    try:
        reference_graphs = get_reference_dataset(dataset, split="test")
        train_graphs = get_reference_dataset(dataset, split="train")
    except Exception as e:
        logger.error("Error loading reference dataset: {}", e)
        maybe_append_jsonl(
            {
                "experiment": "05_benchmark",
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
                "experiment": "05_benchmark",
                "script": "compute.py",
                "dataset": dataset,
                "model": model,
                "status": "skipped",
                "reason": "no_generated_graphs",
            }
        )
        return

    generated_graphs = generated_graphs[:len(reference_graphs)]
    result: Dict = {
        "dataset": dataset,
        "model": model,
        "tabpfn_package_version": pkg_version("tabpfn"),
    }

    out_path = RESULTS_DIR / f"{dataset}_{model}.json"

    try:
        pgs_results = compute_pgs_metrics(reference_graphs, generated_graphs, dataset=dataset, subset=subset)
        result["pgs_mean"] = pgs_results.get("polyscore_mean", float("nan"))
        result["pgs_std"] = pgs_results.get("polyscore_std", float("nan"))
        for key, value in pgs_results.get("subscores", {}).items():
            if isinstance(value, dict):
                result[f"{key}_mean"] = value.get("mean", float("nan"))
                result[f"{key}_std"] = value.get("std", float("nan"))
        # Save intermediate results after PGD (before VUN which can be very slow)
        out_path.write_text(json.dumps(result, indent=2))
        logger.info("PGD results saved to {}", out_path)
    except Exception as e:
        logger.error("Error computing PGD for {}/{}: {}", model, dataset, e)

    try:
        vun_results = compute_vun_metrics(train_graphs, generated_graphs, dataset, subset=subset)
        if vun_results:
            result["vun"] = vun_results.get("valid_unique_novel_mle", float("nan"))
    except Exception as e:
        logger.error("Error computing VUN for {}/{}: {}", model, dataset, e)

    out_path.write_text(json.dumps(result, indent=2))
    logger.success("Result saved to {}", out_path)
    maybe_append_jsonl(
        {
            "experiment": "05_benchmark",
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
