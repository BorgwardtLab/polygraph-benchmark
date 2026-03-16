#!/usr/bin/env python3
"""Compute train-vs-test reference PGD values.

Computes PGD between the train and test splits of each dataset to establish
the metric baseline. This produces the PGD rows of train_test_reference_values.tex.

Usage:
    python compute.py                                              # Single run
    python compute.py --multirun                                   # All datasets
    python compute.py --multirun hydra/launcher=slurm_gpu          # On SLURM (p.hpcl8)
    python compute.py --multirun hydra/launcher=slurm_gpu_hpcl93   # On SLURM (p.hpcl93)
    python compute.py --multirun subset=true                       # Quick test
"""

import json
import sys
from importlib.metadata import version as pkg_version
from typing import Dict, List

import hydra
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from polygraph.utils.io import (
    maybe_append_reproducibility_jsonl as maybe_append_jsonl,
)

sys.path.insert(0, str(here() / "reproducibility"))
from utils.data import get_reference_dataset as _get_ref

# ---------------------------------------------------------------------------
# Paths (resolved before Hydra touches CWD)
# ---------------------------------------------------------------------------
REPO_ROOT = here()
_RESULTS_DIR_BASE = REPO_ROOT / "reproducibility" / "tables" / "results"


def get_reference_dataset(dataset: str, split: str = "test"):
    num = 8192 if split == "train" else 4096
    return _get_ref(dataset, split=split, num_graphs=num)


def compute_pgd_metrics(
    reference_graphs: List,
    generated_graphs: List,
    subset: bool = False,
    classifier=None,
) -> Dict:
    """Compute PGD metrics (reference=test, generated=train)."""
    from polygraph.metrics import StandardPGDInterval

    if subset:
        reference_graphs = reference_graphs[:50]
        generated_graphs = generated_graphs[:50]
        subsample_size = 20
        num_samples = 5
    else:
        min_subset = min(len(reference_graphs), len(generated_graphs))
        subsample_size = min(int(min_subset * 0.5), 2048)
        num_samples = 10

    metric = StandardPGDInterval(
        reference_graphs,
        subsample_size=subsample_size,
        num_samples=num_samples,
        classifier=classifier,
    )
    result = metric.compute(generated_graphs)

    return {
        "polyscore_mean": result["pgd"].mean,
        "polyscore_std": result["pgd"].std,
        "subscores": {
            name: {"mean": interval.mean, "std": interval.std}
            for name, interval in result["subscores"].items()
        },
    }


@hydra.main(
    config_path="../configs",
    config_name="09_train_test_reference",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Compute train-vs-test reference PGD for one dataset."""
    results_suffix: str = cfg.get("results_suffix", "")
    tabpfn_weights_version: str = cfg.get("tabpfn_weights_version", "v2.5")
    subset: bool = cfg.get("subset", False)

    suffix = results_suffix or f"_tabpfn_weights_{tabpfn_weights_version}"
    RESULTS_DIR = _RESULTS_DIR_BASE / f"train_test_reference{suffix}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset

    from tabpfn import TabPFNClassifier
    from tabpfn.classifier import ModelVersion

    version_map = {
        "v2": ModelVersion.V2,
        "v2.5": ModelVersion.V2_5,
    }
    if tabpfn_weights_version not in version_map:
        raise ValueError(
            f"Unknown tabpfn_weights_version: {tabpfn_weights_version!r}. "
            f"Must be one of {list(version_map)}"
        )
    classifier = TabPFNClassifier.create_default_for_version(
        version_map[tabpfn_weights_version],
        device="auto",
        n_estimators=4,
    )

    logger.info("Computing train-vs-test reference PGD for {}", dataset)

    try:
        test_graphs = get_reference_dataset(dataset, split="test")
        train_graphs = get_reference_dataset(dataset, split="train")
    except Exception as e:
        logger.error("Error loading dataset {}: {}", dataset, e)
        maybe_append_jsonl(
            {
                "experiment": "09_train_test_reference",
                "script": "compute.py",
                "dataset": dataset,
                "status": "error",
                "error": str(e),
            }
        )
        return

    logger.info("  train={}, test={}", len(train_graphs), len(test_graphs))

    out_path = RESULTS_DIR / f"{dataset}.json"

    try:
        pgd_results = compute_pgd_metrics(
            test_graphs,
            train_graphs,
            subset=subset,
            classifier=classifier,
        )
    except Exception as e:
        logger.error("Error computing PGD for {}: {}", dataset, e)
        maybe_append_jsonl(
            {
                "experiment": "09_train_test_reference",
                "script": "compute.py",
                "dataset": dataset,
                "status": "error",
                "error": str(e),
            }
        )
        return

    result = {
        "dataset": dataset,
        "tabpfn_weights_version": tabpfn_weights_version,
        "tabpfn_package_version": pkg_version("tabpfn"),
        "pgs_mean": pgd_results["polyscore_mean"],
        "pgs_std": pgd_results["polyscore_std"],
    }
    for key, value in pgd_results["subscores"].items():
        result[f"{key}_mean"] = value["mean"]
        result[f"{key}_std"] = value["std"]

    out_path.write_text(json.dumps(result, indent=2))
    logger.success("Result saved to {}", out_path)
    maybe_append_jsonl(
        {
            "experiment": "09_train_test_reference",
            "script": "compute.py",
            "dataset": dataset,
            "status": "ok",
            "output_path": str(out_path),
            "result": result,
        }
    )


if __name__ == "__main__":
    main()
