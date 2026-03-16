#!/usr/bin/env python3
"""Compute MMD² metrics from pre-generated graphs.

Usage:
    python compute.py                                              # Single run (default config)
    python compute.py --multirun                                   # All dataset x model combos
    python compute.py --multirun hydra/launcher=slurm_cpu          # On SLURM
    python compute.py --multirun subset=true                       # Quick test
"""

import json
import sys
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
from utils.data import load_graphs as _load

# ---------------------------------------------------------------------------
# Paths (resolved before Hydra touches CWD)
# ---------------------------------------------------------------------------
REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "reproducibility" / "tables" / "results" / "mmd"


def load_graphs(model: str, dataset: str) -> List:
    return _load(DATA_DIR, model, dataset)


def get_reference_dataset(dataset: str, split: str = "test"):
    return _get_ref(dataset, split=split, num_graphs=4096)


def compute_mmd_metrics(
    reference_graphs: List, generated_graphs: List, subset: bool = False
) -> Dict:
    """Compute MMD² metrics using the polygraph library."""
    import numpy as np

    from polygraph.metrics.base import MaxDescriptorMMD2Interval
    from polygraph.metrics.gaussian_tv_mmd import (
        GaussianTVClusteringMMD2Interval,
        GaussianTVDegreeMMD2Interval,
        GaussianTVOrbitMMD2Interval,
        GaussianTVSpectralMMD2Interval,
    )
    from polygraph.metrics.rbf_mmd import (
        RBFClusteringMMD2Interval,
        RBFDegreeMMD2Interval,
        RBFOrbitMMD2Interval,
        RBFSpectralMMD2Interval,
    )
    from polygraph.utils.descriptors import (
        ClusteringHistogram,
        EigenvalueHistogram,
        OrbitCounts,
        SparseDegreeHistogram,
    )
    from polygraph.utils.kernels import AdaptiveRBFKernel

    if subset:
        reference_graphs = reference_graphs[:50]
        generated_graphs = generated_graphs[:50]
        subsample_size = 20
        num_samples = 5
    else:
        subsample_size = min(len(reference_graphs), len(generated_graphs)) // 4
        num_samples = 10

    results = {}

    rbf_metrics = {
        "rbf_degree": RBFDegreeMMD2Interval,
        "rbf_clustering": RBFClusteringMMD2Interval,
        "rbf_orbit": RBFOrbitMMD2Interval,
        "rbf_spectral": RBFSpectralMMD2Interval,
    }

    for name, MetricClass in rbf_metrics.items():
        try:
            metric = MetricClass(
                reference_graphs,
                subsample_size=subsample_size,
                num_samples=num_samples,
            )
            result = metric.compute(generated_graphs)
            results[f"{name}_mean"] = result.mean
            results[f"{name}_std"] = result.std
        except Exception as e:
            logger.error("Error computing {}: {}", name, e)
            results[f"{name}_mean"] = float("nan")
            results[f"{name}_std"] = float("nan")

    gtv_metrics = {
        "gtv_degree": GaussianTVDegreeMMD2Interval,
        "gtv_clustering": GaussianTVClusteringMMD2Interval,
        "gtv_orbit": GaussianTVOrbitMMD2Interval,
        "gtv_spectral": GaussianTVSpectralMMD2Interval,
    }

    for name, MetricClass in gtv_metrics.items():
        try:
            metric = MetricClass(
                reference_graphs,
                subsample_size=subsample_size,
                num_samples=num_samples,
            )
            result = metric.compute(generated_graphs)
            results[f"{name}_mean"] = result.mean
            results[f"{name}_std"] = result.std
        except Exception as e:
            logger.error("Error computing {}: {}", name, e)
            results[f"{name}_mean"] = float("nan")
            results[f"{name}_std"] = float("nan")

    # UMVE variant of RBF MMD
    bws = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])
    umve_descriptors = {
        "umve_degree": SparseDegreeHistogram(),
        "umve_clustering": ClusteringHistogram(bins=100),
        "umve_orbit": OrbitCounts(graphlet_size=4),
        "umve_spectral": EigenvalueHistogram(),
    }
    for name, descriptor in umve_descriptors.items():
        try:
            kernel = AdaptiveRBFKernel(descriptor, bw=bws)
            metric = MaxDescriptorMMD2Interval(
                reference_graphs,
                kernel=kernel,
                variant="umve",
                subsample_size=subsample_size,
                num_samples=num_samples,
            )
            result = metric.compute(generated_graphs)
            results[f"{name}_mean"] = result.mean
            results[f"{name}_std"] = result.std
        except Exception as e:
            logger.error("Error computing {}: {}", name, e)
            results[f"{name}_mean"] = float("nan")
            results[f"{name}_std"] = float("nan")

    # UMVE placeholder
    return results


@hydra.main(config_path="../configs", config_name="06_mmd", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compute MMD metrics for one (dataset, model) pair and save as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset
    model = cfg.model
    subset = cfg.subset

    logger.info("Computing MMD for {}/{}", model, dataset)

    try:
        reference_graphs = get_reference_dataset(dataset, split="test")
    except Exception as e:
        logger.error("Error loading reference dataset: {}", e)
        maybe_append_jsonl(
            {
                "experiment": "06_mmd",
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
                "experiment": "06_mmd",
                "script": "compute.py",
                "dataset": dataset,
                "model": model,
                "status": "skipped",
                "reason": "no_generated_graphs",
            }
        )
        return

    generated_graphs = generated_graphs[: len(reference_graphs)]
    result: Dict = {"dataset": dataset, "model": model}

    try:
        mmd_results = compute_mmd_metrics(
            reference_graphs, generated_graphs, subset=subset
        )
        result.update(mmd_results)
    except Exception as e:
        logger.error("Error computing MMD for {}/{}: {}", model, dataset, e)

    out_path = RESULTS_DIR / f"{dataset}_{model}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.success("Result saved to {}", out_path)
    maybe_append_jsonl(
        {
            "experiment": "06_mmd",
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
