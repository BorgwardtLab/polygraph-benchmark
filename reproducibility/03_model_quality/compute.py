#!/usr/bin/env python3
"""Compute model quality metrics (training/denoising iterations).

Computes PGD, RBF MMD, and validity metrics for each checkpoint of DiGress
training or denoising curves. Matches original evaluate.py from polygraph.

Usage:
    python compute.py
    python compute.py --multirun
    python compute.py --multirun hydra/launcher=slurm_cpu
    python compute.py --multirun subset=true
"""

import json
import pickle
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import List

import hydra
import networkx as nx
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from polygraph.utils.io import maybe_append_reproducibility_jsonl as maybe_append_jsonl

REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
_RESULTS_DIR_BASE = REPO_ROOT / "reproducibility" / "figures" / "03_model_quality"


def load_graphs(path: Path) -> List[nx.Graph]:
    """Load graphs from pickle file and convert to networkx."""
    if not path.exists():
        logger.warning("{} not found", path)
        return []
    with open(path, "rb") as f:
        data = pickle.load(f)
    graphs = []
    for item in data:
        if isinstance(item, nx.Graph):
            graphs.append(item)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            adj = item[1]
            if hasattr(adj, "numpy"):
                adj = adj.numpy()
            graphs.append(nx.from_numpy_array(adj))
        else:
            graphs.append(nx.from_numpy_array(np.array(item)))
    return graphs


def get_reference_dataset(dataset, split="train", num_graphs=2048):
    """Get reference dataset from polygraph library."""
    if dataset == "planar":
        from polygraph.datasets.planar import ProceduralPlanarGraphDataset
        ds = ProceduralPlanarGraphDataset(split=split, num_graphs=num_graphs)
    elif dataset == "sbm":
        from polygraph.datasets.sbm import ProceduralSBMGraphDataset
        ds = ProceduralSBMGraphDataset(split=split, num_graphs=num_graphs)
    elif dataset == "lobster":
        from polygraph.datasets.lobster import ProceduralLobsterGraphDataset
        ds = ProceduralLobsterGraphDataset(split=split, num_graphs=num_graphs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return ds, list(ds.to_nx())


@hydra.main(
    config_path="../configs",
    config_name="03_model_quality",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Compute PGD, MMD, and validity for all checkpoints."""
    results_suffix: str = cfg.get("results_suffix", "")
    tabpfn_weights_version: str = cfg.get("tabpfn_weights_version", "v2.5")
    RESULTS_DIR = _RESULTS_DIR_BASE / f"results{results_suffix}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    curve_type = cfg.curve_type
    dataset = cfg.dataset
    variant = cfg.variant
    subset = cfg.subset
    pgd_only = cfg.get("pgd_only", False)
    num_graphs = cfg.get("num_graphs", 2048)

    dir_map = {
        "training": "training-iterations",
        "denoising": "denoising-iterations",
    }
    base_dir = DATA_DIR / "DIGRESS" / dir_map[curve_type]
    # For training, use per-dataset subdirectory if it exists
    iteration_dir = base_dir / dataset if (base_dir / dataset).exists() else base_dir

    if not iteration_dir.exists():
        logger.warning("Iteration directory not found: {}", iteration_dir)
        maybe_append_jsonl(
            {
                "experiment": "03_model_quality",
                "script": "compute.py",
                "curve_type": curve_type,
                "dataset": dataset,
                "variant": variant,
                "status": "skipped",
                "reason": "iteration_dir_missing",
            }
        )
        return

    def _parse_steps(p: Path) -> int:
        """Parse step number from either 'N_steps.pkl' or 'epoch_N.pkl' format."""
        parts = p.stem.split("_")
        try:
            return int(parts[0])  # N_steps.pkl
        except ValueError:
            return int(parts[-1])  # epoch_N.pkl

    pkl_files = sorted(
        iteration_dir.glob("*.pkl"),
        key=_parse_steps,
    )
    if not pkl_files:
        logger.warning("No checkpoint files found in {}", iteration_dir)
        maybe_append_jsonl(
            {
                "experiment": "03_model_quality",
                "script": "compute.py",
                "curve_type": curve_type,
                "dataset": dataset,
                "variant": variant,
                "status": "skipped",
                "reason": "no_checkpoint_files",
            }
        )
        return

    logger.info(
        "Processing {} checkpoints for {} curve, dataset={}, variant={}",
        len(pkl_files), curve_type, dataset, variant,
    )

    try:
        ds_obj, reference_graphs = get_reference_dataset(
            dataset, split="train", num_graphs=num_graphs
        )
    except Exception as e:
        logger.error("Error loading reference dataset: {}", e)
        maybe_append_jsonl(
            {
                "experiment": "03_model_quality",
                "script": "compute.py",
                "curve_type": curve_type,
                "dataset": dataset,
                "variant": variant,
                "status": "error",
                "error": str(e),
            }
        )
        return

    from polygraph.metrics import StandardPGD
    from polygraph.metrics.rbf_mmd import (
        RBFClusteringMMD2,
        RBFDegreeMMD2,
        RBFGraphNeuralNetworkMMD2,
        RBFOrbitMMD2,
        RBFSpectralMMD2,
    )
    from polygraph.metrics.base.mmd import MaxDescriptorMMD2
    from polygraph.utils.descriptors import OrbitCounts
    from polygraph.utils.kernels import AdaptiveRBFKernel

    ref = reference_graphs
    if subset:
        ref = ref[:30]

    if tabpfn_weights_version == "v2":
        from tabpfn import TabPFNClassifier
        from tabpfn.classifier import ModelVersion
        classifier = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device="auto", n_estimators=4,
        )
    else:
        classifier = None  # default (v2.5)

    pgd_metric = StandardPGD(
        reference_graphs=ref,
        variant=variant,
        classifier=classifier,
    )

    if not pgd_only:
        bw = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])
        mmd_metrics = {
            "orbit_mmd": RBFOrbitMMD2(ref),
            "orbit5_mmd": MaxDescriptorMMD2(
                ref,
                kernel=AdaptiveRBFKernel(
                    descriptor_fn=OrbitCounts(graphlet_size=5), bw=bw,
                ),
                variant="biased",
            ),
            "degree_mmd": RBFDegreeMMD2(ref),
            "spectral_mmd": RBFSpectralMMD2(ref),
            "clustering_mmd": RBFClusteringMMD2(ref),
            "gin_mmd": RBFGraphNeuralNetworkMMD2(ref),
        }

    results = []
    for pkl_path in pkl_files:
        steps = _parse_steps(pkl_path)
        logger.info("  {} steps...", steps)

        graphs = load_graphs(pkl_path)
        if not graphs:
            continue

        gen = graphs[:num_graphs]
        if subset:
            gen = gen[:30]

        entry = {"steps": steps}

        try:
            pgd_result = pgd_metric.compute(gen)
            entry["polyscore"] = pgd_result["pgd"]
            for key, val in pgd_result["subscores"].items():
                entry[f"{key}_pgs"] = val
        except Exception as e:
            logger.error("PGD error at step {}: {}", steps, e)

        if not pgd_only:
            for mmd_name, mmd_metric in mmd_metrics.items():
                try:
                    entry[mmd_name] = mmd_metric.compute(gen)
                except Exception as e:
                    logger.error("MMD {} error at step {}: {}", mmd_name, steps, e)

            try:
                valid_count = sum(1 for g in gen if ds_obj.is_valid(g))
                entry["validity"] = valid_count / len(gen)
            except Exception as e:
                logger.error("Validity error at step {}: {}", steps, e)

        results.append(entry)

    if results:
        output = {
            "curve_type": curve_type,
            "dataset": dataset,
            "variant": variant,
            "results": results,
            "tabpfn_package_version": pkg_version("tabpfn"),
            "tabpfn_weights_version": tabpfn_weights_version,
        }
        out_path = RESULTS_DIR / f"{curve_type}_{dataset}_{variant}.json"
        out_path.write_text(json.dumps(output, indent=2))
        logger.success("Saved {} results to {}", len(results), out_path)
        maybe_append_jsonl(
            {
                "experiment": "03_model_quality",
                "script": "compute.py",
                "curve_type": curve_type,
                "dataset": dataset,
                "variant": variant,
                "status": "ok",
                "output_path": str(out_path),
                "num_rows": len(results),
            }
        )
    else:
        logger.warning("No results computed for {} curve", curve_type)
        maybe_append_jsonl(
            {
                "experiment": "03_model_quality",
                "script": "compute.py",
                "curve_type": curve_type,
                "dataset": dataset,
                "variant": variant,
                "status": "skipped",
                "reason": "no_results_computed",
            }
        )


if __name__ == "__main__":
    main()
