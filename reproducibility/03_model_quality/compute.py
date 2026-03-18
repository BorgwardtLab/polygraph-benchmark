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
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, List, Literal, Tuple, cast

import hydra
import networkx as nx
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from polygraph.utils.io import (
    maybe_append_jsonl,
)

sys.path.insert(0, str(here() / "reproducibility"))
from utils.data import load_graphs as _load
from utils.data import make_tabpfn_classifier

REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
_RESULTS_DIR_BASE = (
    REPO_ROOT / "reproducibility" / "figures" / "03_model_quality"
)


def load_graphs(path: Path) -> List[nx.Graph]:
    """Load graphs from a single pickle file.

    Delegates to ``utils.data.load_graphs`` by extracting the parent
    directory and stem so that the caller-facing ``(path)`` signature
    is preserved.
    """
    return _load(path.parent, "", path.stem)


def get_reference_dataset(
    dataset: str,
    split: Literal["train", "val", "test"] = "train",
    num_graphs: int = 2048,
) -> Tuple[Any, List[nx.Graph]]:
    """Get reference dataset from polygraph library.

    Returns ``(dataset_object, graphs)`` so callers can also call
    ``dataset_object.is_valid()``.
    """
    from polygraph.datasets.lobster import ProceduralLobsterGraphDataset
    from polygraph.datasets.planar import ProceduralPlanarGraphDataset
    from polygraph.datasets.sbm import ProceduralSBMGraphDataset

    procedural = {
        "planar": ProceduralPlanarGraphDataset,
        "lobster": ProceduralLobsterGraphDataset,
        "sbm": ProceduralSBMGraphDataset,
    }
    if dataset not in procedural:
        raise ValueError(f"Unknown dataset: {dataset}")
    ds = procedural[dataset](split=split, num_graphs=num_graphs)
    return ds, list(ds.to_nx())


@hydra.main(
    config_path="../configs",
    config_name="03_model_quality",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Compute PGD, MMD, and validity for all checkpoints."""
    tabpfn_weights_version: str = cfg.get("tabpfn_weights_version", "v2.5")
    RESULTS_DIR = _RESULTS_DIR_BASE / "results"
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
    iteration_dir = (
        base_dir / dataset if (base_dir / dataset).exists() else base_dir
    )

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
        len(pkl_files),
        curve_type,
        dataset,
        variant,
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

    classifier = make_tabpfn_classifier(tabpfn_weights_version)

    pgd_metric = StandardPGD(
        reference_graphs=ref,
        variant=cast(Literal["informedness", "jsd"], variant),
        classifier=classifier,
    )

    mmd_metrics: dict[str, Any] = {}
    if not pgd_only:
        bw = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])
        mmd_metrics = {
            "orbit_mmd": RBFOrbitMMD2(ref),
            "orbit5_mmd": MaxDescriptorMMD2(
                ref,
                kernel=AdaptiveRBFKernel(
                    descriptor_fn=OrbitCounts(graphlet_size=5),
                    bw=bw,
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

        entry: dict[str, object] = {"steps": steps}

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
                    logger.error(
                        "MMD {} error at step {}: {}", mmd_name, steps, e
                    )

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
