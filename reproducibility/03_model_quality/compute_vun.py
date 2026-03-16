#!/usr/bin/env python3
"""Compute VUN metrics for DiGress denoising-iteration checkpoints.

Patches existing result JSONs (produced by compute.py) with per-step VUN values.
Uses parallel isomorphism checking with a per-pair timeout.

Usage:
    # Single run (defaults: denoising, planar)
    python compute_vun.py

    # Submit to SLURM via Hydra multirun
    python compute_vun.py --multirun hydra/launcher=slurm_cpu_hpcl94c
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Literal

import hydra
import networkx as nx
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here

from polygraph.utils.io import (
    maybe_append_reproducibility_jsonl as maybe_append_jsonl,
)

REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
_RESULTS_DIR_BASE = (
    REPO_ROOT / "reproducibility" / "figures" / "03_model_quality"
)

# Ensure reproducibility/utils is importable
sys.path.insert(0, str(REPO_ROOT / "reproducibility"))
from utils.vun import compute_vun_parallel  # noqa: E402


# ---------------------------------------------------------------------------
# Graph loading (mirrors compute.py)
# ---------------------------------------------------------------------------


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


def get_reference_dataset(
    dataset: str,
    split: Literal["train", "val", "test"] = "train",
    num_graphs: int = 2048,
):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="../configs",
    config_name="03_model_quality_vun",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Compute VUN for all denoising-iteration checkpoints and patch result JSONs."""
    results_suffix: str = cfg.get("results_suffix", "")
    RESULTS_DIR = _RESULTS_DIR_BASE / f"results{results_suffix}"

    dataset: str = cfg.dataset
    iso_timeout: int = cfg.get("iso_timeout", 10)
    n_workers: int = cfg.get("n_workers", 8)
    num_graphs: int = cfg.get("num_graphs", 2048)
    force: bool = cfg.get("force", False)

    # VUN is metric-independent; we patch all variant JSONs with the same values
    variants = ["jsd", "informedness"]

    # Check if VUN already computed for all variants
    if not force:
        all_done = True
        for variant in variants:
            json_path = RESULTS_DIR / f"denoising_{dataset}_{variant}.json"
            if json_path.exists():
                data = json.loads(json_path.read_text())
                if not all("vun" in r for r in data.get("results", [])):
                    all_done = False
                    break
            else:
                all_done = False
                break
        if all_done:
            logger.info(
                "VUN already present for all variants of denoising/{}, skipping",
                dataset,
            )
            return

    # Locate checkpoint files
    base_dir = DATA_DIR / "DIGRESS" / "denoising-iterations"
    iteration_dir = (
        base_dir / dataset if (base_dir / dataset).exists() else base_dir
    )

    if not iteration_dir.exists():
        logger.warning("Iteration directory not found: {}", iteration_dir)
        maybe_append_jsonl(
            {
                "experiment": "03_model_quality",
                "script": "compute_vun.py",
                "dataset": dataset,
                "status": "skipped",
                "reason": "iteration_dir_missing",
            }
        )
        return

    def _parse_steps(p: Path) -> int:
        parts = p.stem.split("_")
        try:
            return int(parts[0])
        except ValueError:
            return int(parts[-1])

    pkl_files = sorted(iteration_dir.glob("*.pkl"), key=_parse_steps)
    if not pkl_files:
        logger.warning("No checkpoint files found in {}", iteration_dir)
        maybe_append_jsonl(
            {
                "experiment": "03_model_quality",
                "script": "compute_vun.py",
                "dataset": dataset,
                "status": "skipped",
                "reason": "no_checkpoint_files",
            }
        )
        return

    logger.info(
        "Computing VUN for {} checkpoints, dataset={}", len(pkl_files), dataset
    )

    # Load training set for novelty checking
    _, train_graphs = get_reference_dataset(
        dataset, split="train", num_graphs=num_graphs
    )
    logger.info(
        "Loaded {} training graphs for novelty checking", len(train_graphs)
    )

    # Compute VUN per checkpoint
    vun_by_steps: Dict[int, Dict[str, float]] = {}
    for pkl_path in pkl_files:
        steps = _parse_steps(pkl_path)
        logger.info("Processing {} steps...", steps)

        graphs = load_graphs(pkl_path)
        if not graphs:
            continue
        gen = graphs[:num_graphs]

        try:
            vun_result = compute_vun_parallel(
                train_graphs,
                gen,
                dataset=dataset,
                iso_timeout=iso_timeout,
                n_workers=n_workers,
            )
            vun_by_steps[steps] = vun_result
            logger.success(
                "  Steps {}: VUN={:.4f} (V={:.4f}, U={:.4f}, N={:.4f})",
                steps,
                vun_result["valid_unique_novel"],
                vun_result["valid"],
                vun_result["unique"],
                vun_result["novel"],
            )
        except Exception as e:
            logger.error("VUN error at step {}: {}", steps, e)

    if not vun_by_steps:
        logger.warning("No VUN results computed")
        maybe_append_jsonl(
            {
                "experiment": "03_model_quality",
                "script": "compute_vun.py",
                "dataset": dataset,
                "status": "skipped",
                "reason": "no_results_computed",
            }
        )
        return

    # Patch all variant result JSONs
    for variant in variants:
        json_path = RESULTS_DIR / f"denoising_{dataset}_{variant}.json"
        if not json_path.exists():
            logger.warning("Result file not found: {}", json_path)
            continue

        data = json.loads(json_path.read_text())
        patched = 0
        for entry in data.get("results", []):
            steps = entry["steps"]
            if steps in vun_by_steps:
                entry["vun"] = vun_by_steps[steps]["valid_unique_novel"]
                entry["vun_valid"] = vun_by_steps[steps]["valid"]
                entry["vun_unique"] = vun_by_steps[steps]["unique"]
                entry["vun_novel"] = vun_by_steps[steps]["novel"]
                patched += 1

        data["vun_computed"] = True
        json_path.write_text(json.dumps(data, indent=2))
        logger.success(
            "Patched {}/{} entries in {}",
            patched,
            len(data.get("results", [])),
            json_path,
        )

    maybe_append_jsonl(
        {
            "experiment": "03_model_quality",
            "script": "compute_vun.py",
            "dataset": dataset,
            "status": "ok",
            "num_steps": len(vun_by_steps),
        }
    )


if __name__ == "__main__":
    main()
