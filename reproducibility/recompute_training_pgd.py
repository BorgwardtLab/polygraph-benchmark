#!/usr/bin/env python3
"""Recompute PGD for v6 training results where it's missing.

Patches the existing results_tabpfn_v6/training_*.json files with
polyscore and *_pgs fields, then copies them to results/ as well.
"""

import json
import pickle
import shutil

import networkx as nx
import numpy as np
from loguru import logger
from pyprojroot import here
from importlib.metadata import version as pkg_version

REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
RESULTS_V6_DIR = (
    REPO_ROOT
    / "reproducibility"
    / "figures"
    / "03_model_quality"
    / "results_tabpfn_v6"
)
RESULTS_DIR = (
    REPO_ROOT / "reproducibility" / "figures" / "03_model_quality" / "results"
)
CHECKPOINT_DIR = DATA_DIR / "DIGRESS" / "training-iterations"

DATASETS = ["planar", "sbm", "lobster"]
VARIANTS = ["jsd", "informedness"]


def load_graphs(path):
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
    return list(ds.to_nx())


def main():
    tabpfn_version = pkg_version("tabpfn")
    logger.info("TabPFN version: {}", tabpfn_version)

    for dataset in DATASETS:
        logger.info("Loading reference graphs for {}", dataset)
        ref_graphs = get_reference_dataset(dataset, num_graphs=1024)

        for variant in VARIANTS:
            v6_path = RESULTS_V6_DIR / f"training_{dataset}_{variant}.json"
            if not v6_path.exists():
                logger.warning("Missing: {}", v6_path)
                continue

            with open(v6_path) as f:
                data = json.load(f)

            # Check if polyscore already exists
            if data["results"] and "polyscore" in data["results"][0]:
                logger.info("Already has polyscore: {}", v6_path.name)
                continue

            logger.info("Computing PGD for {} {} ...", dataset, variant)

            from polygraph.metrics import StandardPGD

            pgd_metric = StandardPGD(
                reference_graphs=ref_graphs,
                variant=variant,
                classifier=None,
            )

            for entry in data["results"]:
                steps = entry["steps"]
                pkl_path = CHECKPOINT_DIR / f"{steps}_steps.pkl"

                if not pkl_path.exists():
                    logger.warning("Checkpoint not found: {}", pkl_path)
                    continue

                logger.info("  Step {} ...", steps)
                graphs = load_graphs(pkl_path)
                gen = graphs[:1024]

                try:
                    pgd_result = pgd_metric.compute(gen)
                    entry["polyscore"] = pgd_result["pgd"]
                    for key, val in pgd_result["subscores"].items():
                        entry[f"{key}_pgs"] = val
                    logger.info("    PGD = {:.4f}", pgd_result["pgd"])
                except Exception as e:
                    logger.error("    PGD error at step {}: {}", steps, e)
                    import traceback

                    traceback.print_exc()

            data["tabpfn_package_version"] = tabpfn_version

            # Save back to v6 dir
            v6_path.write_text(json.dumps(data, indent=2))
            logger.success("Updated: {}", v6_path)

            # Also copy to default results dir
            default_path = RESULTS_DIR / f"training_{dataset}_{variant}.json"
            shutil.copy2(v6_path, default_path)
            logger.success("Copied to: {}", default_path)

    logger.success("Done. TabPFN version: {}", tabpfn_version)


if __name__ == "__main__":
    main()
