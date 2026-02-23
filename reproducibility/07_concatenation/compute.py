#!/usr/bin/env python3
"""Compute concatenation ablation metrics (standard vs concatenated PGD).

Usage:
    python compute.py                                              # Single run (default config)
    python compute.py --multirun                                   # All dataset x model combos
    python compute.py --multirun hydra/launcher=slurm_cpu          # On SLURM
    python compute.py --multirun subset=true                       # Quick test
"""

import json
import sys
from typing import Dict, Iterable, List

import hydra
import networkx as nx
import numpy as np
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
RESULTS_DIR = REPO_ROOT / "reproducibility" / "tables" / "results" / "concatenation"


def load_graphs(model: str, dataset: str) -> List:
    return _load(DATA_DIR, model, dataset)


def get_reference_dataset(dataset: str, split: str = "test"):
    return _get_ref(dataset, split=split, num_graphs=4096)


class ConcatenatedDescriptor:
    """Descriptor that concatenates multiple descriptor outputs with PCA reduction."""

    def __init__(self, descriptors: Dict, max_features: int = 500):
        self.descriptors = descriptors
        self.max_features = max_features
        self._pca = None
        self._is_fitted = False

    def __call__(self, graphs: Iterable[nx.Graph]) -> np.ndarray:
        from sklearn.decomposition import PCA

        graphs = list(graphs)
        features = []
        for name, desc in self.descriptors.items():
            feat = desc(graphs)
            if hasattr(feat, 'toarray'):
                feat = feat.toarray()
            features.append(feat)

        concatenated = np.hstack(features)

        if concatenated.shape[1] > self.max_features:
            n_components = min(self.max_features, concatenated.shape[0] - 1)
            if self._pca is None:
                self._pca = PCA(n_components=n_components)
                concatenated = self._pca.fit_transform(concatenated)
            else:
                concatenated = self._pca.transform(concatenated)

        return concatenated


def compute_pgs_standard(reference_graphs: List, generated_graphs: List, subset: bool = False) -> Dict:
    """Compute standard PGD (max over individual descriptors, TabPFN classifier)."""
    from polygraph.metrics import StandardPGDInterval

    if subset:
        reference_graphs = reference_graphs[:30]
        generated_graphs = generated_graphs[:30]
        subsample_size = 15
        num_samples = 3
    else:
        subsample_size = min(len(reference_graphs), len(generated_graphs)) // 4
        num_samples = 10

    metric = StandardPGDInterval(
        reference_graphs,
        subsample_size=subsample_size,
        num_samples=num_samples,
    )
    result = metric.compute(generated_graphs)

    return {
        "pgs_standard_mean": result["pgd"].mean,
        "pgs_standard_std": result["pgd"].std,
    }


def compute_pgs_concatenated(reference_graphs: List, generated_graphs: List, subset: bool = False) -> Dict:
    """Compute concatenated PGD (all descriptors as one feature vector)."""

    from polygraph.metrics.base import PolyGraphDiscrepancyInterval
    from polygraph.utils.descriptors import (
        ClusteringHistogram,
        EigenvalueHistogram,
        OrbitCounts,
        RandomGIN,
        SparseDegreeHistogram,
    )

    if subset:
        reference_graphs = reference_graphs[:30]
        generated_graphs = generated_graphs[:30]
        subsample_size = 15
        num_samples = 3
    else:
        subsample_size = min(len(reference_graphs), len(generated_graphs)) // 4
        num_samples = 10

    if subset:
        desc_dict = {
            "orbit4": OrbitCounts(graphlet_size=4),
            "degree": SparseDegreeHistogram(),
            "spectral": EigenvalueHistogram(),
            "clustering": ClusteringHistogram(bins=100),
            "gin": RandomGIN(seed=42),
        }
    else:
        desc_dict = {
            "orbit4": OrbitCounts(graphlet_size=4),
            "orbit5": OrbitCounts(graphlet_size=5),
            "degree": SparseDegreeHistogram(),
            "spectral": EigenvalueHistogram(),
            "clustering": ClusteringHistogram(bins=100),
            "gin": RandomGIN(seed=42),
        }
    concat_desc = ConcatenatedDescriptor(desc_dict, max_features=500)

    metric = PolyGraphDiscrepancyInterval(
        reference_graphs,
        descriptors={"concatenated": concat_desc},
        subsample_size=subsample_size,
        num_samples=num_samples,
        variant="jsd",
        classifier=None,
    )

    result = metric.compute(generated_graphs)

    return {
        "pgs_concatenated_mean": result["pgd"].mean,
        "pgs_concatenated_std": result["pgd"].std,
    }


@hydra.main(config_path="../configs", config_name="07_concatenation", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compute concatenation metrics for one (dataset, model) pair and save as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset
    model = cfg.model
    subset = cfg.subset

    logger.info("Computing concatenation ablation for {}/{}", model, dataset)

    try:
        reference_graphs = get_reference_dataset(dataset, split="test")
    except Exception as e:
        logger.error("Error loading reference dataset: {}", e)
        maybe_append_jsonl(
            {
                "experiment": "07_concatenation",
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
                "experiment": "07_concatenation",
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
        std_results = compute_pgs_standard(reference_graphs, generated_graphs, subset=subset)
        result.update(std_results)
    except Exception as e:
        logger.error("Error computing standard PGD for {}/{}: {}", model, dataset, e)

    try:
        cat_results = compute_pgs_concatenated(reference_graphs, generated_graphs, subset=subset)
        result.update(cat_results)
    except Exception as e:
        logger.error("Error computing concatenated PGD for {}/{}: {}", model, dataset, e)

    out_path = RESULTS_DIR / f"{dataset}_{model}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.success("Result saved to {}", out_path)
    maybe_append_jsonl(
        {
            "experiment": "07_concatenation",
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
