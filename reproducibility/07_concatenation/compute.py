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
from importlib.metadata import version as pkg_version
from typing import Dict, Iterable, List, Optional

import hydra
import networkx as nx
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
_RESULTS_DIR_BASE = REPO_ROOT / "reproducibility" / "tables" / "results"


def load_graphs(model: str, dataset: str) -> List:
    return _load(DATA_DIR, model, dataset)


def get_reference_dataset(dataset: str, split: str = "test"):
    return _get_ref(dataset, split=split, num_graphs=4096)


class ConcatenatedDescriptor:
    """Descriptor that concatenates multiple descriptor outputs, optionally with PCA.

    Matches the original polygraph CombinedDescriptor: per-descriptor
    StandardScaler + optional PCA, both fit on the first call (reference data).

    Set ``max_features=None`` to skip PCA (e.g. when using TabPFN v2.5 which
    handles high-dimensional inputs natively).
    """

    def __init__(
        self,
        descriptors: Dict,
        max_features: Optional[int] = 500,
        dataset_size: Optional[int] = None,
    ):
        self.descriptors = descriptors
        self._use_pca = max_features is not None
        if max_features is not None:
            n_components = max_features
            if dataset_size is not None:
                n_components = min(max_features, dataset_size)
            self._pca = PCA(n_components=n_components)
        self._scalers: Dict[str, StandardScaler] = {}
        self._fitted = False

    def __call__(self, graphs: Iterable[nx.Graph]) -> np.ndarray:
        graphs = list(graphs)
        features = []
        for name, desc in self.descriptors.items():
            feat = desc(graphs)
            if issparse(feat):
                feat = feat.toarray()
            if not self._fitted:
                scaler = StandardScaler()
                feat = scaler.fit_transform(feat)
                self._scalers[name] = scaler
            else:
                feat = self._scalers[name].transform(feat)
            features.append(feat)

        concatenated = np.concatenate(features, axis=1)
        if not self._use_pca:
            self._fitted = True
            return concatenated
        if not self._fitted:
            self._pca.fit(concatenated)
            self._fitted = True
        return self._pca.transform(concatenated)


def compute_pgs_standard(
    reference_graphs: List,
    generated_graphs: List,
    subset: bool = False,
    classifier=None,
) -> Dict:
    """Compute standard PGD (max over individual descriptors, TabPFN classifier)."""
    from polygraph.metrics import StandardPGDInterval

    if subset:
        reference_graphs = reference_graphs[:30]
        generated_graphs = generated_graphs[:30]
        subsample_size = 15
        num_samples = 3
    else:
        min_subset_size = min(len(reference_graphs), len(generated_graphs))
        subsample_size = min(int(min_subset_size * 0.5), 2048)
        num_samples = 10

    metric = StandardPGDInterval(
        reference_graphs,
        subsample_size=subsample_size,
        num_samples=num_samples,
        classifier=classifier,
    )
    result = metric.compute(generated_graphs)

    return {
        "pgs_standard_mean": result["pgd"].mean,
        "pgs_standard_std": result["pgd"].std,
    }


def compute_pgs_concatenated(
    reference_graphs: List,
    generated_graphs: List,
    subset: bool = False,
    classifier=None,
    use_pca: bool = True,
) -> Dict:
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
        min_subset_size = min(len(reference_graphs), len(generated_graphs))
        subsample_size = min(int(min_subset_size * 0.5), 2048)
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
    dataset_size = min(len(reference_graphs), len(generated_graphs))
    concat_desc = ConcatenatedDescriptor(
        desc_dict,
        max_features=500 if use_pca else None,
        dataset_size=dataset_size,
    )

    metric = PolyGraphDiscrepancyInterval(
        reference_graphs,
        descriptors={"concatenated": concat_desc},
        subsample_size=subsample_size,
        num_samples=num_samples,
        variant="jsd",
        classifier=classifier,
    )

    result = metric.compute(generated_graphs)

    return {
        "pgs_concatenated_mean": result["pgd"].mean,
        "pgs_concatenated_std": result["pgd"].std,
    }


@hydra.main(
    config_path="../configs", config_name="07_concatenation", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Compute concatenation metrics for one (dataset, model) pair and save as JSON."""
    tabpfn_weights_version: str = cfg.get("tabpfn_weights_version", "v2.5")
    RESULTS_DIR = _RESULTS_DIR_BASE / "concatenation"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset
    model = cfg.model
    subset = cfg.subset

    from tabpfn import TabPFNClassifier
    from tabpfn.classifier import ModelVersion

    version_map = {
        "v2": ModelVersion.V2,
        "v2.5": ModelVersion.V2_5,
    }
    if tabpfn_weights_version not in version_map:
        raise ValueError(
            f"Unknown tabpfn_weights_version: {tabpfn_weights_version!r}. Must be one of {list(version_map)}"
        )
    classifier = TabPFNClassifier.create_default_for_version(
        version_map[tabpfn_weights_version],
        device="auto",
        n_estimators=4,
    )

    # V2.5 handles high-dimensional inputs natively; skip PCA
    use_pca = tabpfn_weights_version != "v2.5"

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

    generated_graphs = generated_graphs[: len(reference_graphs)]
    result: Dict = {
        "dataset": dataset,
        "model": model,
        "tabpfn_package_version": pkg_version("tabpfn"),
        "tabpfn_weights_version": tabpfn_weights_version,
    }

    try:
        std_results = compute_pgs_standard(
            reference_graphs,
            generated_graphs,
            subset=subset,
            classifier=classifier,
        )
        result.update(std_results)
    except Exception as e:
        logger.error(
            "Error computing standard PGD for {}/{}: {}", model, dataset, e
        )

    try:
        cat_results = compute_pgs_concatenated(
            reference_graphs,
            generated_graphs,
            subset=subset,
            classifier=classifier,
            use_pca=use_pca,
        )
        result.update(cat_results)
    except Exception as e:
        logger.error(
            "Error computing concatenated PGD for {}/{}: {}", model, dataset, e
        )

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
