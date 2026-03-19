"""Shared graph loading and reference dataset utilities.

All reproducibility scripts need to load generated graphs from pickle files
and reference datasets from the polygraph library.  This module provides
the canonical implementations to avoid duplication across experiments.

Usage (from any reproducibility script)::

    from utils.data import load_graphs, get_reference_dataset
"""

import pickle
from pathlib import Path
from typing import List

import networkx as nx
import torch
from loguru import logger


def load_graphs(data_dir: Path, model: str, dataset: str) -> List[nx.Graph]:
    """Load model-generated graphs from ``data_dir/{model}/{dataset}.pkl``.

    Handles networkx Graph objects and ``(node_feat, adjacency)`` tuples
    produced by various generators.  Self-loops are removed.

    Returns an empty list (with a warning) when the pickle file is missing.
    """
    pkl_path = data_dir / model / f"{dataset}.pkl"
    if not pkl_path.exists():
        logger.warning("{} not found", pkl_path)
        return []

    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)

    cleaned: List[nx.Graph] = []
    for g in graphs:
        if isinstance(g, nx.Graph):
            simple = nx.Graph(g)
        elif isinstance(g, (list, tuple)) and len(g) == 2:
            try:
                _node_feat, adj = g
                if isinstance(adj, torch.Tensor):
                    adj = adj.numpy()
                simple = nx.from_numpy_array(adj)
            except Exception as e:
                logger.warning("Could not convert graph: {}", e)
                continue
        else:
            logger.warning("Unknown graph format: {}", type(g))
            continue

        simple.remove_edges_from(nx.selfloop_edges(simple))
        cleaned.append(simple)
    return cleaned


def get_reference_dataset(
    dataset: str,
    split: str = "test",
    num_graphs: int = 4096,
) -> List[nx.Graph]:
    """Load a reference dataset from the polygraph library.

    Supports procedural datasets (planar, lobster, sbm) and real datasets
    (proteins, ego).  For real datasets *num_graphs* is ignored (the full
    split is returned).
    """
    from polygraph.datasets.lobster import ProceduralLobsterGraphDataset
    from polygraph.datasets.planar import ProceduralPlanarGraphDataset
    from polygraph.datasets.sbm import ProceduralSBMGraphDataset

    procedural = {
        "planar": ProceduralPlanarGraphDataset,
        "lobster": ProceduralLobsterGraphDataset,
        "sbm": ProceduralSBMGraphDataset,
    }

    if dataset in procedural:
        return list(
            procedural[dataset](split=split, num_graphs=num_graphs).to_nx()
        )

    if dataset == "proteins":
        from polygraph.datasets.proteins import DobsonDoigGraphDataset

        return list(DobsonDoigGraphDataset(split=split).to_nx())

    if dataset == "ego":
        from polygraph.datasets.ego import EgoGraphDataset

        graphs = list(EgoGraphDataset(split=split).to_nx())
        for g in graphs:
            g.remove_edges_from(nx.selfloop_edges(g))
        return graphs

    raise ValueError(f"Unknown dataset: {dataset}")


def make_tabpfn_classifier(weights_version: str = "v2.5"):
    """Create a TabPFN classifier for the given weights version."""
    from tabpfn import TabPFNClassifier
    from tabpfn.classifier import ModelVersion

    version_map = {
        "v2": ModelVersion.V2,
        "v2.5": ModelVersion.V2_5,
    }
    if weights_version not in version_map:
        raise ValueError(
            f"Unknown weights_version: {weights_version!r}. "
            f"Must be one of {list(version_map)}"
        )
    return TabPFNClassifier.create_default_for_version(
        version_map[weights_version],
        device="auto",
        n_estimators=4,
    )
