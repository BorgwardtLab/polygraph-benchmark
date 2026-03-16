#!/usr/bin/env python3
"""Run GKLR computation for a single (dataset, model) pair.

Usage:
    python run_single.py --dataset planar --model AUTOGRAPH
"""

import argparse
import json
import pickle
from typing import Dict, List

from loguru import logger
from pyprojroot import here

from polygraph.utils.io import (
    maybe_append_reproducibility_jsonl as maybe_append_jsonl,
)

REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "reproducibility" / "tables" / "results" / "gklr"


def load_graphs(model: str, dataset: str) -> List:
    import networkx as nx
    import torch

    pkl_path = DATA_DIR / model / f"{dataset}.pkl"
    if not pkl_path.exists():
        logger.error("{} not found", pkl_path)
        return []
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)

    cleaned = []
    for g in graphs:
        if isinstance(g, nx.Graph):
            simple = nx.Graph(g)
        elif isinstance(g, (list, tuple)) and len(g) == 2:
            try:
                node_feat, adj = g
                if isinstance(adj, torch.Tensor):
                    adj = adj.numpy()
                simple = nx.from_numpy_array(adj)
            except Exception as e:
                logger.warning("Could not convert graph: {}", e)
                continue
        else:
            continue
        simple.remove_edges_from(nx.selfloop_edges(simple))
        cleaned.append(simple)
    return cleaned


def get_reference_dataset(
    dataset: str, split: str = "test", num_graphs: int = 512
):
    from polygraph.datasets.lobster import ProceduralLobsterGraphDataset
    from polygraph.datasets.planar import ProceduralPlanarGraphDataset
    from polygraph.datasets.proteins import DobsonDoigGraphDataset
    from polygraph.datasets.sbm import ProceduralSBMGraphDataset

    if dataset == "planar":
        return list(
            ProceduralPlanarGraphDataset(
                split=split, num_graphs=num_graphs
            ).to_nx()
        )
    elif dataset == "lobster":
        return list(
            ProceduralLobsterGraphDataset(
                split=split, num_graphs=num_graphs
            ).to_nx()
        )
    elif dataset == "sbm":
        return list(
            ProceduralSBMGraphDataset(
                split=split, num_graphs=num_graphs
            ).to_nx()
        )
    elif dataset == "proteins":
        return list(DobsonDoigGraphDataset(split=split).to_nx())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset
    model = args.model

    logger.info("Computing GKLR for {}/{}", model, dataset)

    reference_graphs = get_reference_dataset(dataset, split="test")
    generated_graphs = load_graphs(model, dataset)
    if not generated_graphs:
        logger.error("No graphs found for {}/{}", model, dataset)
        maybe_append_jsonl(
            {
                "experiment": "08_gklr",
                "script": "run_single.py",
                "dataset": dataset,
                "model": model,
                "status": "skipped",
                "reason": "no_generated_graphs",
            }
        )
        return

    generated_graphs = generated_graphs[: len(reference_graphs)]
    logger.info(
        "Reference: {} graphs, Generated: {} graphs",
        len(reference_graphs),
        len(generated_graphs),
    )

    from polygraph.metrics.base import (
        KernelLogisticRegression,
        PolyGraphDiscrepancyInterval,
    )
    from polygraph.utils.descriptors import (
        WeisfeilerLehmanDescriptor,
        ShortestPathHistogramDescriptor,
        PyramidMatchDescriptor,
    )

    subsample_size = min(len(reference_graphs), len(generated_graphs)) // 4
    num_samples = 10

    descriptors = {
        "wl": WeisfeilerLehmanDescriptor(),
        "shortest_path": ShortestPathHistogramDescriptor(),
        "pyramid_match": PyramidMatchDescriptor(),
    }

    classifier = KernelLogisticRegression(max_iter=1000)

    metric = PolyGraphDiscrepancyInterval(
        reference_graphs,
        descriptors=descriptors,
        subsample_size=subsample_size,
        num_samples=num_samples,
        variant="jsd",
        classifier=classifier,
        scale=False,
    )

    result_data = metric.compute(generated_graphs)

    output: Dict = {
        "dataset": dataset,
        "model": model,
        "pgs_mean": result_data["pgd"].mean,
        "pgs_std": result_data["pgd"].std,
    }
    for name, interval in result_data["subscores"].items():
        output[f"{name}_mean"] = interval.mean
        output[f"{name}_std"] = interval.std

    out_path = RESULTS_DIR / f"{dataset}_{model}.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.success("Result saved to {}", out_path)
    maybe_append_jsonl(
        {
            "experiment": "08_gklr",
            "script": "run_single.py",
            "dataset": dataset,
            "model": model,
            "status": "ok",
            "output_path": str(out_path),
            "result": output,
        }
    )


if __name__ == "__main__":
    main()
