import graph_tool as gt  # noqa: F401

import pickle as pkl
import pandas as pd
import numpy as np
import torch
import networkx as nx
import argparse
from tqdm import tqdm
from pathlib import Path
import os

from polygraph.datasets import (
    ProceduralPlanarGraphDataset,
    ProceduralLobsterGraphDataset,
    ProceduralSBMGraphDataset,
)
from polygraph.metrics.base import (
    AggregateClassifierMetric,
)
from polygraph.metrics.gran import (
    RBFClusteringMMD2,
    RBFOrbitMMD2,
    RBFDegreeMMD2,
    RBFSpectralMMD2,
)
from polygraph.metrics.gin import (
    RBFGraphNeuralNetworkMMD2,
)
from polygraph.utils.graph_descriptors import (
    OrbitCounts,
    SparseDegreeHistogram,
    EigenvalueHistogram,
    ClusteringHistogram,
    RandomGIN,
)


class AggregateMMD:
    def __init__(self, reference):
        self._metrics = {
            "orbit_mmd": RBFOrbitMMD2(reference),
            "degree_mmd": RBFDegreeMMD2(reference),
            "spectral_mmd": RBFSpectralMMD2(reference),
            "clustering_mmd": RBFClusteringMMD2(reference),
            "gin_mmd": RBFGraphNeuralNetworkMMD2(reference),
        }

    def compute(self, data):
        return {
            key: metric.compute(data) for key, metric in self._metrics.items()
        }


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-folder",
        type=str,
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="jsd",
        choices=["jsd", "informedness", "informedness-adaptive"],
    )
    parser.add_argument(
        "--reference",
        type=str,
        choices=["planar", "sbm", "lobster"],
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="results.csv",
    )
    args = parser.parse_args()

    result_path = os.path.join(args.checkpoint_folder, args.filename)

    if os.path.exists(result_path):
        print(f"Result file {result_path} already exists, skipping...")
        exit()

    checkpoints = list(Path(args.checkpoint_folder).glob("*.pkl"))
    steps = []
    for p in checkpoints:
        try:
            steps.append(int(p.stem.split("_")[-1]))
        except ValueError:
            steps.append(int(p.stem.split("_")[-2]))

    perm = np.argsort(steps)
    checkpoints = [checkpoints[i] for i in perm]
    steps = [steps[i] for i in perm]

    if args.reference == "planar":
        ds = ProceduralPlanarGraphDataset(
            "reference", num_graphs=args.num_graphs
        )
    elif args.reference == "sbm":
        ds = ProceduralSBMGraphDataset("reference", num_graphs=args.num_graphs)
    elif args.reference == "lobster":
        ds = ProceduralLobsterGraphDataset(
            "reference", num_graphs=args.num_graphs
        )
    else:
        raise ValueError(f"Invalid reference: {args.reference}")

    reference = ds.to_nx()
    descriptors = {
        "orbit_pgs": OrbitCounts(),
        "degree_pgs": SparseDegreeHistogram(),
        "spectral_pgs": EigenvalueHistogram(),
        "clustering_pgs": ClusteringHistogram(100),
        "gin_pgs": RandomGIN(seed=42),
    }
    metric = AggregateClassifierMetric(reference, descriptors, args.metric)

    mmd = AggregateMMD(reference)

    results = {
        "orbit_pgs": [],
        "degree_pgs": [],
        "spectral_pgs": [],
        "clustering_pgs": [],
        "gin_pgs": [],
        "orbit_mmd": [],
        "degree_mmd": [],
        "spectral_mmd": [],
        "clustering_mmd": [],
        "gin_mmd": [],
        "polyscore": [],
        "validity": [],
        "num_steps": [],
    }

    for ckpt, step in tqdm(zip(checkpoints, steps), total=len(checkpoints)):
        with open(ckpt, "rb") as f:
            data = pkl.load(f)
        data = [nx.from_numpy_array(d[1].numpy()) for d in data]
        assert len(data) == 8192, f"Expected 8192 graphs, got {len(data)}"
        data = data[: args.num_graphs]
        eval_result = metric.compute(data)
        print(eval_result, flush=True)

        mmd_eval = mmd.compute(data)

        for key, val in mmd_eval.items():
            results[key].append(val)

        for key, val in eval_result["subscores"].items():
            results[key].append(val)

        results["num_steps"].append(step)
        results["polyscore"].append(eval_result["polyscore"])
        results["validity"].append(
            sum(ds.is_valid(d) for d in data) / len(data)
        )

    results = pd.DataFrame(results)
    # sort by step
    results = results.sort_values(by="num_steps")
    results.to_csv(
        result_path,
        index=False,
    )
