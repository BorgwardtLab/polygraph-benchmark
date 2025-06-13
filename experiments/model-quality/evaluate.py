import pickle as pkl
import pandas as pd
import numpy as np
import torch
import networkx as nx
import argparse
from tqdm import tqdm

from polygraph.datasets import ProceduralPlanarGraphDataset
from polygraph.metrics.base import (
    AggregateLogisticRegressionClassifierMetric,
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
        "--experiment",
        type=str,
        default="denoising-iterations",
        choices=["denoising-iterations", "training-iterations"],
    )
    args = parser.parse_args()

    if args.experiment == "denoising-iterations":
        steps = [15, 30, 45, 60, 75]
    else:
        steps = [119, 209, 299, 419, 509, 1019, 1499, 2009, 2519, 2999, 3479]

    ds = ProceduralPlanarGraphDataset("reference", num_graphs=1024)
    reference = ds.to_nx()
    descriptors = {
        "orbit_pgs": OrbitCounts(),
        "degree_pgs": SparseDegreeHistogram(),
        "spectral_pgs": EigenvalueHistogram(),
        "clustering_pgs": ClusteringHistogram(100),
        "gin_pgs": RandomGIN(seed=42),
    }
    metric = AggregateLogisticRegressionClassifierMetric(
        reference, descriptors, "informedness"
    )
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

    for num_steps in tqdm(steps):
        with open(
            f"/fs/pool/pool-mlsb/polygraph/model-quality/{args.experiment}/{num_steps}_steps.pkl",
            "rb",
        ) as f:
            data = pkl.load(f)
        data = [nx.from_numpy_array(d[1].numpy()) for d in data]
        assert len(data) == 1024
        eval = metric.compute(data)

        mmd_eval = mmd.compute(data)

        for key, val in mmd_eval.items():
            results[key].append(val)

        for key, val in eval["subscores"].items():
            results[key].append(val)

        results["num_steps"].append(num_steps)
        results["polyscore"].append(eval["polyscore"])
        results["validity"].append(
            sum(ds.is_valid(d) for d in data) / len(data)
        )

    results = pd.DataFrame(results)
    print(results)
    results.to_csv(
        f"/fs/pool/pool-mlsb/polygraph/model-quality/{args.experiment}/results.csv",
        index=False,
    )
