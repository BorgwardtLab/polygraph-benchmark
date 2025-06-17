import pickle as pkl
import pandas as pd
import os
import networkx as nx
import argparse
from tqdm import tqdm

from polygraph.datasets import (
    DobsonDoigGraphDataset,
    ProceduralLobsterGraphDataset,
    ProceduralPlanarGraphDataset,
    ProceduralSBMGraphDataset,
)
from polygraph.metrics.base import (
    AggregateLogisticRegressionClassifierMetric,
)
from polygraph.utils.graph_descriptors import (
    OrbitCounts,
    SparseDegreeHistogram,
    EigenvalueHistogram,
    ClusteringHistogram,
    RandomGIN,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="dobson-doig",
        choices=[
            "dobson-doig",
            "lobster-procedural",
            "planar-procedural",
            "sbm-procedural",
        ],
    )
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.experiment == "dobson-doig":
        reference = DobsonDoigGraphDataset(
            split="val" if not args.test else "test"
        ).to_nx()
    elif args.experiment == "lobster-procedural":
        reference = ProceduralLobsterGraphDataset(
            split="val" if not args.test else "test", num_graphs=1024
        ).to_nx()
    elif args.experiment == "planar-procedural":
        reference = ProceduralPlanarGraphDataset(
            split="val" if not args.test else "test", num_graphs=1024
        ).to_nx()
    elif args.experiment == "sbm-procedural":
        reference = ProceduralSBMGraphDataset(
            split="val" if not args.test else "test", num_graphs=1024
        ).to_nx()
    else:
        raise ValueError(f"Experiment {args.experiment} not found")

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
        "pgs": [],
        "epoch": [],
        "orbit_mmd": [],
        "degree_mmd": [],
        "spectral_mmd": [],
        "clustering_mmd": [],
        "gin_mmd": [],
    }

    sample_path = os.path.join(
        "/fs/pool/pool-mlsb/polygraph/digress-samples/", args.experiment
    )

    for file in tqdm(list(os.listdir(sample_path))):
        if not file.endswith(".pkl"):
            continue
        epoch = int(file.split("_")[-1].split(".")[0])
        with open(os.path.join(sample_path, file), "rb") as f:
            data = pkl.load(f)
        assert len(data) == 1024
        data = [
            nx.from_numpy_array(d[1].numpy()) for d in data[: len(reference)]
        ]
        eval = metric.compute(data)
        mmd_eval = mmd.compute(data)

        results["epoch"].append(epoch)
        results["pgs"].append(eval["polyscore"])
        for key, value in eval["subscores"].items():
            results[key].append(value)
        for key, value in mmd_eval.items():
            results[key].append(value)

    results = pd.DataFrame(results)
    results = results.sort_values(by="epoch")
    print(results)
    results.to_csv(
        os.path.join(
            sample_path,
            "pgs_validation.csv" if not args.test else "pgs_test.csv",
        ),
        index=False,
    )
