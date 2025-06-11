import pickle as pkl
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json

from polygraph.datasets import ProceduralPlanarGraphDataset
from polygraph.metrics.base.classifier_metric import (
    AggregateLogisticRegressionClassifierMetric,
)
from polygraph.utils.graph_descriptors import (
    OrbitCounts,
    SparseDegreeHistogram,
    EigenvalueHistogram,
    ClusteringHistogram,
    RandomGIN,
)

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if not args.plot_only:
        ds = ProceduralPlanarGraphDataset("reference", num_graphs=1024)
        reference = ds.to_nx()
        descriptors = {
            "orbit": OrbitCounts(),
            "degree": SparseDegreeHistogram(),
            "spectral": EigenvalueHistogram(),
            "clustering": ClusteringHistogram(100),
            "gin": RandomGIN(seed=42),
        }
        metric = AggregateLogisticRegressionClassifierMetric(
            reference, descriptors, "informedness"
        )

        results = {
            "orbit": [],
            "degree": [],
            "spectral": [],
            "clustering": [],
            "gin": [],
            "polyscore": [],
            "validity": [],
            "num_steps": [],
        }

        for num_steps in [15, 30, 45, 60, 75]:
            with open(
                f"/fs/pool/pool-mlsb/polygraph/digress-iterations/{num_steps}_steps.pkl",
                "rb",
            ) as f:
                data = pkl.load(f)
            data = [nx.from_numpy_array(d[1].numpy()) for d in data]
            assert len(data) == 1024
            eval = metric.compute(data)

            for key, val in eval["subscores"].items():
                results[key].append(val)
            results["num_steps"].append(num_steps)
            results["polyscore"].append(eval["polyscore"])
            results["validity"].append(
                sum(ds.is_valid(d) for d in data) / len(data)
            )

        results = pd.DataFrame(results)
        results.to_csv(
            "/fs/pool/pool-mlsb/polygraph/digress-iterations/results.csv",
            index=False,
        )

    results = pd.read_csv(
        "/fs/pool/pool-mlsb/polygraph/digress-iterations/results.csv"
    )
    with open("/fs/pool/pool-mlsb/polygraph/rcparams.json", "r") as f:
        style = json.load(f)

    plt.rcParams.update(style)
    plt.figure(figsize=(3, 2))
    sns.set_palette("colorblind")

    ax1 = plt.gca()
    color1 = sns.color_palette("colorblind")[0]
    ax1.set_xlabel("Number of Steps")
    ax1.set_ylabel("Validity", color=color1)
    line1 = ax1.plot(
        results["num_steps"],
        results["validity"],
        label="Validity",
        color=color1,
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))

    # Create secondary y-axis and plot polyscore
    ax2 = ax1.twinx()
    color2 = sns.color_palette("colorblind")[1]
    ax2.set_ylabel("PolyGraphScore", color=color2)
    line2 = ax2.plot(
        results["num_steps"],
        results["polyscore"],
        label="PolyGraphScore",
        color=color2,
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))

    plt.savefig(
        "/fs/pool/pool-mlsb/polygraph/digress-iterations/validity_polyscore.pdf"
    )
    plt.close()
