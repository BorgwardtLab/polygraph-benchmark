"""benchmark.py

Here we want to benchmark the performance of the model on the different datasets.
"""

import itertools
import pickle
import sys

import graph_tool.all as gt  # noqa: F401
import numpy as np
import pandas as pd
import typer
from loguru import logger

from polygraph.datasets.lobster import (
    ProceduralLobsterGraphDataset,
    is_lobster_graph,
)
from polygraph.datasets.planar import (
    ProceduralPlanarGraphDataset,
    is_planar_graph,
)
from polygraph.datasets.point_clouds import PointCloudGraphDataset
from polygraph.datasets.proteins import (
    DobsonDoigGraphDataset,
)
from polygraph.datasets.sbm import ProceduralSBMGraphDataset, is_sbm_graph
from polygraph.metrics.base import VUN
from polygraph.metrics.base.mmd import MaxDescriptorMMD2Interval
from polygraph.utils.graph_descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    EigenvalueHistogram,
    OrbitCounts,
)
from polygraph.utils.kernels import RBFKernel
from polygraph.utils.parallel import distribute_function

# Configure loguru logger
logger.remove()  # Remove existing handlers
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

app = typer.Typer()

GRAN_ROOT = "/fs/pool/pool-hartout/Documents/Git/GRAN/exp/GRAN/"
DIGRESS_ROOT = (
    "/fs/pool/pool-hartout/Documents/Git/polygraph/data/digress/converted/"
)


def _load_gran_graphs(dataset_name):
    """Load GRAN generated graphs for a specific dataset"""
    gran_paths = {
        "SBM": "sbm_procedural/generated_graphs_ggg_sbm_procedural/gen_graphs_1040.pkl",
        "PLANAR": "planar_procedural/generated_graphs_ggg_planar_procedural/gen_graphs_1040.pkl",
        "LOBSTER": "lobster_procedural/generated_graphs_ggg_lobster_procedural/gen_graphs_1040.pkl",
    }

    if dataset_name not in gran_paths:
        raise ValueError(f"Invalid dataset for GRAN: {dataset_name}")

    file_path = GRAN_ROOT + gran_paths[dataset_name]
    return pickle.load(open(file_path, "rb"))


def _load_digress_graphs(dataset_name):
    """Load DiGraphs generated graphs for a specific dataset"""
    digress_paths = {
        "SBM": "sbm-procedural/epoch_1499.nx.pkl",
        "PLANAR": "planar-procedural/epoch_3479.nx.pkl",
        "LOBSTER": "lobster-procedural/epoch_989.nx.pkl",
        "PROTEINS": "dobson-doig/epoch_4499.nx.pkl",
    }

    if dataset_name not in digress_paths:
        raise ValueError(f"Invalid dataset for DIGRESS: {dataset_name}")

    file_path = DIGRESS_ROOT + digress_paths[dataset_name]
    return pickle.load(open(file_path, "rb"))


def _load_autograph_graphs(dataset_name):
    autograph_root = (
        "/fs/pool/pool-hartout/Documents/Git/AutoGraph/generated_graphs/"
    )
    autograph_paths = {
        "PLANAR": "planar_procedural.pkl",
        "POINTCLOUD": "pointcloud.pkl",
        "PROTEINS": "proteins.pkl",
        # "LOBSTER": "logs/train/polygraph_lobster_procedural/llama2-s/0/runs/2025-06-22_19-19-32/generated_graphs.pkl",
        # "SBM": "logs/train/polygraph_sbm_procedural/llama2-s/0/runs/2025-06-22_19-19-32/generated_graphs.pkl",
    }

    if dataset_name not in autograph_paths:
        raise ValueError(f"Invalid dataset for AutoGraph: {dataset_name}")

    file_path = autograph_root + autograph_paths[dataset_name]
    graphs = pickle.load(open(file_path, "rb"))

    graphs = [g.to_undirected() for g in graphs]

    return graphs


def get_generated_graphs(model_name, dataset_name, debug):
    """Get generated graphs for a specific model and dataset"""
    if model_name == "GRAN":
        generated_graphs = _load_gran_graphs(dataset_name)
    elif model_name == "DIGRESS":
        generated_graphs = _load_digress_graphs(dataset_name)
    elif model_name == "AUTOGRAPH":
        generated_graphs = _load_autograph_graphs(dataset_name)
    else:
        raise ValueError(f"Invalid model: {model_name}")

    if debug:
        return generated_graphs[:100]
    else:
        return generated_graphs


def get_dataset(dataset_name, model_name=None, debug=False):
    """Get the dataset for a given dataset name"""

    generated_graphs = get_generated_graphs(model_name, dataset_name, debug)

    if dataset_name == "SBM":
        train_set = ProceduralSBMGraphDataset(
            split="train",
            num_graphs=8192 if not debug else 100,
            show_generation_progress=True,
        ).to_nx()
        test_set = ProceduralSBMGraphDataset(
            split="test",
            num_graphs=1024 if not debug else 100,
            show_generation_progress=True,
        ).to_nx()
        return train_set, test_set, generated_graphs

    elif dataset_name == "PLANAR":
        train_set = ProceduralPlanarGraphDataset(
            split="train",
            num_graphs=8192 if not debug else 100,
            show_generation_progress=True,
        ).to_nx()
        test_set = ProceduralPlanarGraphDataset(
            split="test",
            num_graphs=1024 if not debug else 100,
            show_generation_progress=True,
        ).to_nx()
        return train_set, test_set, generated_graphs

    elif dataset_name == "LOBSTER":
        train_set = ProceduralLobsterGraphDataset(
            split="train",
            num_graphs=8192 if not debug else 100,
            show_generation_progress=True,
        ).to_nx()
        test_set = ProceduralLobsterGraphDataset(
            split="test",
            num_graphs=1024 if not debug else 100,
            show_generation_progress=True,
        ).to_nx()
        return train_set, test_set, generated_graphs

    elif dataset_name == "PROTEINS":
        train_set = DobsonDoigGraphDataset(
            split="train",
        ).to_nx()
        test_set = DobsonDoigGraphDataset(
            split="test",
        ).to_nx()
        return train_set, test_set, generated_graphs

    elif dataset_name == "POINTCLOUD":
        train_set = PointCloudGraphDataset(
            split="train",
        ).to_nx()
        test_set = PointCloudGraphDataset(
            split="test",
        ).to_nx()
        return train_set, test_set, generated_graphs
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")


def get_mmd_metric(metric, test_dataset):
    """Get the MMD metric for a given dataset"""
    bws = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

    if metric == "MMD_DEGREE":
        return MaxDescriptorMMD2Interval(
            reference_graphs=test_dataset,
            kernel=RBFKernel(DegreeHistogram(max_degree=1000), bw=bws),
            variant="umve",
        )
    elif metric == "MMD_CLUSTERING":
        return MaxDescriptorMMD2Interval(
            reference_graphs=test_dataset,
            kernel=RBFKernel(
                ClusteringHistogram(bins=1000, sparse=True), bw=bws
            ),
            variant="umve",
        )
    elif metric == "MMD_ORBIT":
        return MaxDescriptorMMD2Interval(
            reference_graphs=test_dataset,
            kernel=RBFKernel(OrbitCounts(graphlet_size=4), bw=bws),
            variant="umve",
        )
    elif metric == "MMD_SPECTRE":
        return MaxDescriptorMMD2Interval(
            reference_graphs=test_dataset,
            kernel=RBFKernel(EigenvalueHistogram(sparse=True), bw=bws),
            variant="umve",
        )
    else:
        raise ValueError(
            f"Invalid metric: {metric} for dataset: {test_dataset}"
        )


def get_vun_metric(train_dataset, dataset_name):
    """Get the VUN metric for a given dataset"""
    if dataset_name == "SBM":
        validity_fn = is_sbm_graph
    elif dataset_name == "PLANAR":
        validity_fn = is_planar_graph
    elif dataset_name == "LOBSTER":
        validity_fn = is_lobster_graph
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    return VUN(
        train_graphs=train_dataset,
        validity_fn=validity_fn,
    )


def get_metric(
    metric_name,
    train_dataset,
    test_dataset,
    model_generated_graphs,
    dataset_name,
):
    if metric_name == "VUN":
        return get_vun_metric(train_dataset, dataset_name)
    elif "MMD" in metric_name:
        return get_mmd_metric(metric_name, test_dataset)
    else:
        raise ValueError(f"Invalid metric: {metric_name}")


def compute_metrics_for_model(parameters, subsample_size, num_samples, debug):
    model_name, dataset_name, metric_name = parameters
    try:
        train_set, test_set, model_generated_graphs = get_dataset(
            dataset_name, model_name, debug
        )
    except Exception:
        result = {
            "model": model_name,
            "dataset": dataset_name,
            "metric": metric_name,
            "error": "Error loading generated graphs",
        }
        return None

    metric = get_metric(
        metric_name, train_set, test_set, model_generated_graphs, dataset_name
    )

    if "MMD" in metric_name:
        result = metric.compute(
            generated_graphs=model_generated_graphs,
            subsample_size=subsample_size,
            num_samples=num_samples,
            as_scalar_value_dict=True,
        )
    else:
        result = metric.compute(
            generated_graphs=model_generated_graphs,
            as_scalar_value_dict=True,
        )
    result["model"] = model_name
    result["dataset"] = dataset_name
    result["metric"] = metric_name
    result["error"] = ""
    return result


def normalize_result_keys(result_list):
    """
    Add missing keys to result dictionaries to ensure consistent structure.

    Args:
        result_list: List of dictionaries containing metric computation results

    Returns:
        List of dictionaries with consistent keys, missing keys filled with None
    """
    if not result_list:
        return result_list

    # Filter out None results (from errors)
    valid_results = [r for r in result_list if r is not None]

    if not valid_results:
        return []

    # Collect all unique keys from all dictionaries
    all_keys = set()
    for result_dict in valid_results:
        all_keys.update(result_dict.keys())

    # Add missing keys to each dictionary with appropriate default values
    normalized_results = []
    for result_dict in valid_results:
        normalized_dict = result_dict.copy()

        for key in all_keys:
            if key not in normalized_dict:
                # Set appropriate default values based on key type
                if key == "error":
                    normalized_dict[key] = ""
                elif key in ["model", "dataset", "metric"]:
                    normalized_dict[key] = "unknown"
                else:
                    # For numeric metric values, use None (will become NaN in pandas)
                    normalized_dict[key] = None

        normalized_results.append(normalized_dict)

    return normalized_results


@app.command()
def main(
    dataset_names: list[str] = typer.Option(
        ["PLANAR", "LOBSTER", "SBM", "PROTEINS", "POINTCLOUD"],
        "--dataset-names",
        "-d",
        help="Dataset names",
    ),
    model_names: list[str] = typer.Option(
        ["GRAN", "DIGRESS", "AUTOGRAPH"],
        "--model-names",
        "-m",
        help="Model names",
    ),
    metric_names: list[str] = typer.Option(
        ["VUN", "MMD_DEGREE", "MMD_CLUSTERING", "MMD_ORBIT", "MMD_SPECTRE"],
        "--metric-names",
        "-t",
        help="Metric names",
    ),
    subsample_size: int = typer.Option(
        1024,
        "--subsample-size",
        "-s",
        help="Subsample size for MMD calculations",
    ),
    num_samples: int = typer.Option(
        100,
        "--num-samples",
        "-n",
        help="Number of samples for MMD calculations",
    ),
    n_jobs: int = typer.Option(
        20, "--n-jobs", "-j", help="Number of parallel jobs"
    ),
    debug: bool = typer.Option(False, "--debug", "-b", help="Debug mode"),
):
    parameters = list(
        itertools.product(
            model_names,
            dataset_names,
            ["VUN", "MMD_DEGREE", "MMD_CLUSTERING", "MMD_ORBIT", "MMD_SPECTRE"],
        )
    )
    result = distribute_function(
        compute_metrics_for_model,
        X=parameters,
        subsample_size=subsample_size,
        num_samples=num_samples,
        n_jobs=n_jobs,
        debug=debug,
        show_progress=not debug,
    )
    result = normalize_result_keys(result)
    result = pd.DataFrame(result)
    result.to_csv(
        "./experiments/model_benchmark/results/results.csv", index=False
    )


if __name__ == "__main__":
    app()
