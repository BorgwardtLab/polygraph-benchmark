import itertools
import os
import pickle
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger
from pyprojroot import here
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from polygraph.datasets import (
    ProceduralLobsterGraphDataset,
    ProceduralPlanarGraphDataset,
    ProceduralSBMGraphDataset,
)
from polygraph.metrics.base.mmd import MaxDescriptorMMD2Interval
from polygraph.utils.graph_descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    EigenvalueHistogram,
    OrbitCounts,
)
from polygraph.utils.kernels import RBFKernel
from polygraph.utils.parallel import distribute_function

app = typer.Typer()

DESCRIPTORS = [
    DegreeHistogram(max_degree=500),
    ClusteringHistogram(bins=100, sparse=False),
    EigenvalueHistogram(n_bins=100, sparse=False),
    OrbitCounts(),
]


def construct_dataset_paths(
    gran_logs_root: str,
    autograph_logs_root: str,
    digress_logs_root: str,
    digress_procedural_root: str,
    gran_filename: str = "gen_graphs_1040.pkl",
) -> dict:
    """
    Construct all dataset paths for different models and datasets.

    Args:
        gran_logs_root: Root directory for GRAN logs
        autograph_logs_root: Root directory for AutoGraph logs
        digress_logs_root: Root directory for DiGress logs
        digress_procedural_root: Root directory for DiGress procedural logs
        gran_filename: Filename for GRAN generated graphs

    Returns:
        Dictionary containing organized paths for all datasets and models
    """
    return {
        "sbm": {
            "gran_generated_procedural": (
                gran_logs_root
                + "sbm_procedural/generated_graphs_ggg_sbm_procedural/"
                + gran_filename
            ),
            "autograph_generated_fixed": autograph_logs_root
            + "/autograph_sbm.pt",
            "digress_generated_procedural": (
                digress_procedural_root + "/sbm-procedural/epoch_1499.nx.pkl"
            ),
        },
        "planar": {
            "gran_generated_procedural": (
                gran_logs_root
                + "planar_procedural/generated_graphs_ggg_planar_procedural/"
                + gran_filename
            ),
            "autograph_generated_fixed": autograph_logs_root
            + "/autograph_planar.pt",
            "digress_generated_procedural": (
                digress_procedural_root + "/planar-procedural/epoch_3479.nx.pkl"
            ),
        },
        "lobster": {
            "gran_generated_procedural": (
                gran_logs_root
                + "lobster_procedural/generated_graphs_ggg_lobster_procedural/"
                + gran_filename
            ),
            "digress_generated_procedural": (
                digress_procedural_root + "/lobster-procedural/epoch_989.nx.pkl"
            ),
        },
    }


def generate_datasets(
    n_big: int,
    n_generated_graphs: int,
    gran_logs_root: str,
    autograph_logs_root: str,
    digress_logs_root: str,
    digress_procedural_root: str,
    gran_filename: str = "gen_graphs_1040.pkl",
):
    logger.info("Generating datasets")

    # Get all paths from the dedicated function
    dataset_paths = construct_dataset_paths(
        gran_logs_root=gran_logs_root,
        autograph_logs_root=autograph_logs_root,
        digress_logs_root=digress_logs_root,
        digress_procedural_root=digress_procedural_root,
        gran_filename=gran_filename,
    )

    datasets = {
        "sbm": {
            "procedural": {
                "train": ProceduralSBMGraphDataset(
                    split="train",
                    num_graphs=n_big,
                    show_generation_progress=True,
                ).to_nx(),
                "test": ProceduralSBMGraphDataset(
                    split="test",
                    num_graphs=n_big,
                    show_generation_progress=True,
                ).to_nx(),
                "gran_generated_procedural": dataset_paths["sbm"][
                    "gran_generated_procedural"
                ],
                "digress_generated_procedural": dataset_paths["sbm"][
                    "digress_generated_procedural"
                ],
            },
        },
        "planar": {
            "procedural": {
                "train": ProceduralPlanarGraphDataset(
                    split="train",
                    num_graphs=n_big,
                    show_generation_progress=True,
                ).to_nx(),
                "test": ProceduralPlanarGraphDataset(
                    split="test",
                    num_graphs=n_big,
                    show_generation_progress=True,
                ).to_nx(),
                "gran_generated_procedural": dataset_paths["planar"][
                    "gran_generated_procedural"
                ],
                "digress_generated_procedural": dataset_paths["planar"][
                    "digress_generated_procedural"
                ],
            },
        },
        "lobster": {
            "procedural": {
                "train": ProceduralLobsterGraphDataset(
                    split="train",
                    num_graphs=n_big,
                    show_generation_progress=True,
                ).to_nx(),
                "test": ProceduralLobsterGraphDataset(
                    split="test",
                    num_graphs=n_big,
                    show_generation_progress=True,
                ).to_nx(),
                "gran_generated_procedural": dataset_paths["lobster"][
                    "gran_generated_procedural"
                ],
                "digress_generated_procedural": dataset_paths["lobster"][
                    "digress_generated_procedural"
                ],
            },
        },
    }

    logger.info("Loading generated graphs")
    for dataset in tqdm(datasets.keys(), desc="Loading generated graphs"):
        for training_dataset in ["procedural"]:
            logger.info(
                f"Processing dataset: {dataset}, training_dataset: {training_dataset}"
            )
            # Load GRAN generated graphs
            generated_data_path = datasets[dataset][training_dataset][
                "gran_generated_procedural"
            ]
            logger.info(
                f"Loading GRAN generated graphs from: {generated_data_path}"
            )
            datasets[dataset][training_dataset]["gran_generated_procedural"] = (
                pickle.load(open(generated_data_path, "rb"))[
                    :n_generated_graphs
                ]
            )
            logger.info(
                f"Successfully loaded {len(datasets[dataset][training_dataset]['gran_generated_procedural'])} GRAN generated graphs"
            )

            # Load Autograph generated graphs if available
            if (
                dataset in dataset_paths
                and "autograph_generated_fixed" in dataset_paths[dataset]
            ):
                logger.info(
                    f"Found Autograph generated graphs for {dataset}/{training_dataset}"
                )
                autograph_path = dataset_paths[dataset][
                    "autograph_generated_fixed"
                ]
                logger.info(f"Loading Autograph graphs from: {autograph_path}")
                try:
                    pyg_graphs = torch.load(autograph_path, weights_only=False)
                    datasets[dataset][training_dataset][
                        "autograph_generated_fixed"
                    ] = [
                        to_networkx(g, to_undirected=True) for g in pyg_graphs
                    ][:n_generated_graphs]
                    logger.info(
                        f"Successfully loaded {len(datasets[dataset][training_dataset]['autograph_generated_fixed'])} Autograph generated graphs"
                    )
                except FileNotFoundError:
                    logger.warning(
                        f"Autograph file not found: {autograph_path}"
                    )
                    datasets[dataset][training_dataset][
                        "autograph_generated_fixed"
                    ] = None
            else:
                logger.info(
                    f"No Autograph generated graphs found for {dataset}/{training_dataset}"
                )
                datasets[dataset][training_dataset][
                    "autograph_generated_fixed"
                ] = None

            # Load DiGress generated graphs
            digress_path = datasets[dataset][training_dataset][
                "digress_generated_procedural"
            ]
            logger.info(
                f"Loading DiGress generated graphs from: {digress_path}"
            )
            datasets[dataset][training_dataset][
                "digress_generated_procedural"
            ] = pickle.load(open(digress_path, "rb"))[:n_generated_graphs]
            logger.info(
                f"Successfully loaded {len(datasets[dataset][training_dataset]['digress_generated_procedural'])} DiGress generated graphs"
            )

    logger.info("Datasets initialized")
    for dataset in datasets:
        for training_dataset in datasets[dataset]:
            for key in datasets[dataset][training_dataset]:
                if datasets[dataset][training_dataset][key] is not None:
                    if isinstance(
                        datasets[dataset][training_dataset][key], str
                    ):
                        logger.warning(
                            f"String value found in {dataset}/{training_dataset}/{key}"
                        )
    return datasets


def save_mmd_samples(
    mmd_samples,
    dataset_type,
    generation_procedure,
    descriptor,
    n_graphs,
    variant,
    test_set_type,
):
    samples_dir = here() / "./data/mmd_samples/"
    os.makedirs(samples_dir, exist_ok=True)
    filename = f"{dataset_type}_{generation_procedure}_{descriptor.__class__.__name__}_{n_graphs}_{variant}_{test_set_type}.npy"
    np.save(samples_dir / filename, mmd_samples)


def return_mmd_results(
    dataset_type,
    generation_procedure,
    descriptor,
    n_graphs,
    variant,
    test_set_type,
    reason=None,
    mmd_results=None,
):
    return {
        "dataset_type": dataset_type,
        "generation_procedure": generation_procedure,
        "test_set_type": test_set_type,
        "descriptor": descriptor.__class__.__name__,
        "n_graphs": n_graphs,
        "variant": variant,
        "mmd_results_mean": mmd_results.mean
        if mmd_results is not None
        else None,
        "mmd_results_std": mmd_results.std if mmd_results is not None else None,
        "mmd_results_low": mmd_results.low if mmd_results is not None else None,
        "mmd_results_high": mmd_results.high
        if mmd_results is not None
        else None,
        "reason": reason,
    }


def get_dataset_class(dataset_type):
    if dataset_type == "sbm":
        return ProceduralSBMGraphDataset
    elif dataset_type == "planar":
        return ProceduralPlanarGraphDataset
    elif dataset_type == "lobster":
        return ProceduralLobsterGraphDataset


def compute_reference_set(
    datasets, dataset_type, generation_procedure, test_set_type, n_graphs
):
    if generation_procedure == "fixed":
        if test_set_type == "test":
            reference_dataset = list(
                datasets[dataset_type][generation_procedure]["train"]
            )
        else:
            reference_dataset = datasets[dataset_type][generation_procedure][
                "test"
            ]
    else:
        dataset_class = get_dataset_class(dataset_type)
        reference_dataset = dataset_class(
            split="train",
            num_graphs=n_graphs * 10,
            show_generation_progress=False,
        ).to_nx()
    return reference_dataset


def compute_subsampling_experiment(
    parameters, datasets, n_bootstraps: int, debug=False
):
    try:
        (
            dataset_type,
            generation_procedure,
            descriptor,
            n_graphs,
            variant,
            test_set_type,
        ) = parameters

        if debug:
            n_bootstraps_local = 2  # Reduce bootstraps for quick testing
            n_graphs = min(n_graphs, 64)  # Limit graph size for quick testing
        else:
            n_bootstraps_local = n_bootstraps

        if n_graphs > len(datasets[dataset_type][generation_procedure]["test"]):
            return return_mmd_results(
                dataset_type,
                generation_procedure,
                descriptor,
                n_graphs,
                variant,
                test_set_type,
                reason="n_graphs > len(test_set)",
                mmd_results=None,
            )

        reference_dataset = compute_reference_set(
            datasets,
            dataset_type,
            generation_procedure,
            test_set_type,
            n_graphs,
        )

        if test_set_type in datasets[dataset_type][generation_procedure].keys():
            test_set = datasets[dataset_type][generation_procedure][
                test_set_type
            ]
        else:
            test_set = None
            return return_mmd_results(
                dataset_type,
                generation_procedure,
                descriptor,
                n_graphs,
                variant,
                test_set_type,
                reason="No test set",
                mmd_results=None,
            )

        max_mmd_bootstrap = MaxDescriptorMMD2Interval(
            reference_dataset,
            RBFKernel(
                descriptor,
                bw=np.array(
                    [
                        0.01,
                        0.1,
                        0.25,
                        0.5,
                        0.75,
                        1.0,
                        2.5,
                        5.0,
                        7.5,
                        10.0,
                    ]
                ),
            ),
            variant=variant,
        )
        mmd_results, mmd_samples = max_mmd_bootstrap.compute(
            test_set,
            subsample_size=min(n_graphs, len(test_set)),
            num_samples=n_bootstraps_local,
            return_samples=True,
        )
        save_mmd_samples(
            mmd_samples,
            dataset_type,
            generation_procedure,
            descriptor,
            n_graphs,
            variant,
            test_set_type,
        )
        return return_mmd_results(
            dataset_type,
            generation_procedure,
            descriptor,
            n_graphs,
            variant,
            test_set_type,
            reason="Success",
            mmd_results=mmd_results,
        )
    except Exception as e:
        return return_mmd_results(
            dataset_type,
            generation_procedure,
            descriptor,
            n_graphs,
            variant,
            test_set_type,
            reason=f"Error: {e}",
            mmd_results=None,
        )


@app.command()
def main(
    n_bootstraps: int = typer.Option(100, help="Number of bootstrap samples"),
    min_n_graphs: int = typer.Option(
        32, help="Minimum number of graphs for sampling"
    ),
    max_n_graphs: int = typer.Option(
        8192, help="Maximum number of graphs for sampling"
    ),
    n_generated_graphs: int = typer.Option(
        1024, help="Number of generated graphs to load"
    ),
    n_big: int = typer.Option(
        8192 * 2, help="Number of graphs for large datasets"
    ),
    n_big_bootstraps: int = typer.Option(
        8192, help="Number of bootstraps for large datasets"
    ),
    n_small: int = typer.Option(
        128, help="Number of graphs for small datasets"
    ),
    n_lobster: int = typer.Option(60, help="Number of lobster graphs"),
    n_jobs: int = typer.Option(20, help="Number of parallel jobs"),
    gran_logs_root: str = typer.Option(
        "/fs/pool/pool-hartout/Documents/Git/GRAN/exp/GRAN/",
        help="Root directory for GRAN logs",
    ),
    autograph_logs_root: str = typer.Option(
        "/fs/pool/pool-hartout/Documents/Git/polygraph/data/autograph",
        help="Root directory for AutoGraph logs",
    ),
    digress_logs_root: str = typer.Option(
        "/fs/pool/pool-hartout/Documents/Git/polygraph/data/digress",
        help="Root directory for DiGress logs",
    ),
    digress_procedural_root: str = typer.Option(
        "/fs/pool/pool-hartout/Documents/Git/polygraph/data/digress/converted",
        help="Root directory for DiGress procedural logs",
    ),
    debug: bool = typer.Option(
        False, help="Enable debug mode for faster testing"
    ),
    output_file: str = typer.Option(
        "./experiments/results/subsampling.csv", help="Output CSV file path"
    ),
):
    datasets = generate_datasets(
        n_big=n_big,
        n_generated_graphs=n_generated_graphs,
        gran_logs_root=gran_logs_root,
        autograph_logs_root=autograph_logs_root,
        digress_logs_root=digress_logs_root,
        digress_procedural_root=digress_procedural_root,
    )

    df = pd.DataFrame()
    SAMPLE_SIZE_RANGE = [
        min_n_graphs * (2**i)
        for i in range(int(np.log2(max_n_graphs) - np.log2(min_n_graphs)))
    ]
    VARIANTS = ["biased", "umve"]
    DATASET_TYPES = datasets.keys()
    GENERATION_PROCEDURES = ["procedural"]
    TEST_SET_TYPES = [
        "gran_generated_procedural",
        "digress_generated_procedural",
        "test",
    ]

    parameters = list(
        itertools.product(
            DATASET_TYPES,
            GENERATION_PROCEDURES,
            DESCRIPTORS,
            SAMPLE_SIZE_RANGE,
            VARIANTS,
            TEST_SET_TYPES,
        )
    )
    random.shuffle(parameters)

    logger.info(f"Generated {len(parameters)} experiments")

    result = distribute_function(
        compute_subsampling_experiment,
        parameters,
        n_jobs=n_jobs,
        datasets=datasets,
        n_bootstraps=n_bootstraps,
        show_progress=True,
        debug=debug,
    )
    logger.info(f"Concatenating {len(result)} results")
    df = pd.DataFrame.from_dict(result)
    logger.info(f"Saved {len(df)} results to {output_file}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(here() / output_file, index=False)


if __name__ == "__main__":
    app()
