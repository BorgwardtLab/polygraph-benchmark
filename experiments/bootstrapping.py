import itertools
import os
import pickle

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pyprojroot import here
from torch_geometric.utils import to_networkx

from polygraph.datasets import (
    LobsterGraphDataset,
    PlanarGraphDataset,
    ProceduralLobsterGraphDataset,
    ProceduralPlanarGraphDataset,
    ProceduralSBMGraphDataset,
    SBMGraphDataset,
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

N_BOOTSTRAPS = 100

MIN_N_GRAPHS = 32
MAX_N_GRAPHS = 8192
N_GENERATED_GRAPHS = 1024

N_BIG = 8192 * 2
N_BIG_BOOTSTRAPS = 8192
N_SMALL = 128
N_LOBSTER = 60


DESCRIPTORS = [
    DegreeHistogram(max_degree=500),
    ClusteringHistogram(bins=100, sparse=False),
    EigenvalueHistogram(n_bins=100, sparse=False),
    OrbitCounts(),
]

gran_logs_root = "/fs/pool/pool-hartout/Documents/Git/GRAN/exp/GRAN/"
gran_filename = "gen_graphs_1040.pkl"

gran_sbm_fixed_generated_graphs_path = (
    gran_logs_root
    + "GRANMixtureBernoulli_ggg_sbm_2025-Apr-07-23-14-11_119120/generated_graphs_ggg_sbm/"
    + gran_filename
)
gran_sbm_procedural_generated_graphs_path = (
    gran_logs_root
    + "GRANMixtureBernoulli_ggg_sbm_procedural_2025-Apr-07-23-16-17_40179/generated_graphs_ggg_sbm_procedural/"
    + gran_filename
)

gran_planar_fixed_generated_graphs_path = (
    gran_logs_root
    + "GRANMixtureBernoulli_ggg_planar_2025-Apr-07-23-16-17_93528/generated_graphs_ggg_planar/"
    + gran_filename
)
gran_planar_procedural_generated_graphs_path = (
    gran_logs_root
    + "GRANMixtureBernoulli_ggg_planar_procedural_2025-Apr-07-23-16-17_40183/generated_graphs_ggg_planar_procedural/"
    + gran_filename
)

gran_lobster_fixed_generated_graphs_path = (
    gran_logs_root
    + "GRANMixtureBernoulli_ggg_lobster_2025-Apr-07-23-16-17_93527/generated_graphs_ggg_lobster/"
    + gran_filename
)
gran_lobster_procedural_generated_graphs_path = (
    gran_logs_root
    + "GRANMixtureBernoulli_ggg_lobster_procedural_2025-Apr-07-23-16-17_40181/generated_graphs_ggg_lobster_procedural/"
    + gran_filename
)

autograph_logs_root = (
    "/fs/pool/pool-hartout/Documents/Git/polygraph/data/autograph"
)

autograph_sbm_fixed_generated_graphs_path = (
    autograph_logs_root + "/autograph_sbm.pt"
)
autograph_planar_fixed_generated_graphs_path = (
    autograph_logs_root + "/autograph_planar.pt"
)

digress_logs_root = "/fs/pool/pool-hartout/Documents/Git/polygraph/data/digress"
digress_lobster_fixed_generated_graphs_path = (
    digress_logs_root + "/lobster_8k_samples_nx.pkl"
)
digress_planar_fixed_generated_graphs_path = (
    digress_logs_root + "/planar_150k_samples_nx.pkl"
)
digress_sbm_fixed_generated_graphs_path = (
    digress_logs_root + "/sbm_50k_samples_nx.pkl"
)


def generate_datasets():
    logger.info("Generating datasets")
    datasets = {
        "sbm": {
            "fixed": {
                "train": SBMGraphDataset(split="train").to_nx(),
                "test": SBMGraphDataset(split="test").to_nx(),
                "autograph_generated_fixed": autograph_sbm_fixed_generated_graphs_path,
                "gran_generated_fixed": gran_sbm_fixed_generated_graphs_path,
                "gran_generated_procedural": gran_sbm_procedural_generated_graphs_path,
                "digress_generated_fixed": digress_sbm_fixed_generated_graphs_path,
            },
            "procedural": {
                "train": ProceduralSBMGraphDataset(
                    split="train",
                    num_graphs=N_BIG,
                    show_generation_progress=True,
                ).to_nx(),
                "test": ProceduralSBMGraphDataset(
                    split="test",
                    num_graphs=N_BIG,
                    show_generation_progress=True,
                ).to_nx(),
                "autograph_generated_fixed": autograph_sbm_fixed_generated_graphs_path,
                "gran_generated_fixed": gran_sbm_fixed_generated_graphs_path,
                "gran_generated_procedural": gran_sbm_procedural_generated_graphs_path,
                "digress_generated_fixed": digress_sbm_fixed_generated_graphs_path,
            },
        },
        "planar": {
            "fixed": {
                "train": PlanarGraphDataset(split="train").to_nx(),
                "test": PlanarGraphDataset(split="test").to_nx(),
                "autograph_generated_fixed": autograph_planar_fixed_generated_graphs_path,
                "gran_generated_fixed": gran_planar_fixed_generated_graphs_path,
                "gran_generated_procedural": gran_planar_procedural_generated_graphs_path,
                "digress_generated_fixed": digress_planar_fixed_generated_graphs_path,
            },
            "procedural": {
                "train": ProceduralPlanarGraphDataset(
                    split="train",
                    num_graphs=N_BIG,
                    show_generation_progress=True,
                ).to_nx(),
                "test": ProceduralPlanarGraphDataset(
                    split="test",
                    num_graphs=N_BIG,
                    show_generation_progress=True,
                ).to_nx(),
                "autograph_generated_fixed": autograph_planar_fixed_generated_graphs_path,
                "gran_generated_fixed": gran_planar_fixed_generated_graphs_path,
                "gran_generated_procedural": gran_planar_procedural_generated_graphs_path,
                "digress_generated_fixed": digress_planar_fixed_generated_graphs_path,
            },
        },
        "lobster": {
            "fixed": {
                "train": LobsterGraphDataset(split="train").to_nx(),
                "test": LobsterGraphDataset(split="test").to_nx(),
                "gran_generated_fixed": gran_lobster_fixed_generated_graphs_path,
                "gran_generated_procedural": gran_lobster_procedural_generated_graphs_path,
                "digress_generated_fixed": digress_lobster_fixed_generated_graphs_path,
            },
            "procedural": {
                "train": ProceduralLobsterGraphDataset(
                    split="train",
                    num_graphs=N_BIG,
                    show_generation_progress=True,
                ).to_nx(),
                "test": ProceduralLobsterGraphDataset(
                    split="test",
                    num_graphs=N_BIG,
                    show_generation_progress=True,
                ).to_nx(),
                "gran_generated_fixed": gran_lobster_fixed_generated_graphs_path,
                "gran_generated_procedural": gran_lobster_procedural_generated_graphs_path,
                "digress_generated_fixed": digress_lobster_fixed_generated_graphs_path,
            },
        },
    }
    logger.info("Loading generated graphs")
    for dataset in datasets.keys():
        for training_dataset in ["procedural", "fixed"]:
            # Replace path with actual data
            generated_data_path = datasets[dataset][training_dataset][
                "gran_generated_procedural"
            ]
            datasets[dataset][training_dataset]["gran_generated_procedural"] = (
                pickle.load(open(generated_data_path, "rb"))[
                    :N_GENERATED_GRAPHS
                ]
            )

            generated_data_path = datasets[dataset][training_dataset][
                "gran_generated_fixed"
            ]
            datasets[dataset][training_dataset]["gran_generated_fixed"] = (
                pickle.load(open(generated_data_path, "rb"))[
                    :N_GENERATED_GRAPHS
                ]
            )

            if any(
                "autograph_generated" in key
                for key in datasets[dataset][training_dataset].keys()
            ):
                pyg_graphs = torch.load(
                    datasets[dataset][training_dataset][
                        "autograph_generated_fixed"
                    ],
                    weights_only=False,
                )
                datasets[dataset][training_dataset][
                    "autograph_generated_fixed"
                ] = ([to_networkx(g, to_undirected=True) for g in pyg_graphs])[
                    :N_GENERATED_GRAPHS
                ]
            else:
                datasets[dataset][training_dataset][
                    "autograph_generated_fixed"
                ] = None

            datasets[dataset][training_dataset]["digress_generated_fixed"] = (
                pickle.load(
                    open(
                        datasets[dataset][training_dataset][
                            "digress_generated_fixed"
                        ],
                        "rb",
                    )
                )
            )
    logger.info("Datasets initialized")
    for dataset in datasets:
        for training_dataset in datasets[dataset]:
            for key in datasets[dataset][training_dataset]:
                if datasets[dataset][training_dataset][key] is not None:
                    if not isinstance(
                        datasets[dataset][training_dataset][key], str
                    ):
                        logger.warning(
                            f"Non-string value found in {dataset}/{training_dataset}/{key}"
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


def compute_bootstrapping_experiment(parameters, datasets):
    try:
        (
            dataset_type,
            generation_procedure,
            descriptor,
            n_graphs,
            variant,
            test_set_type,
        ) = parameters

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
            num_samples=N_BOOTSTRAPS,
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


def main():
    datasets = generate_datasets()
    df = pd.DataFrame()
    SAMPLE_SIZE_RANGE = [
        MIN_N_GRAPHS * (2**i)
        for i in range(int(np.log2(MAX_N_GRAPHS) - np.log2(MIN_N_GRAPHS)))
    ]
    VARIANTS = ["biased", "umve"]
    DATASET_TYPES = datasets.keys()
    GENERATION_PROCEDURES = ["procedural", "fixed"]
    TEST_SET_TYPES = [
        "gran_generated_procedural",
        "gran_generated_fixed",
        "autograph_generated_fixed",
        "digress_generated_fixed",
        "test",
    ]
    N_JOBS = 2

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

    logger.info(f"Generated {len(parameters)} experiments")

    result = distribute_function(
        compute_bootstrapping_experiment,
        parameters,
        n_jobs=N_JOBS,
        datasets=datasets,
        show_progress=True,
    )
    logger.info(f"Concatenating {len(result)} results")
    df = pd.DataFrame.from_dict(result)
    logger.info(f"Saved {len(df)} results to bootstrapping.csv")
    df.to_csv(here() / "./experiments/results/bootstrapping.csv", index=False)


if __name__ == "__main__":
    main()
