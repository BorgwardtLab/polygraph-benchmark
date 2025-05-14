import itertools
import pickle

import pandas as pd
import torch
from loguru import logger
from pyprojroot import here
from torch_geometric.utils import to_networkx

from polygraph.datasets.lobster import (
    LobsterGraphDataset,
    ProceduralLobsterGraphDataset,
)
from polygraph.datasets.planar import (
    PlanarGraphDataset,
    ProceduralPlanarGraphDataset,
)
from polygraph.datasets.sbm import ProceduralSBMGraphDataset, SBMGraphDataset
from polygraph.metrics.base.data_copying import (
    CelledTrainDistanceCopyingMetric,
    TrainDistanceCopyingMetric,
)
from polygraph.utils.graph_descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    EigenvalueHistogram,
    OrbitCounts,
)
from polygraph.utils.parallel import distribute_function

N_BIG = 8192
N_SMALL = 128
N_LOBSTER = 60
N_GENERATED_GRAPHS = 1024


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
        for test_set_type in ["fixed", "procedural"]:
            for training_dataset in ["procedural", "fixed"]:
                for model in ["gran", "autograph", "digress"]:
                    key = f"{model}_generated_{training_dataset}"

                    if key not in datasets[dataset][test_set_type]:
                        datasets[dataset][test_set_type][key] = None
                        continue

                    if datasets[dataset][test_set_type][key].endswith(".pkl"):
                        datasets[dataset][test_set_type][key] = pickle.load(
                            open(datasets[dataset][test_set_type][key], "rb")
                        )[:N_GENERATED_GRAPHS]

                    elif datasets[dataset][test_set_type][key].endswith(".pt"):
                        pyg_graphs = torch.load(
                            datasets[dataset][test_set_type][key],
                            weights_only=False,
                        )
                        datasets[dataset][test_set_type][key] = (
                            [
                                to_networkx(g, to_undirected=True)
                                for g in pyg_graphs
                            ]
                        )[:N_GENERATED_GRAPHS]
                    else:
                        datasets[dataset][test_set_type][key] = None
                    logger.info(
                        f"Loaded {len(datasets[dataset][test_set_type][key])} graphs for {dataset} {test_set_type} {key}"
                    )

    logger.info("Datasets initialized")
    return datasets


def compute_experiment(parameters, datasets):
    (
        dataset_type,
        generated_graph_type,
        training_dataset,
        # model,
        k,
        descriptor,
    ) = parameters
    train_graphs = datasets[dataset_type][training_dataset]["train"]
    test_graphs = datasets[dataset_type][training_dataset]["test"]
    generated_graphs = datasets[dataset_type][training_dataset][
        generated_graph_type
    ]

    if generated_graphs is None:
        return {
            "dataset": dataset_type,
            "training_dataset": training_dataset,
            "training_set_size": len(train_graphs),
            "test_set_size": len(test_graphs),
            "generated_graph_type": generated_graph_type,
            "descriptor": descriptor.__class__.__name__,
            # "model": model,
            "train_test_distance_p_value": None,
            "train_test_celled_distance_p_value": None,
            "k": k,
            "reason": "No graphs generated for this configuration",
        }

    metric_train_test_overfit = TrainDistanceCopyingMetric(
        train_graphs,
        test_graphs,
        descriptor,
        distance="l1",
    )
    pval_overfit = metric_train_test_overfit.compute(generated_graphs)

    pval_overfit_celled = CelledTrainDistanceCopyingMetric(
        train_graphs,
        test_graphs,
        descriptor,
        distance="l1",
        k=k,
    )
    try:
        pval_celled = pval_overfit_celled.compute(
            generated_graphs, tau=10 / len(generated_graphs)
        )
        reason = "Successfully computed pval_celled"
    except Exception as e:
        pval_celled = None
        reason = f"Error computing pval_celled: {e}"

    return {
        "dataset": dataset_type,
        "training_dataset": training_dataset,
        "training_set_size": len(train_graphs),
        "test_set_size": len(test_graphs),
        "generated_graph_type": generated_graph_type,
        "descriptor": descriptor.__class__.__name__,
        # "model": model,
        "train_test_distance_p_value": pval_overfit,
        "train_test_celled_distance_p_value": pval_celled,
        "k": k,
        "reason": reason,
    }


def main():
    datasets = generate_datasets()
    K = [4, 10, 20]
    DESCRIPTORS = [
        DegreeHistogram(max_degree=500),
        ClusteringHistogram(bins=100, sparse=False),
        EigenvalueHistogram(n_bins=100, sparse=False),
        OrbitCounts(graphlet_size=4),
    ]
    TRAINING_DATASETS = ["fixed", "procedural"]
    DATASETS = datasets.keys()
    GENERATED_GRAPH_TYPES = [
        "gran_generated_procedural",
        "gran_generated_fixed",
        "autograph_generated_fixed",
        "digress_generated_fixed",
    ]
    N_JOBS = 100

    parameters = itertools.product(
        DATASETS,
        GENERATED_GRAPH_TYPES,
        TRAINING_DATASETS,
        K,
        DESCRIPTORS,
    )
    parameters = list(parameters)
    logger.info(f"Computing {len(parameters)} experiments")
    results = distribute_function(
        compute_experiment,
        parameters,
        datasets=datasets,
        n_jobs=N_JOBS,
        show_progress=False if N_JOBS <= 1 else True,
    )
    df = pd.DataFrame.from_dict(results)
    df.to_csv(here() / "experiments/results/testing_results.csv", index=False)


if __name__ == "__main__":
    main()
