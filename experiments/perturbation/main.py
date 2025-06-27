import os
import numpy as np
import argparse
import random
import networkx as nx
from itertools import product
from functools import partial

from polygraph.metrics.gran import (
    GRANOrbitMMD2,
    GRANDegreeMMD2,
    GRANSpectralMMD2,
    GRANClusteringMMD2,
)
from polygraph.metrics.gran import (
    RBFOrbitMMD2,
    RBFDegreeMMD2,
    RBFSpectralMMD2,
    RBFClusteringMMD2,
)
from polygraph.metrics.gran import (
    RBFOrbitInformedness,
    RBFDegreeInformedness,
    RBFSpectralInformedness,
    RBFClusteringInformedness,
    ClassifierClusteringMetric,
    ClassifierOrbitMetric,
    ClassifierDegreeeMetric,
    ClassifierSpectralMetric,
)
from polygraph.metrics.gin import (
    RBFGraphNeuralNetworkMMD2,
    RBFGraphNeuralNetworkInformedness,
    GraphNeuralNetworkClassifierMetric,
)
from polygraph.datasets import (
    ProceduralSBMGraphDataset,
    ProceduralPlanarGraphDataset,
    ProceduralLobsterGraphDataset,
    DobsonDoigGraphDataset,
    EgoGraphDataset,
)
from perturbations import (
    EdgeRewiringPerturbation,
    EdgeSwappingPerturbation,
    MixingPerturbation,
    EdgeDeletionPerturbation,
    EdgeAdditionPerturbation,
)

import faulthandler
import threadpoolctl
import torch

faulthandler.enable()
print(threadpoolctl.threadpool_info())


def run_evaluation(graphs, metrics_dict):
    """A top-level, picklable function to run evaluations."""
    metric_items = list(metrics_dict.items())
    random.shuffle(metric_items)  # Try to ensure uniform resource usage
    result = {}
    for name, m in metric_items:
        print(f"Computing {name}...")
        result[name] = m.compute(graphs)
        # Force garbage collection after each metric
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-graphs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="sbm")
    parser.add_argument(
        "--perturbation-type", type=str, default="edge_rewiring"
    )
    parser.add_argument("--dump-dir", type=str, default="")
    parser.add_argument("--max-noise-level", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--classifiers", type=str, nargs="+", default=["tabpfn", "lr"]
    )
    parser.add_argument(
        "--variants", type=str, nargs="+", default=["informedness", "jsd"]
    )
    parser.add_argument("--no-memoize", action="store_true")
    args = parser.parse_args()

    output_path = os.path.join(
        args.dump_dir,
        f"perturbation_{args.dataset}_{args.perturbation_type}.csv",
    )

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Exiting.")
        exit(0)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "sbm":
        reference_graphs = ProceduralSBMGraphDataset(
            num_graphs=args.num_graphs, split="reference", seed=args.seed
        ).to_nx()
        perturbed_graphs = ProceduralSBMGraphDataset(
            num_graphs=args.num_graphs, split="perturbed", seed=args.seed
        ).to_nx()
    elif args.dataset == "planar":
        reference_graphs = ProceduralPlanarGraphDataset(
            num_graphs=args.num_graphs, split="reference", seed=args.seed
        ).to_nx()
        perturbed_graphs = ProceduralPlanarGraphDataset(
            num_graphs=args.num_graphs, split="perturbed", seed=args.seed
        ).to_nx()
    elif args.dataset == "lobster":
        reference_graphs = ProceduralLobsterGraphDataset(
            num_graphs=args.num_graphs, split="reference", seed=args.seed
        ).to_nx()
        perturbed_graphs = ProceduralLobsterGraphDataset(
            num_graphs=args.num_graphs, split="perturbed", seed=args.seed
        ).to_nx()
    elif args.dataset == "proteins":
        train = list(DobsonDoigGraphDataset(split="train").to_nx())
        test = list(DobsonDoigGraphDataset(split="test").to_nx())
        val = list(DobsonDoigGraphDataset(split="val").to_nx())
        all = train + test + val

        # remove attributes
        for g in all:
            del g.graph["is_enzyme"]
            for n in g.nodes:
                del g.nodes[n]["residues"]

        random.shuffle(all)
        reference_graphs = all[: len(all) // 2]
        perturbed_graphs = all[len(all) // 2 :]
    elif args.dataset == "ego":
        train = list(EgoGraphDataset(split="train").to_nx())
        test = list(EgoGraphDataset(split="test").to_nx())
        val = list(EgoGraphDataset(split="val").to_nx())
        all = train + test + val

        # Delete self-loops
        for g in all:
            g.remove_edges_from(nx.selfloop_edges(g))

        random.shuffle(all)
        reference_graphs = all[: len(all) // 2]
        perturbed_graphs = all[len(all) // 2 :]
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    reference_graphs = list(reference_graphs)
    perturbed_graphs = list(perturbed_graphs)

    metrics = {}

    mmd_metrics = {
        "orbit_tv": GRANOrbitMMD2(reference_graphs),
        "degree_tv": GRANDegreeMMD2(reference_graphs),
        "spectral_tv": GRANSpectralMMD2(reference_graphs),
        "clustering_tv": GRANClusteringMMD2(reference_graphs),
        "orbit_rbf": RBFOrbitMMD2(reference_graphs),
        "degree_rbf": RBFDegreeMMD2(reference_graphs),
        "spectral_rbf": RBFSpectralMMD2(reference_graphs),
        "clustering_rbf": RBFClusteringMMD2(reference_graphs),
        "gin_rbf": RBFGraphNeuralNetworkMMD2(reference_graphs),
    }

    if "informedness" in args.variants:
        rkhs_informedness_metrics = {
            "gin_rbf_informedness": RBFGraphNeuralNetworkInformedness(
                reference_graphs
            ),
            "gin_lr_informedness": GraphNeuralNetworkClassifierMetric(
                reference_graphs
            ),
            "orbit_rbf_informedness": RBFOrbitInformedness(reference_graphs),
            "degree_rbf_informedness": RBFDegreeInformedness(reference_graphs),
            "spectral_rbf_informedness": RBFSpectralInformedness(
                reference_graphs
            ),
            "clustering_rbf_informedness": RBFClusteringInformedness(
                reference_graphs
            ),
        }
    else:
        rkhs_informedness_metrics = {}

    prob_classifier_metrics = {}

    possible_classifiers = {"lr": "logistic", "tabpfn": "tabpfn"}
    possible_variants = {"informedness": "informedness-adaptive", "jsd": "jsd"}

    for classifier, metric_variant in product(args.classifiers, args.variants):
        prob_classifier_metrics[f"orbit_{classifier}_{metric_variant}"] = (
            ClassifierOrbitMetric(
                reference_graphs,
                variant=possible_variants[metric_variant],
                classifier=possible_classifiers[classifier],
            )
        )
        prob_classifier_metrics[f"degree_{classifier}_{metric_variant}"] = (
            ClassifierDegreeeMetric(
                reference_graphs,
                variant=possible_variants[metric_variant],
                classifier=possible_classifiers[classifier],
            )
        )
        prob_classifier_metrics[f"spectral_{classifier}_{metric_variant}"] = (
            ClassifierSpectralMetric(
                reference_graphs,
                variant=possible_variants[metric_variant],
                classifier=possible_classifiers[classifier],
            )
        )
        prob_classifier_metrics[f"clustering_{classifier}_{metric_variant}"] = (
            ClassifierClusteringMetric(
                reference_graphs,
                variant=possible_variants[metric_variant],
                classifier=possible_classifiers[classifier],
            )
        )
        prob_classifier_metrics[f"gin_{classifier}_{metric_variant}"] = (
            GraphNeuralNetworkClassifierMetric(
                reference_graphs,
                variant=possible_variants[metric_variant],
                classifier=possible_classifiers[classifier],
            )
        )

    metrics = {
        **mmd_metrics,
        **rkhs_informedness_metrics,
        **prob_classifier_metrics,
    }

    evaluator = partial(run_evaluation, metrics_dict=metrics)

    perturbations = {
        "edge_rewiring": EdgeRewiringPerturbation,
        "edge_swapping": EdgeSwappingPerturbation,
        "mixing": MixingPerturbation,
        "edge_deletion": EdgeDeletionPerturbation,
        "edge_addition": EdgeAdditionPerturbation,
    }
    perturbation = perturbations[args.perturbation_type](
        perturbed_graphs, evaluator
    )

    df = perturbation.evaluate(
        np.linspace(0, args.max_noise_level, args.num_steps),
        num_workers=args.num_workers,
        memoize=not args.no_memoize,
        verbose=True,
    )

    fname = f"perturbation_{args.dataset}_{args.perturbation_type}.csv"
    df.to_csv(output_path, index=False)
