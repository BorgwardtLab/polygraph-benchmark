import os
import numpy as np
import argparse
import random
import networkx as nx

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
    LRClusteringInformedness,
    LROrbitInformedness,
    LRDegreeInformedness,
    LRSpectralInformedness,
)
from polygraph.metrics.gin import (
    RBFGraphNeuralNetworkMMD2,
    RBFGraphNeuralNetworkInformedness,
    LRGraphNeuralNetworkInformedness,
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
    args = parser.parse_args()

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

    metrics = {
        "orbit_tv": GRANOrbitMMD2(reference_graphs),
        "degree_tv": GRANDegreeMMD2(reference_graphs),
        "spectral_tv": GRANSpectralMMD2(reference_graphs),
        "clustering_tv": GRANClusteringMMD2(reference_graphs),
        "orbit_rbf": RBFOrbitMMD2(reference_graphs),
        "degree_rbf": RBFDegreeMMD2(reference_graphs),
        "spectral_rbf": RBFSpectralMMD2(reference_graphs),
        "clustering_rbf": RBFClusteringMMD2(reference_graphs),
        "gin_rbf": RBFGraphNeuralNetworkMMD2(reference_graphs),
        "gin_rbf_informedness": RBFGraphNeuralNetworkInformedness(
            reference_graphs
        ),
        "gin_lr_informedness": LRGraphNeuralNetworkInformedness(
            reference_graphs
        ),
        "orbit_rbf_informedness": RBFOrbitInformedness(reference_graphs),
        "degree_rbf_informedness": RBFDegreeInformedness(reference_graphs),
        "spectral_rbf_informedness": RBFSpectralInformedness(reference_graphs),
        "clustering_rbf_informedness": RBFClusteringInformedness(
            reference_graphs
        ),
        "orbit_lr_informedness": LROrbitInformedness(reference_graphs),
        "degree_lr_informedness": LRDegreeInformedness(reference_graphs),
        "spectral_lr_informedness": LRSpectralInformedness(reference_graphs),
        "clustering_lr_informedness": LRClusteringInformedness(
            reference_graphs
        ),
    }

    evaluator = lambda graphs: {
        name: m.compute(graphs) for name, m in metrics.items()
    }

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
        np.linspace(0, args.max_noise_level, args.num_steps)
    )

    fname = f"perturbation_{args.dataset}_{args.perturbation_type}.csv"
    df.to_csv(os.path.join(args.dump_dir, fname), index=False)
