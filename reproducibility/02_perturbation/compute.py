#!/usr/bin/env python3
"""Compute perturbation metrics showing sensitivity of various graph metrics.

Reproduces the perturbation experiment from the original polygraph library,
computing ~34 metrics across perturbation types and datasets at varying noise levels.

Metrics computed (all per noise level):
  1. MMD metrics (10): orbit_tv, degree_tv, spectral_tv, clustering_tv (GaussianTV kernel),
     orbit_rbf, orbit5_rbf, degree_rbf, spectral_rbf, clustering_rbf, gin_rbf (RBF/AdaptiveRBF)
  2. Classifier metrics for each (classifier, variant) combo across 6 descriptors:
     orbit, orbit5, degree, spectral, clustering, gin
     - Classifiers: tabpfn (default), lr (LogisticRegression)
     - Variants: jsd, informedness

Perturbation types (5): edge_rewiring, edge_swapping, mixing, edge_deletion, edge_addition
Datasets (5): sbm, planar, lobster, proteins, ego

Usage:
    python compute.py                                              # Single run
    python compute.py dataset=sbm perturbation=edge_rewiring       # Specific config
    python compute.py --multirun                                   # All combos
    python compute.py --multirun hydra/launcher=slurm_cpu          # On SLURM
    python compute.py subset=true                                  # Quick test
"""

import gc
import json
import random
from importlib.metadata import version as pkg_version
from itertools import product
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, cast

import hydra
import networkx as nx
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from pyprojroot import here
from sklearn.linear_model import LogisticRegression

from polygraph.utils.io import (
    maybe_append_jsonl,
)

from polygraph.datasets.ego import EgoGraphDataset
from polygraph.datasets.lobster import ProceduralLobsterGraphDataset
from polygraph.datasets.planar import ProceduralPlanarGraphDataset
from polygraph.datasets.proteins import DobsonDoigGraphDataset
from polygraph.datasets.sbm import ProceduralSBMGraphDataset
from polygraph.metrics.base.mmd import MaxDescriptorMMD2
from polygraph.metrics.gaussian_tv_mmd import (
    GaussianTVClusteringMMD2,
    GaussianTVDegreeMMD2,
    GaussianTVOrbitMMD2,
    GaussianTVSpectralMMD2,
)
from polygraph.metrics.rbf_mmd import (
    RBFClusteringMMD2,
    RBFDegreeMMD2,
    RBFGraphNeuralNetworkMMD2,
    RBFOrbitMMD2,
    RBFSpectralMMD2,
)
from polygraph.metrics.standard_pgd import (
    ClassifierClusteringMetric,
    ClassifierDegreeMetric,
    ClassifierOrbit4Metric,
    ClassifierOrbit5Metric,
    ClassifierSpectralMetric,
    GraphNeuralNetworkClassifierMetric,
)
from polygraph.utils.descriptors import OrbitCounts
from polygraph.utils.kernels import AdaptiveRBFKernel

REPO_ROOT = here()
_RESULTS_DIR_BASE = (
    REPO_ROOT / "reproducibility" / "figures" / "02_perturbation"
)

# Adaptive RBF bandwidths (matching the library's RBFOrbitMMD2 internals)
_RBF_BW = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])


def edge_rewiring(graph: nx.Graph, noise_level: float) -> nx.Graph:
    """Rewire edges: each selected with P(noise_level), one endpoint reconnected."""
    if not (0 <= noise_level <= 1):
        raise ValueError("Noise level must be between 0 and 1")

    perturbed = graph.copy()
    edges = list(perturbed.edges())
    nodes = list(perturbed.nodes())

    edges_to_rewire = np.random.binomial(1, noise_level, size=len(edges))
    edge_indices_to_rewire = np.where(edges_to_rewire == 1)[0]

    for edge_index in edge_indices_to_rewire:
        edge = edges[edge_index]
        perturbed.remove_edge(*edge)

        if random.random() > 0.5:
            keep_node, detach_node = edge
        else:
            detach_node, keep_node = edge

        possible_nodes = [n for n in nodes if n not in [keep_node, detach_node]]
        if possible_nodes:
            attach_node = random.choice(possible_nodes)
            perturbed.add_edge(keep_node, attach_node)
        else:
            perturbed.add_edge(*edge)

    return perturbed


def edge_swapping(graph: nx.Graph, noise_level: float) -> nx.Graph:
    """Swap edge endpoints in pairs while preserving node degrees."""
    if not (0 <= noise_level <= 1):
        raise ValueError("Noise level must be between 0 and 1")

    perturbed = graph.copy()
    original_degrees = dict(perturbed.degree())

    edges = list(perturbed.edges())
    selected = [
        i for i in range(len(edges)) if np.random.random() < noise_level
    ]
    random.shuffle(selected)

    num_pairs = len(selected) // 2
    for i in range(num_pairs):
        edge1_idx = selected[2 * i]
        edge2_idx = selected[2 * i + 1]

        a, b = edges[edge1_idx]
        c, d = edges[edge2_idx]

        if len({a, b, c, d}) < 4:
            continue

        option1_valid = not perturbed.has_edge(a, d) and not perturbed.has_edge(
            c, b
        )
        option2_valid = not perturbed.has_edge(a, c) and not perturbed.has_edge(
            b, d
        )

        if not option1_valid and not option2_valid:
            continue

        perturbed.remove_edge(a, b)
        perturbed.remove_edge(c, d)

        if option1_valid and option2_valid:
            if random.random() < 0.5:
                perturbed.add_edge(a, d)
                perturbed.add_edge(c, b)
            else:
                perturbed.add_edge(a, c)
                perturbed.add_edge(b, d)
        elif option1_valid:
            perturbed.add_edge(a, d)
            perturbed.add_edge(c, b)
        else:
            perturbed.add_edge(a, c)
            perturbed.add_edge(b, d)

    final_degrees = dict(perturbed.degree())
    for node in perturbed.nodes():
        assert original_degrees[node] == final_degrees[node], (
            f"Degree changed for node {node}: "
            f"{original_degrees[node]} -> {final_degrees[node]}"
        )

    return perturbed


def mixing(graph: nx.Graph, noise_level: float) -> nx.Graph:
    """With P(noise_level), replace graph with ER graph of same size and density."""
    if not (0 <= noise_level <= 1):
        raise ValueError("Noise level must be between 0 and 1")

    if random.random() < noise_level:
        n = graph.number_of_nodes()
        p = nx.density(graph)
        return nx.erdos_renyi_graph(n, p)

    return graph.copy()


def edge_deletion(graph: nx.Graph, noise_level: float) -> nx.Graph:
    """Delete each edge independently with P(noise_level), keeping at least one."""
    if not (0 <= noise_level <= 1):
        raise ValueError("Noise level must be between 0 and 1")

    perturbed = graph.copy()
    edges = list(perturbed.edges())
    num_edges = len(edges)

    if num_edges <= 1:
        return perturbed

    edges_to_delete = [e for e in edges if np.random.random() < noise_level]

    if len(edges_to_delete) >= num_edges:
        edge_to_keep = edges[np.random.randint(num_edges)]
        edges_to_delete = [e for e in edges_to_delete if e != edge_to_keep]

    perturbed.remove_edges_from(edges_to_delete)
    return perturbed


def edge_addition(graph: nx.Graph, noise_level: float) -> nx.Graph:
    """Add random edges; expected count = noise_level * original_edge_count."""
    if not (0 <= noise_level <= 1):
        raise ValueError("Noise level must be between 0 and 1")

    perturbed = graph.copy()
    num_edges = perturbed.number_of_edges()
    num_nodes = perturbed.number_of_nodes()
    max_possible_edges = (num_nodes * (num_nodes - 1)) // 2
    remaining_possible_edges = max_possible_edges - num_edges

    if remaining_possible_edges == 0:
        return perturbed

    target_new_edges = noise_level * num_edges
    probability = min(1.0, target_new_edges / remaining_possible_edges)

    nodes = list(perturbed.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if perturbed.has_edge(nodes[i], nodes[j]):
                continue
            if np.random.random() < probability:
                perturbed.add_edge(nodes[i], nodes[j])

    return perturbed


PERTURBATION_FNS: Dict[str, Callable[[nx.Graph, float], nx.Graph]] = {
    "edge_rewiring": edge_rewiring,
    "edge_swapping": edge_swapping,
    "mixing": mixing,
    "edge_deletion": edge_deletion,
    "edge_addition": edge_addition,
}


def load_dataset(
    dataset: str, num_graphs: int, seed: int
) -> Tuple[List[nx.Graph], List[nx.Graph]]:
    """Load reference and perturbed base graph sets.

    Procedural datasets use 'train'/'test' splits for two independent sets
    (analogous to the original 'reference'/'perturbed' splits).
    Real datasets combine all splits, shuffle, and split in half.
    """
    if dataset == "sbm":
        reference_graphs = list(
            ProceduralSBMGraphDataset(
                num_graphs=num_graphs, split="train", seed=seed
            ).to_nx()
        )
        perturbed_graphs = list(
            ProceduralSBMGraphDataset(
                num_graphs=num_graphs, split="test", seed=seed
            ).to_nx()
        )
    elif dataset == "planar":
        reference_graphs = list(
            ProceduralPlanarGraphDataset(
                num_graphs=num_graphs, split="train", seed=seed
            ).to_nx()
        )
        perturbed_graphs = list(
            ProceduralPlanarGraphDataset(
                num_graphs=num_graphs, split="test", seed=seed
            ).to_nx()
        )
    elif dataset == "lobster":
        reference_graphs = list(
            ProceduralLobsterGraphDataset(
                num_graphs=num_graphs, split="train", seed=seed
            ).to_nx()
        )
        perturbed_graphs = list(
            ProceduralLobsterGraphDataset(
                num_graphs=num_graphs, split="test", seed=seed
            ).to_nx()
        )
    elif dataset == "proteins":
        train = list(DobsonDoigGraphDataset(split="train").to_nx())
        test = list(DobsonDoigGraphDataset(split="test").to_nx())
        val = list(DobsonDoigGraphDataset(split="val").to_nx())
        all_graphs = train + test + val

        for g in all_graphs:
            if "is_enzyme" in g.graph:
                del g.graph["is_enzyme"]
            for n in g.nodes:
                if "residues" in g.nodes[n]:
                    del g.nodes[n]["residues"]

        random.shuffle(all_graphs)
        half = len(all_graphs) // 2
        reference_graphs = all_graphs[:half]
        perturbed_graphs = all_graphs[half : 2 * half]
    elif dataset == "ego":
        train = list(EgoGraphDataset(split="train").to_nx())
        test = list(EgoGraphDataset(split="test").to_nx())
        val = list(EgoGraphDataset(split="val").to_nx())
        all_graphs = train + test + val

        for g in all_graphs:
            g.remove_edges_from(nx.selfloop_edges(g))

        random.shuffle(all_graphs)
        half = len(all_graphs) // 2
        reference_graphs = all_graphs[:half]
        perturbed_graphs = all_graphs[half : 2 * half]
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    return reference_graphs, perturbed_graphs


def _make_classifier(name: str, tabpfn_weights_version: str = "v2.5"):
    """Build a classifier by name. For TabPFN, respects weights version."""
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        from tabpfn.classifier import ModelVersion

        version_map = {
            "v2": ModelVersion.V2,
            "v2.5": ModelVersion.V2_5,
        }
        if tabpfn_weights_version not in version_map:
            raise ValueError(
                f"Unknown tabpfn_weights_version: {tabpfn_weights_version!r}. Must be one of {list(version_map)}"
            )
        return TabPFNClassifier.create_default_for_version(
            version_map[tabpfn_weights_version],
            device="auto",
            n_estimators=4,
        )
    elif name == "lr":
        return LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def build_metrics(
    reference_graphs: List[nx.Graph],
    classifiers: Optional[List[str]] = None,
    variants: Optional[List[str]] = None,
    tabpfn_weights_version: str = "v2.5",
    tabpfn_only: bool = False,
) -> Dict[str, Any]:
    """Build all metrics: 10 MMD + 6 descriptors x classifiers x variants.

    If tabpfn_only is True, skip MMD metrics and force classifiers=["tabpfn"],
    building only the 12 TabPFN classifier metrics.
    """
    if classifiers is None:
        classifiers = ["tabpfn", "lr"]
    if variants is None:
        variants = ["informedness", "jsd"]

    if tabpfn_only:
        classifiers = ["tabpfn"]
        logger.info(
            "tabpfn_only=True: skipping MMD metrics, using classifiers={}",
            classifiers,
        )

    metrics: Dict[str, Any] = {}

    if not tabpfn_only:
        metrics.update(
            {
                # GaussianTV kernel (4 metrics)
                "orbit_tv": GaussianTVOrbitMMD2(reference_graphs),
                "degree_tv": GaussianTVDegreeMMD2(reference_graphs),
                "spectral_tv": GaussianTVSpectralMMD2(reference_graphs),
                "clustering_tv": GaussianTVClusteringMMD2(reference_graphs),
                # Adaptive RBF kernel (6 metrics)
                "orbit_rbf": RBFOrbitMMD2(reference_graphs),
                "orbit5_rbf": MaxDescriptorMMD2(
                    reference_graphs=reference_graphs,
                    kernel=AdaptiveRBFKernel(
                        descriptor_fn=OrbitCounts(graphlet_size=5),
                        bw=_RBF_BW,
                    ),
                    variant="biased",
                ),
                "degree_rbf": RBFDegreeMMD2(reference_graphs),
                "spectral_rbf": RBFSpectralMMD2(reference_graphs),
                "clustering_rbf": RBFClusteringMMD2(reference_graphs),
                "gin_rbf": RBFGraphNeuralNetworkMMD2(reference_graphs),
            }
        )

    # Classifier metrics (6 descriptors x classifiers x variants)
    for clf_name, variant in product(classifiers, variants):
        variant_lit = cast(Literal["informedness", "jsd"], variant)
        metrics[f"orbit_{clf_name}_{variant}"] = ClassifierOrbit4Metric(
            reference_graphs,
            variant=variant_lit,
            classifier=_make_classifier(clf_name, tabpfn_weights_version),
        )
        metrics[f"orbit5_{clf_name}_{variant}"] = ClassifierOrbit5Metric(
            reference_graphs,
            variant=variant_lit,
            classifier=_make_classifier(clf_name, tabpfn_weights_version),
        )
        metrics[f"degree_{clf_name}_{variant}"] = ClassifierDegreeMetric(
            reference_graphs,
            variant=variant_lit,
            classifier=_make_classifier(clf_name, tabpfn_weights_version),
        )
        metrics[f"spectral_{clf_name}_{variant}"] = ClassifierSpectralMetric(
            reference_graphs,
            variant=variant_lit,
            classifier=_make_classifier(clf_name, tabpfn_weights_version),
        )
        metrics[f"clustering_{clf_name}_{variant}"] = (
            ClassifierClusteringMetric(
                reference_graphs,
                variant=variant_lit,
                classifier=_make_classifier(clf_name, tabpfn_weights_version),
            )
        )
        metrics[f"gin_{clf_name}_{variant}"] = (
            GraphNeuralNetworkClassifierMetric(
                reference_graphs,
                variant=variant_lit,
                classifier=_make_classifier(clf_name, tabpfn_weights_version),
            )
        )

    return metrics


def evaluate_metrics(
    perturbed_graphs: List[nx.Graph],
    metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Compute all metrics on perturbed graphs.

    Randomizes evaluation order (matching original) for uniform resource usage.
    ClassifierMetric returns (train, test) tuple; we use the test score.
    """
    metric_items = list(metrics.items())
    random.shuffle(metric_items)

    result: Dict[str, float] = {}
    for name, m in metric_items:
        logger.info("Computing {}...", name)
        score = m.compute(perturbed_graphs)

        if isinstance(score, tuple):
            result[name] = float(score[1])
        else:
            result[name] = float(score)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


@hydra.main(
    config_path="../configs", config_name="02_perturbation", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Compute perturbation metrics for one (dataset, perturbation) pair."""
    tabpfn_weights_version: str = cfg.get("tabpfn_weights_version", "v2.5")
    RESULTS_DIR = _RESULTS_DIR_BASE / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset: str = cfg.dataset
    perturbation: str = cfg.perturbation
    num_graphs: int = cfg.get("num_graphs", 2048)
    seed: int = cfg.get("seed", 42)
    num_steps: int = cfg.get("num_steps", 100)
    max_noise_level: float = cfg.get("max_noise_level", 1.0)
    subset: bool = cfg.get("subset", False)
    classifiers: List[str] = list(cfg.get("classifiers", ["tabpfn", "lr"]))
    variants: List[str] = list(cfg.get("variants", ["informedness", "jsd"]))
    tabpfn_only: bool = cfg.get("tabpfn_only", False)

    if subset:
        num_graphs = 100
        num_steps = 10

    output_path = RESULTS_DIR / f"perturbation_{dataset}_{perturbation}.json"

    if output_path.exists():
        logger.info("Output file {} already exists. Skipping.", output_path)
        maybe_append_jsonl(
            {
                "experiment": "02_perturbation",
                "script": "compute.py",
                "dataset": dataset,
                "perturbation": perturbation,
                "status": "skipped",
                "reason": "output_exists",
                "output_path": str(output_path),
            }
        )
        return

    logger.info(
        "Computing perturbation metrics: dataset={}, perturbation={}, "
        "num_graphs={}, num_steps={}, max_noise_level={}",
        dataset,
        perturbation,
        num_graphs,
        num_steps,
        max_noise_level,
    )

    random.seed(seed)
    np.random.seed(seed)

    reference_graphs, perturbed_base_graphs = load_dataset(
        dataset, num_graphs, seed
    )
    logger.info(
        "Loaded {} reference graphs and {} base graphs for perturbation",
        len(reference_graphs),
        len(perturbed_base_graphs),
    )

    metrics = build_metrics(
        reference_graphs,
        classifiers=classifiers,
        variants=variants,
        tabpfn_weights_version=tabpfn_weights_version,
        tabpfn_only=tabpfn_only,
    )
    logger.info("Initialized {} metrics", len(metrics))

    if perturbation not in PERTURBATION_FNS:
        raise ValueError(
            f"Unknown perturbation: {perturbation}. "
            f"Choose from: {list(PERTURBATION_FNS.keys())}"
        )
    perturb_fn = PERTURBATION_FNS[perturbation]

    noise_levels = np.linspace(0, max_noise_level, num_steps)
    all_results: List[Dict[str, float]] = []

    for i, noise_level in enumerate(noise_levels):
        logger.info("[{}/{}] noise_level={:.4f}", i + 1, num_steps, noise_level)

        perturbed_graphs = [
            perturb_fn(g, noise_level) for g in perturbed_base_graphs
        ]

        scores = evaluate_metrics(perturbed_graphs, metrics)
        scores["noise_level"] = float(noise_level)
        all_results.append(scores)

        logger.info(
            "  Completed {} metrics at noise_level={:.4f}",
            len(scores) - 1,
            noise_level,
        )

    output = {
        "dataset": dataset,
        "perturbation": perturbation,
        "num_graphs": num_graphs,
        "seed": seed,
        "num_steps": num_steps,
        "max_noise_level": max_noise_level,
        "classifiers": classifiers,
        "variants": variants,
        "results": all_results,
        "tabpfn_package_version": pkg_version("tabpfn"),
        "tabpfn_weights_version": tabpfn_weights_version,
        "tabpfn_only": tabpfn_only,
    }

    output_path.write_text(json.dumps(output, indent=2))
    logger.success("Results saved to {}", output_path)
    maybe_append_jsonl(
        {
            "experiment": "02_perturbation",
            "script": "compute.py",
            "dataset": dataset,
            "perturbation": perturbation,
            "status": "ok",
            "output_path": str(output_path),
            "num_rows": len(all_results),
        }
    )


if __name__ == "__main__":
    main()
