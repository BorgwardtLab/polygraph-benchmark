#!/usr/bin/env python3
"""Generate concatenation ablation tables from pre-generated graphs.

This script compares standard PGD (max over individual descriptors) vs
concatenated PGD (all descriptors concatenated into one feature vector).

Usage:
    python generate_concatenation_tables.py
    python generate_concatenation_tables.py --subset  # Use smaller sample for testing
"""

import pickle
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer()

# Paths
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "polygraph_graphs"
OUTPUT_DIR = Path(__file__).parent / "tables"

# Configuration
MODELS = ["AUTOGRAPH", "DIGRESS", "GRAN", "ESGG"]
DATASETS = ["planar", "lobster", "sbm", "proteins"]

DATASET_DISPLAY = {
    "planar": "\\textsc{Planar-L}",
    "lobster": "\\textsc{Lobster-L}",
    "sbm": "\\textsc{SBM-L}",
    "proteins": "Proteins",
}

MODEL_DISPLAY = {
    "AUTOGRAPH": "AutoGraph",
    "DIGRESS": "\\textsc{DiGress}",
    "GRAN": "GRAN",
    "ESGG": "ESGG",
}


def load_graphs(model: str, dataset: str) -> List:
    """Load generated graphs from pickle file and convert to networkx."""
    import torch

    pkl_path = DATA_DIR / model / f"{dataset}.pkl"
    if not pkl_path.exists():
        print(f"Warning: {pkl_path} not found")
        return []
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)

    # Convert to simple undirected graphs
    cleaned = []
    for g in graphs:
        if isinstance(g, nx.Graph):
            simple = nx.Graph(g)
        elif isinstance(g, (list, tuple)) and len(g) == 2:
            # DIGRESS format: [node_feat, adj_matrix]
            try:
                node_feat, adj = g
                if isinstance(adj, torch.Tensor):
                    adj = adj.numpy()
                simple = nx.from_numpy_array(adj)
            except Exception as e:
                print(f"    Warning: Could not convert graph: {e}")
                continue
        else:
            print(f"    Warning: Unknown graph format: {type(g)}")
            continue

        simple.remove_edges_from(nx.selfloop_edges(simple))
        cleaned.append(simple)
    return cleaned


def get_reference_dataset(dataset: str, split: str = "test"):
    """Get reference dataset from polygraph library."""
    from polygraph.datasets.lobster import ProceduralLobsterGraphDataset
    from polygraph.datasets.planar import ProceduralPlanarGraphDataset
    from polygraph.datasets.proteins import DobsonDoigGraphDataset
    from polygraph.datasets.sbm import ProceduralSBMGraphDataset

    if dataset == "planar":
        return list(ProceduralPlanarGraphDataset(split=split, num_graphs=4096).to_nx())
    elif dataset == "lobster":
        return list(ProceduralLobsterGraphDataset(split=split, num_graphs=4096).to_nx())
    elif dataset == "sbm":
        return list(ProceduralSBMGraphDataset(split=split, num_graphs=4096).to_nx())
    elif dataset == "proteins":
        return list(DobsonDoigGraphDataset(split=split).to_nx())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


class ConcatenatedDescriptor:
    """Descriptor that concatenates multiple descriptor outputs with PCA reduction."""

    def __init__(self, descriptors: Dict, max_features: int = 500):
        self.descriptors = descriptors
        self.max_features = max_features
        self._pca = None
        self._is_fitted = False

    def __call__(self, graphs: Iterable[nx.Graph]) -> np.ndarray:
        from sklearn.decomposition import PCA

        graphs = list(graphs)
        features = []
        for name, desc in self.descriptors.items():
            feat = desc(graphs)
            # Handle sparse matrices
            if hasattr(feat, 'toarray'):
                feat = feat.toarray()
            features.append(feat)

        concatenated = np.hstack(features)

        # Apply PCA if features exceed max_features
        if concatenated.shape[1] > self.max_features:
            n_components = min(self.max_features, concatenated.shape[0] - 1)
            if self._pca is None:
                self._pca = PCA(n_components=n_components)
                concatenated = self._pca.fit_transform(concatenated)
            else:
                concatenated = self._pca.transform(concatenated)

        return concatenated


def compute_pgs_standard(reference_graphs: List, generated_graphs: List, subset: bool = False) -> Dict:
    """Compute standard PGD (max over individual descriptors)."""
    from sklearn.linear_model import LogisticRegression

    from polygraph.metrics.base import PolyGraphDiscrepancyInterval
    from polygraph.utils.descriptors import (
        ClusteringHistogram,
        EigenvalueHistogram,
        OrbitCounts,
        RandomGIN,
        SparseDegreeHistogram,
    )

    if subset:
        reference_graphs = reference_graphs[:30]
        generated_graphs = generated_graphs[:30]
        subsample_size = 15
        num_samples = 3
    else:
        subsample_size = min(len(reference_graphs), len(generated_graphs)) // 4
        num_samples = 10

    # Use same descriptors as concatenated for fair comparison (skip orbit5 in subset mode)
    if subset:
        descriptors = {
            "orbit4": OrbitCounts(graphlet_size=4),
            "degree": SparseDegreeHistogram(),
            "spectral": EigenvalueHistogram(),
            "clustering": ClusteringHistogram(bins=100),
            "gin": RandomGIN(seed=42),
        }
    else:
        descriptors = {
            "orbit4": OrbitCounts(graphlet_size=4),
            "orbit5": OrbitCounts(graphlet_size=5),
            "degree": SparseDegreeHistogram(),
            "spectral": EigenvalueHistogram(),
            "clustering": ClusteringHistogram(bins=100),
            "gin": RandomGIN(seed=42),
        }

    classifier = LogisticRegression(max_iter=1000, solver="lbfgs")

    metric = PolyGraphDiscrepancyInterval(
        reference_graphs,
        descriptors=descriptors,
        subsample_size=subsample_size,
        num_samples=num_samples,
        variant="jsd",
        classifier=classifier,
    )
    result = metric.compute(generated_graphs)

    return {
        "pgs_standard_mean": result["pgd"].mean,
        "pgs_standard_std": result["pgd"].std,
    }


def compute_pgs_concatenated(reference_graphs: List, generated_graphs: List, subset: bool = False) -> Dict:
    """Compute concatenated PGD (all descriptors as one feature vector)."""
    from sklearn.linear_model import LogisticRegression

    from polygraph.metrics.base import PolyGraphDiscrepancyInterval
    from polygraph.utils.descriptors import (
        ClusteringHistogram,
        EigenvalueHistogram,
        OrbitCounts,
        RandomGIN,
        SparseDegreeHistogram,
    )

    if subset:
        reference_graphs = reference_graphs[:30]
        generated_graphs = generated_graphs[:30]
        subsample_size = 15
        num_samples = 3
    else:
        subsample_size = min(len(reference_graphs), len(generated_graphs)) // 4
        num_samples = 10

    # Create concatenated descriptor with PCA reduction (skip orbit5 in subset mode)
    if subset:
        desc_dict = {
            "orbit4": OrbitCounts(graphlet_size=4),
            "degree": SparseDegreeHistogram(),
            "spectral": EigenvalueHistogram(),
            "clustering": ClusteringHistogram(bins=100),
            "gin": RandomGIN(seed=42),
        }
    else:
        desc_dict = {
            "orbit4": OrbitCounts(graphlet_size=4),
            "orbit5": OrbitCounts(graphlet_size=5),
            "degree": SparseDegreeHistogram(),
            "spectral": EigenvalueHistogram(),
            "clustering": ClusteringHistogram(bins=100),
            "gin": RandomGIN(seed=42),
        }
    concat_desc = ConcatenatedDescriptor(desc_dict, max_features=100)

    # Use LogisticRegression since concatenated features can be high-dimensional
    classifier = LogisticRegression(max_iter=1000, solver="lbfgs")

    metric = PolyGraphDiscrepancyInterval(
        reference_graphs,
        descriptors={"concatenated": concat_desc},
        subsample_size=subsample_size,
        num_samples=num_samples,
        variant="jsd",
        classifier=classifier,
    )

    result = metric.compute(generated_graphs)

    return {
        "pgs_concatenated_mean": result["pgd"].mean,
        "pgs_concatenated_std": result["pgd"].std,
    }


def format_value(mean: float, std: float, is_best: bool = False, is_second: bool = False) -> str:
    """Format a metric value with optional styling."""
    if pd.isna(mean):
        return "-"

    # Multiply by 100 for display
    mean_scaled = mean * 100
    std_scaled = std * 100

    text = f"{mean_scaled:.1f} $\\pm\\,\\scriptstyle{{{std_scaled:.1f}}}$"

    if is_best:
        return f"\\textbf{{{text}}}"
    elif is_second:
        return f"\\underline{{{text}}}"
    return text


def find_best_models(results: Dict[str, Dict], metric_key: str) -> Tuple[Optional[str], Optional[str]]:
    """Find best and second-best models for a metric (lower is better for PGD)."""
    values = {}
    for model, metrics in results.items():
        if metric_key in metrics and not pd.isna(metrics[metric_key]):
            values[model] = metrics[metric_key]

    if not values:
        return None, None

    sorted_models = sorted(values.keys(), key=lambda m: values[m])  # Lower is better
    best = sorted_models[0] if len(sorted_models) > 0 else None
    second = sorted_models[1] if len(sorted_models) > 1 else None
    return best, second


def generate_latex_table(all_results: Dict) -> str:
    """Generate LaTeX table from results."""
    lines = []
    lines.append("\\begin{table*}")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of standard PGD (max over descriptors) vs concatenated PGD (single classifier on all features).}")
    lines.append("\\label{tab:concatenation}")
    lines.append("\\begin{tabular}{llcc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Dataset} & \\textbf{Model} & \\textbf{PGD-Standard ($\\downarrow$)} & \\textbf{PGD-Concat ($\\downarrow$)} \\\\")
    lines.append("\\midrule")

    for dataset in DATASETS:
        dataset_results = all_results.get(dataset, {})

        # Find best models
        std_best, std_second = find_best_models(dataset_results, "pgs_standard_mean")
        cat_best, cat_second = find_best_models(dataset_results, "pgs_concatenated_mean")

        first_model = True
        for model in MODELS:
            if model not in dataset_results:
                continue

            results = dataset_results[model]

            row = []
            if first_model:
                row.append(DATASET_DISPLAY.get(dataset, dataset))
                first_model = False
            else:
                row.append("")

            row.append(MODEL_DISPLAY.get(model, model))

            # Standard PGD
            std_mean = results.get("pgs_standard_mean", float("nan"))
            std_std = results.get("pgs_standard_std", float("nan"))
            row.append(format_value(std_mean, std_std, model == std_best, model == std_second))

            # Concatenated PGD
            cat_mean = results.get("pgs_concatenated_mean", float("nan"))
            cat_std = results.get("pgs_concatenated_std", float("nan"))
            row.append(format_value(cat_mean, cat_std, model == cat_best, model == cat_second))

            lines.append(" & ".join(row) + " \\\\")

        if dataset != DATASETS[-1]:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


@app.command()
def main(
    subset: bool = typer.Option(False, "--subset", help="Use smaller sample for quick testing"),
    output: Path = typer.Option(OUTPUT_DIR / "concatenation.tex", "--output", "-o"),
):
    """Generate concatenation ablation tables from pre-generated graphs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset in tqdm(DATASETS, desc="Datasets"):
        print(f"\nProcessing {dataset}...")

        # Load reference dataset
        try:
            reference_graphs = get_reference_dataset(dataset, split="test")
        except Exception as e:
            print(f"  Error loading reference dataset: {e}")
            continue

        all_results[dataset] = {}

        for model in tqdm(MODELS, desc="Models", leave=False):
            print(f"  {model}...")

            # Load generated graphs
            generated_graphs = load_graphs(model, dataset)
            if not generated_graphs:
                print(f"    No graphs found for {model}/{dataset}")
                continue

            # Limit to reference size
            generated_graphs = generated_graphs[:len(reference_graphs)]

            results = {}

            # Compute standard PGD
            try:
                std_results = compute_pgs_standard(reference_graphs, generated_graphs, subset=subset)
                results.update(std_results)
            except Exception as e:
                print(f"    Error computing standard PGD: {e}")
                import traceback
                traceback.print_exc()

            # Compute concatenated PGD
            try:
                cat_results = compute_pgs_concatenated(reference_graphs, generated_graphs, subset=subset)
                results.update(cat_results)
            except Exception as e:
                print(f"    Error computing concatenated PGD: {e}")
                import traceback
                traceback.print_exc()

            all_results[dataset][model] = results

    # Generate and save LaTeX table
    latex_table = generate_latex_table(all_results)

    with open(output, "w") as f:
        f.write(latex_table)

    print(f"\nTable saved to: {output}")


if __name__ == "__main__":
    app()
