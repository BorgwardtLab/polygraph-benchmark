#!/usr/bin/env python3
"""Generate GKLR (Graph Kernel Logistic Regression) tables from pre-generated graphs.

This script computes PGD metrics using kernel logistic regression with
Weisfeiler-Lehman and Shortest Path kernels, then formats the results as LaTeX tables.

Usage:
    python generate_gklr_tables.py
    python generate_gklr_tables.py --subset  # Use smaller sample for testing
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import typer
from tqdm import tqdm

from cluster import (
    SlurmConfig,
    collect_results,
    save_job_metadata,
    submit_jobs,
)

app = typer.Typer()

# Paths
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "polygraph_graphs"
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
    import networkx as nx
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


def compute_gklr_metrics(reference_graphs: List, generated_graphs: List, subset: bool = False) -> Dict:
    """Compute PGD metrics using logistic regression classifier with WL descriptor."""
    from sklearn.linear_model import LogisticRegression

    from polygraph.metrics.base import PolyGraphDiscrepancyInterval
    from polygraph.utils.descriptors import (
        ClusteringHistogram,
        EigenvalueHistogram,
        OrbitCounts,
        RandomGIN,
        SparseDegreeHistogram,
    )

    # Limit samples for subset mode
    if subset:
        reference_graphs = reference_graphs[:50]
        generated_graphs = generated_graphs[:50]
        subsample_size = 20
        num_samples = 5
    else:
        subsample_size = min(len(reference_graphs), len(generated_graphs)) // 4
        num_samples = 10

    # Standard descriptors with Logistic Regression classifier
    descriptors = {
        "orbit4": OrbitCounts(graphlet_size=4),
        "orbit5": OrbitCounts(graphlet_size=5),
        "degree": SparseDegreeHistogram(),
        "spectral": EigenvalueHistogram(),
        "clustering": ClusteringHistogram(bins=100),
        "gin": RandomGIN(seed=42),
    }

    # Use logistic regression classifier instead of TabPFN
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

    # Extract results - result has pgd (MetricInterval) and subscores (Dict[str, MetricInterval])
    return {
        "pgd_mean": result["pgd"].mean,
        "pgd_std": result["pgd"].std,
        "subscores": {
            name: {"mean": interval.mean, "std": interval.std}
            for name, interval in result["subscores"].items()
        }
    }


def compute_gklr_task(
    dataset: str, model: str, subset: bool
) -> Dict:
    """Compute GKLR metrics for one (dataset, model) pair.

    Designed to run as a SLURM job via submitit.
    """
    reference_graphs = get_reference_dataset(
        dataset, split="test"
    )
    generated_graphs = load_graphs(model, dataset)
    if not generated_graphs:
        return {
            "dataset": dataset,
            "model": model,
            "error": "no graphs",
        }

    generated_graphs = generated_graphs[
        : len(reference_graphs)
    ]
    results: Dict = {"dataset": dataset, "model": model}

    try:
        gklr_results = compute_gklr_metrics(
            reference_graphs, generated_graphs, subset=subset
        )
        results["pgs_mean"] = gklr_results.get(
            "pgd_mean", float("nan")
        )
        results["pgs_std"] = gklr_results.get(
            "pgd_std", float("nan")
        )
        for key, value in gklr_results.get(
            "subscores", {}
        ).items():
            if isinstance(value, dict):
                results[f"{key}_mean"] = value.get(
                    "mean", float("nan")
                )
                results[f"{key}_std"] = value.get(
                    "std", float("nan")
                )
    except Exception as e:
        print(f"Error computing GKLR for {model}/{dataset}: {e}")

    return results


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
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{0.9}")
    lines.append("\\scalebox{0.7}{")
    lines.append("\\begin{minipage}{\\textwidth}")
    lines.append("\\caption{PGD metrics using Kernel Logistic Regression with Weisfeiler-Lehman and Shortest Path kernels.}")
    lines.append("\\label{tab:gklr}")

    # Metrics to display
    metrics = ["pgs", "clustering", "degree", "gin", "orbit4", "orbit5", "spectral"]
    metric_display = {
        "pgs": "PGD-LR",
        "clustering": "Clust.",
        "degree": "Deg.",
        "gin": "GIN",
        "orbit4": "Orb4.",
        "orbit5": "Orb5.",
        "spectral": "Eig.",
    }

    lines.append("\\begin{tabular}{ll" + "c" * len(metrics) + "}")
    lines.append("\\toprule")

    # Header
    header = ["\\textbf{Dataset}", "\\textbf{Model}"]
    for m in metrics:
        header.append(f"\\textbf{{{metric_display.get(m, m)} ($\\downarrow$)}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for dataset in DATASETS:
        dataset_results = all_results.get(dataset, {})

        # Find best models for PGS (lower is better)
        pgs_best, pgs_second = find_best_models(dataset_results, "pgs_mean")

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

            # PGS and subscores
            for m in metrics:
                mean_key = f"{m}_mean"
                std_key = f"{m}_std"
                mean_val = results.get(mean_key, float("nan"))
                std_val = results.get(std_key, float("nan"))
                is_best = (m == "pgs" and model == pgs_best)
                is_second = (m == "pgs" and model == pgs_second)
                row.append(format_value(mean_val, std_val, is_best, is_second))

            lines.append(" & ".join(row) + " \\\\")

        if dataset != DATASETS[-1]:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{minipage}")
    lines.append("}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def _reshape_results(
    result_list: List[Dict],
) -> Dict[str, Dict]:
    """Reshape a flat list of result dicts into nested dict."""
    all_results: Dict[str, Dict] = {}
    for r in result_list:
        ds = r.pop("dataset", None)
        model = r.pop("model", None)
        r.pop("error", None)
        if ds and model:
            all_results.setdefault(ds, {})[model] = r
    return all_results


LOG_DIR = Path(__file__).parent / "logs" / "gklr"
JOBS_FILE = LOG_DIR / "jobs.json"


@app.command()
def main(
    subset: bool = typer.Option(
        False,
        "--subset",
        help="Use smaller sample for quick testing",
    ),
    output: Path = typer.Option(
        OUTPUT_DIR / "gklr.tex", "--output", "-o"
    ),
    slurm_config: Optional[Path] = typer.Option(
        None,
        "--slurm-config",
        help="Path to SLURM YAML config. Submits jobs to cluster.",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Run submitit jobs in-process (for debugging).",
    ),
    collect: bool = typer.Option(
        False,
        "--collect",
        help="Collect results from previously submitted SLURM jobs.",
    ),
):
    """Generate GKLR tables from pre-generated graphs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect mode
    if collect:
        result_list = collect_results(JOBS_FILE, LOG_DIR)
        all_results = _reshape_results(result_list)
        latex_table = generate_latex_table(all_results)
        with open(output, "w") as f:
            f.write(latex_table)
        print(f"\nTable saved to: {output}")
        return

    # SLURM submission mode
    if slurm_config is not None:
        config = SlurmConfig.from_yaml(slurm_config)
        args_list = [
            (dataset, model, subset)
            for dataset in DATASETS
            for model in MODELS
        ]
        jobs = submit_jobs(
            compute_gklr_task,
            args_list,
            config,
            log_dir=LOG_DIR,
            local=local,
        )

        if local:
            result_list = [job.result() for job in jobs]
            all_results = _reshape_results(result_list)
            latex_table = generate_latex_table(all_results)
            with open(output, "w") as f:
                f.write(latex_table)
            print(f"\nTable saved to: {output}")
        else:
            save_job_metadata(jobs, args_list, JOBS_FILE)
            print(
                "\nRun with --collect after jobs complete."
            )
        return

    # Default: run locally without submitit
    all_results = {}

    for dataset in tqdm(DATASETS, desc="Datasets"):
        print(f"\nProcessing {dataset}...")

        try:
            reference_graphs = get_reference_dataset(dataset, split="test")
        except Exception as e:
            print(f"  Error loading reference dataset: {e}")
            continue

        all_results[dataset] = {}

        for model in tqdm(MODELS, desc="Models", leave=False):
            print(f"  {model}...")

            generated_graphs = load_graphs(model, dataset)
            if not generated_graphs:
                print(f"    No graphs found for {model}/{dataset}")
                continue

            generated_graphs = generated_graphs[:len(reference_graphs)]

            results = {}

            try:
                gklr_results = compute_gklr_metrics(reference_graphs, generated_graphs, subset=subset)
                results["pgs_mean"] = gklr_results.get("pgd_mean", float("nan"))
                results["pgs_std"] = gklr_results.get("pgd_std", float("nan"))

                subscores = gklr_results.get("subscores", {})
                for key, value in subscores.items():
                    if isinstance(value, dict):
                        results[f"{key}_mean"] = value.get("mean", float("nan"))
                        results[f"{key}_std"] = value.get("std", float("nan"))
            except Exception as e:
                print(f"    Error computing GKLR: {e}")
                import traceback
                traceback.print_exc()

            all_results[dataset][model] = results

    latex_table = generate_latex_table(all_results)

    with open(output, "w") as f:
        f.write(latex_table)

    print(f"\nTable saved to: {output}")


if __name__ == "__main__":
    app()
