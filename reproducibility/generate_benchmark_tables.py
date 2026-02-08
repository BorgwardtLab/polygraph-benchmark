#!/usr/bin/env python3
"""Generate benchmark tables (Table 1) from pre-generated graphs.

This script computes PGD metrics for all models and datasets,
then formats the results as LaTeX tables.

Usage:
    python generate_benchmark_tables.py
    python generate_benchmark_tables.py --subset  # Use smaller sample for testing
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

    # Convert to networkx graphs based on format
    cleaned = []
    for g in graphs:
        if isinstance(g, nx.Graph):
            # Already a networkx graph (AUTOGRAPH, GRAN, etc.)
            simple = nx.Graph(g)
        elif isinstance(g, (list, tuple)) and len(g) == 2:
            # [node_features, adjacency_matrix] format (DIGRESS)
            try:
                node_feat, adj = g
                if isinstance(adj, torch.Tensor):
                    adj = adj.numpy()
                # Create graph from adjacency matrix
                simple = nx.from_numpy_array(adj)
            except Exception as e:
                print(f"    Warning: Could not convert graph: {e}")
                continue
        else:
            print(f"    Warning: Unknown graph format: {type(g)}")
            continue

        # Clean: remove self-loops
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


def compute_pgs_metrics(reference_graphs: List, generated_graphs: List, subset: bool = False) -> Dict:
    """Compute PGD metrics using the polygraph library."""
    from polygraph.metrics import StandardPGDInterval

    # Limit samples for subset mode
    if subset:
        reference_graphs = reference_graphs[:50]
        generated_graphs = generated_graphs[:50]
        subsample_size = 20
        num_samples = 5
    else:
        subsample_size = min(len(reference_graphs), len(generated_graphs)) // 4
        num_samples = 10

    metric = StandardPGDInterval(reference_graphs, subsample_size=subsample_size, num_samples=num_samples)
    result = metric.compute(generated_graphs)

    # Convert to dict format compatible with table generation
    return {
        "polyscore_mean": result["pgd"].mean,
        "polyscore_std": result["pgd"].std,
        "subscores": {
            name: {"mean": interval.mean, "std": interval.std}
            for name, interval in result["subscores"].items()
        }
    }


def compute_vun_metrics(train_graphs: List, generated_graphs: List, dataset: str, subset: bool = False) -> Optional[Dict]:
    """Compute VUN metrics for datasets that support validity checking."""
    from polygraph.datasets.lobster import is_lobster_graph
    from polygraph.datasets.planar import is_planar_graph
    from polygraph.datasets.sbm import is_sbm_graph
    from polygraph.metrics import VUN

    validity_fns = {
        "planar": is_planar_graph,
        "lobster": is_lobster_graph,
        "sbm": is_sbm_graph,
    }

    if dataset not in validity_fns:
        return None

    if subset:
        train_graphs = train_graphs[:50]
        generated_graphs = generated_graphs[:50]

    vun_metric = VUN(
        train_graphs=train_graphs,
        validity_fn=validity_fns[dataset],
    )

    results = vun_metric.compute(generated_graphs)
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


def find_best_models(results: Dict[str, Dict], metric_key: str, higher_is_better: bool) -> Tuple[str, str]:
    """Find best and second-best models for a metric."""
    values = {}
    for model, metrics in results.items():
        if metric_key in metrics and not pd.isna(metrics[metric_key]):
            values[model] = metrics[metric_key]

    if not values:
        return None, None

    sorted_models = sorted(values.keys(), key=lambda m: values[m], reverse=higher_is_better)
    best = sorted_models[0] if len(sorted_models) > 0 else None
    second = sorted_models[1] if len(sorted_models) > 1 else None
    return best, second


def compute_benchmark_task(
    dataset: str, model: str, subset: bool
) -> Dict:
    """Compute benchmark metrics for one (dataset, model) pair.

    Designed to run as a SLURM job via submitit.
    """
    reference_graphs = get_reference_dataset(
        dataset, split="test"
    )
    train_graphs = get_reference_dataset(
        dataset, split="train"
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
        pgs_results = compute_pgs_metrics(
            reference_graphs, generated_graphs, subset=subset
        )
        results["pgs_mean"] = pgs_results.get(
            "polyscore_mean", float("nan")
        )
        results["pgs_std"] = pgs_results.get(
            "polyscore_std", float("nan")
        )
        for key, value in pgs_results.get(
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
        print(f"Error computing PGD for {model}/{dataset}: {e}")

    try:
        vun_results = compute_vun_metrics(
            train_graphs,
            generated_graphs,
            dataset,
            subset=subset,
        )
        if vun_results:
            results["vun"] = vun_results.get(
                "valid_unique_novel_mle", float("nan")
            )
    except Exception as e:
        print(f"Error computing VUN for {model}/{dataset}: {e}")

    return results


def generate_latex_table(all_results: Dict) -> str:
    """Generate LaTeX table from results."""
    lines = []
    lines.append("\\begin{table*}")
    lines.append("\\centering")
    lines.append("\\caption{Mean PGD $\\pm$ standard deviation across synthetic and real-world graphs. Values are multiplied by 100 for readability.}")
    lines.append("\\label{tab:pgs_benchmark}")
    lines.append("\\begin{tabular}{llccccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Dataset} & \\textbf{Model} & \\textbf{VUN ($\\uparrow$)} & \\textbf{PGD ($\\downarrow$)} & \\textbf{Clust.} & \\textbf{Deg.} & \\textbf{GIN} & \\textbf{Orb5.} & \\textbf{Eig.} \\\\")
    lines.append("\\midrule")

    for dataset in DATASETS:
        dataset_results = all_results.get(dataset, {})

        # Find best models for PGD (lower is better)
        pgs_best, pgs_second = find_best_models(dataset_results, "pgs_mean", higher_is_better=False)
        vun_best, vun_second = find_best_models(dataset_results, "vun", higher_is_better=True)

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

            # VUN
            if "vun" in results:
                vun_val = results["vun"] * 100
                vun_text = f"{vun_val:.1f}"
                if model == vun_best:
                    vun_text = f"\\textbf{{{vun_text}}}"
                elif model == vun_second:
                    vun_text = f"\\underline{{{vun_text}}}"
                row.append(vun_text)
            else:
                row.append("-")

            # PGD and subscores
            pgs_mean = results.get("pgs_mean", float("nan"))
            pgs_std = results.get("pgs_std", float("nan"))
            is_best = model == pgs_best
            is_second = model == pgs_second
            row.append(format_value(pgs_mean, pgs_std, is_best, is_second))

            # Subscores
            for subscore in ["clustering_pgs", "degree_pgs", "gin_pgs", "orbit5_pgs", "spectral_pgs"]:
                mean_key = f"{subscore}_mean"
                std_key = f"{subscore}_std"
                mean_val = results.get(mean_key, float("nan"))
                std_val = results.get(std_key, float("nan"))
                row.append(format_value(mean_val, std_val))

            lines.append(" & ".join(row) + " \\\\")

        if dataset != DATASETS[-1]:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
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


LOG_DIR = Path(__file__).parent / "logs" / "benchmark"
JOBS_FILE = LOG_DIR / "jobs.json"


@app.command()
def main(
    subset: bool = typer.Option(
        False,
        "--subset",
        help="Use smaller sample for quick testing",
    ),
    output: Path = typer.Option(
        OUTPUT_DIR / "benchmark_results.tex",
        "--output",
        "-o",
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
    """Generate benchmark tables from pre-generated graphs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect mode: gather results from completed SLURM jobs
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
            compute_benchmark_task,
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
            train_graphs = get_reference_dataset(dataset, split="train")
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
                pgs_results = compute_pgs_metrics(reference_graphs, generated_graphs, subset=subset)
                results["pgs_mean"] = pgs_results.get("polyscore_mean", float("nan"))
                results["pgs_std"] = pgs_results.get("polyscore_std", float("nan"))

                subscores = pgs_results.get("subscores", {})
                for key, value in subscores.items():
                    if isinstance(value, dict):
                        results[f"{key}_mean"] = value.get("mean", float("nan"))
                        results[f"{key}_std"] = value.get("std", float("nan"))
            except Exception as e:
                print(f"    Error computing PGD: {e}")

            try:
                vun_results = compute_vun_metrics(train_graphs, generated_graphs, dataset, subset=subset)
                if vun_results:
                    results["vun"] = vun_results.get("valid_unique_novel_mle", float("nan"))
            except Exception as e:
                print(f"    Error computing VUN: {e}")

            all_results[dataset][model] = results

    latex_table = generate_latex_table(all_results)

    with open(output, "w") as f:
        f.write(latex_table)

    print(f"\nTable saved to: {output}")


if __name__ == "__main__":
    app()
