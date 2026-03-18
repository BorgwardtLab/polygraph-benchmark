#!/usr/bin/env python3
"""Compute VUN metrics for a single (dataset, model) combo and patch JSON results.

Uses a 10s per-pair timeout on isomorphism checks with parallel novelty
checking across generated graphs.

Usage:
    # Single combo
    python compute_vun.py --dataset sbm --model AUTOGRAPH

    # All missing combos (for local use)
    python compute_vun.py --all-missing
"""

import json
import sys
from typing import List, Optional

import typer
from loguru import logger
from pyprojroot import here

sys.path.insert(0, str(here() / "reproducibility"))
from utils.data import get_reference_dataset as _get_ref
from utils.data import load_graphs as _load
from utils.vun import compute_vun_parallel

REPO_ROOT = here()
DATA_DIR = REPO_ROOT / "data"
RESULTS_BASE = REPO_ROOT / "reproducibility" / "tables" / "results"

DATASETS = ["planar", "lobster", "sbm"]
MODELS = ["AUTOGRAPH", "DIGRESS", "GRAN", "ESGG"]

app = typer.Typer()


def _get_result_dirs() -> List:
    dirs = []
    for d in RESULTS_BASE.iterdir():
        if d.is_dir() and d.name.startswith("benchmark"):
            dirs.append(d)
    return dirs


def _has_vun(result_dirs, dataset, model) -> bool:
    for rdir in result_dirs:
        p = rdir / f"{dataset}_{model}.json"
        if p.exists():
            data = json.loads(p.read_text())
            if "vun" in data and data["vun"] is not None:
                return True
    return False


def _run_one(
    dataset: str, model: str, iso_timeout: int, n_workers: int, force: bool
):
    """Compute VUN for a single (dataset, model) and patch all result JSONs."""
    result_dirs = _get_result_dirs()
    if not force and _has_vun(result_dirs, dataset, model):
        logger.info("Skipping {}/{} (VUN already present)", model, dataset)
        return

    logger.info("Loading data for {}/{}...", model, dataset)
    train_graphs = _get_ref(dataset, split="train", num_graphs=8192)
    ref_graphs = _get_ref(dataset, split="test", num_graphs=4096)
    generated_graphs = _load(DATA_DIR, model, dataset)
    if not generated_graphs:
        logger.warning("No graphs for {}/{}", model, dataset)
        return
    generated_graphs = generated_graphs[: len(ref_graphs)]

    logger.info(
        "Computing VUN for {}/{} ({} gen, {} train)...",
        model,
        dataset,
        len(generated_graphs),
        len(train_graphs),
    )

    result = compute_vun_parallel(
        train_graphs,
        generated_graphs,
        dataset=dataset,
        iso_timeout=iso_timeout,
        n_workers=n_workers,
    )

    vun_value = result["valid_unique_novel"]
    logger.success("{}/{}: VUN = {:.4f}", model, dataset, vun_value)

    for rdir in result_dirs:
        json_path = rdir / f"{dataset}_{model}.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())
        data["vun"] = vun_value
        json_path.write_text(json.dumps(data, indent=2))
        logger.info("Patched {}", json_path)


@app.command()
def main(
    dataset: Optional[str] = typer.Option(
        None, help="Dataset name (planar/lobster/sbm)"
    ),
    model: Optional[str] = typer.Option(
        None, help="Model name (AUTOGRAPH/DIGRESS/GRAN/ESGG)"
    ),
    all_missing: bool = typer.Option(
        False, "--all-missing", help="Run all combos missing VUN"
    ),
    iso_timeout: int = typer.Option(
        10, help="Per-pair isomorphism timeout in seconds"
    ),
    n_workers: int = typer.Option(
        8, help="Number of parallel workers for novelty checks"
    ),
    force: bool = typer.Option(
        False, help="Recompute even if VUN already present"
    ),
):
    """Compute VUN for one or all dataset/model combos and patch JSON results."""
    if all_missing:
        for ds in DATASETS:
            for m in MODELS:
                _run_one(ds, m, iso_timeout, n_workers, force)
    elif dataset and model:
        _run_one(dataset, model, iso_timeout, n_workers, force)
    else:
        logger.error("Provide --dataset + --model, or --all-missing")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
