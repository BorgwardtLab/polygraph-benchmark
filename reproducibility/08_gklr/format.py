#!/usr/bin/env python3
"""Format GKLR results into LaTeX table.

Produces: tables/gklr.tex (matching paper format)

Usage:
    python format.py
    python format.py --paper-dir /path/to/paper/tables/
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import typer
from loguru import logger
from pyprojroot import here

sys.path.insert(0, str(here() / "reproducibility"))
from utils.formatting import (
    DATASETS,
    DATASET_DISPLAY,
    MODEL_DISPLAY,
    MODELS,
    best_two,
    fmt_pgs,
)

app = typer.Typer()

REPO_ROOT = here()
OUTPUT_DIR = REPO_ROOT / "reproducibility" / "tables"
RESULTS_DIR = OUTPUT_DIR / "results" / "gklr"
BENCHMARK_DIR = OUTPUT_DIR / "results" / "benchmark"

KERNEL_SUBSCORES = [
    ("pyramid_match", "PM"),
    ("shortest_path", "SP"),
    ("wl", "WL"),
]


def load_results(results_dir: Path) -> Dict[str, Dict]:
    all_r: Dict[str, Dict] = {}
    if not results_dir.exists():
        return all_r
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            r = json.load(fh)
        ds, model = r.get("dataset"), r.get("model")
        if ds and model:
            all_r.setdefault(ds, {})[model] = r
    return all_r


_fmt = fmt_pgs
_best_two = best_two


def generate_table(gklr_results: Dict, bench_results: Dict) -> str:
    lines = []
    lines.append("\\resizebox{\\columnwidth}{!}{")
    lines.append("\\begin{tabular}{lllllll}")
    lines.append("        \\toprule")
    lines.append(
        "        \\textbf{Dataset} & \\textbf{Model} & & & \\multicolumn{3}{c}{\\textbf{Subscores}} \\\\"
    )
    lines.append("        \\cmidrule(lr){5-7}")
    lines.append(
        "         &  & \\textbf{PGD ($\\downarrow$)} & \\textbf{PGD-GKLR ($\\downarrow$)} & "
        + " & ".join(
            f"\\textbf{{{label} ($\\downarrow$)}}"
            for _, label in KERNEL_SUBSCORES
        )
        + " \\\\"
    )
    lines.append("        \\midrule")

    for i, ds in enumerate(DATASETS):
        ds_gklr = gklr_results.get(ds, {})
        ds_bench = bench_results.get(ds, {})

        pgd_best, pgd_second = _best_two(ds_bench, "pgs_mean", lower=True)
        gklr_best, gklr_second = _best_two(ds_gklr, "pgs_mean", lower=True)

        kernel_rankings = {}
        for key, _ in KERNEL_SUBSCORES:
            kernel_rankings[key] = _best_two(ds_gklr, f"{key}_mean", lower=True)

        first = True
        for model in MODELS:
            gr = ds_gklr.get(model, {})
            br = ds_bench.get(model, {})

            row = []
            row.append(
                f"        {DATASET_DISPLAY[ds]}" if first else "        "
            )
            first = False
            row.append(MODEL_DISPLAY.get(model, model))

            row.append(
                _fmt(
                    br.get("pgs_mean", float("nan")),
                    br.get("pgs_std", float("nan")),
                    model == pgd_best,
                    model == pgd_second,
                )
            )

            row.append(
                _fmt(
                    gr.get("pgs_mean", float("nan")),
                    gr.get("pgs_std", float("nan")),
                    model == gklr_best,
                    model == gklr_second,
                )
            )

            for key, _ in KERNEL_SUBSCORES:
                k_best, k_second = kernel_rankings[key]
                row.append(
                    _fmt(
                        gr.get(f"{key}_mean", float("nan")),
                        gr.get(f"{key}_std", float("nan")),
                        model == k_best,
                        model == k_second,
                    )
                )

            lines.append(" & ".join(row) + " \\\\")
        if i < len(DATASETS) - 1:
            lines.append("        \\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    return "\n".join(lines)


@app.command()
def main(
    paper_dir: Optional[Path] = typer.Option(None, "--paper-dir"),
):
    gklr_results = load_results(RESULTS_DIR)
    bench_results = load_results(BENCHMARK_DIR)

    if not gklr_results:
        logger.error("No GKLR results in {}.", RESULTS_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = generate_table(gklr_results, bench_results)
    out = OUTPUT_DIR / "gklr.tex"
    out.write_text(table)
    logger.success("Saved: {}", out)

    if paper_dir:
        import shutil

        Path(paper_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, Path(paper_dir) / "gklr.tex")


if __name__ == "__main__":
    app()
