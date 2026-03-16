#!/usr/bin/env python3
"""Plot phase plot figure from pre-computed JSON or raw CSV results.

Reproduces: figures/phase_plot/phase_plot.pdf
  - VUN vs Validation Loss colored by training step for two SBM dataset sizes.

Usage:
    python plot.py
    python plot.py --paper-dir /path/to/paper/figures/phase_plot/
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import typer
from loguru import logger
from matplotlib.collections import LineCollection
from pyprojroot import here

app = typer.Typer()

REPO_ROOT = here()
OUTPUT_DIR = REPO_ROOT / "reproducibility" / "figures" / "04_phase_plot"
RESULTS_DIR = OUTPUT_DIR / "results"
DATA_DIR = REPO_ROOT / "data"
STYLE_FILE = Path(__file__).resolve().parent.parent / "polygraph.mplstyle"


def load_phase_data():
    """Load phase plot data from JSON results or fall back to raw CSV logs."""
    json_path = RESULTS_DIR / "phase_plot.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    results = {}
    log_files = {
        "sbm_small": DATA_DIR
        / "AUTOGRAPH"
        / "logs"
        / "sbm_proc_small_metrics.csv",
        "sbm_large": DATA_DIR
        / "AUTOGRAPH"
        / "logs"
        / "sbm_proc_large_metrics.csv",
    }
    for name, path in log_files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        x = df["val/loss_epoch"].dropna()
        y = df["val/valid_unique_novel_mle"].dropna()
        n = min(len(x), len(y))
        results[name] = {
            "val_loss": x.iloc[:n].tolist(),
            "vun": y.iloc[:n].tolist(),
            "steps": list(range(n)),
        }
    return results if results else None


@app.command()
def main(
    paper_dir: Path = typer.Option(
        None, "--paper-dir", help="Copy output into paper figures dir"
    ),
):
    """Generate phase plot matching the paper exactly."""
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))

    data = load_phase_data()
    if not data:
        logger.error(
            "No phase plot data found. Run compute.py or check data/AUTOGRAPH/logs/."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 2.8))
    ax.set_box_aspect(1)

    for series_name, cmap_name, edge_color, label in [
        ("sbm_small", "autumn", "black", "SBM-S"),
        ("sbm_large", "winter", "white", "SBM-L"),
    ]:
        if series_name not in data:
            logger.warning("Missing series: {}", series_name)
            continue

        series = data[series_name]
        x_vals = np.array(series["val_loss"])
        y_vals = np.array(series["vun"])
        indices = np.arange(len(x_vals))

        points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = Normalize(indices.min(), indices.max())
        lc = LineCollection(segments, cmap=cmap_name, norm=norm)  # type: ignore[arg-type]
        lc.set_array(indices[:-1])
        lc.set_linewidth(1)
        ax.add_collection(lc)

        ax.scatter(
            x_vals,
            y_vals,
            c=indices,
            cmap=cmap_name,
            s=15,
            alpha=0.7,
            edgecolors=edge_color,
            linewidth=0.3,
            label=label,
        )

    all_x = []
    all_y = []
    for series_name in ["sbm_small", "sbm_large"]:
        if series_name in data:
            all_x.extend(data[series_name]["val_loss"])
            all_y.extend(data[series_name]["vun"])

    if all_x:
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        ax.set_xlim(all_x.min() * 0.99, all_x.max() * 1.01)
        ax.set_ylim(all_y.min() * 0.95, all_y.max() * 1.05)

    if "sbm_small" in data:
        indices_small = np.arange(len(data["sbm_small"]["val_loss"]))
        norm_small = Normalize(indices_small.min(), indices_small.max())
        sm_small = plt.cm.ScalarMappable(cmap="autumn", norm=norm_small)
        sm_small.set_array(indices_small)
        cbar_small = fig.colorbar(sm_small, ax=ax, aspect=30, pad=0.02)
        cbar_small.set_label("SBM-S - Time Step", rotation=270, labelpad=15)

    if "sbm_large" in data:
        indices_large = np.arange(len(data["sbm_large"]["val_loss"]))
        norm_large = Normalize(indices_large.min(), indices_large.max())
        sm_large = plt.cm.ScalarMappable(cmap="winter", norm=norm_large)
        sm_large.set_array(indices_large)
        cbar_large = fig.colorbar(sm_large, ax=ax, aspect=30, pad=0.08)
        cbar_large.set_label("SBM-L - Time Step", rotation=270, labelpad=15)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xlabel("Validation Loss")
    ax.set_ylabel("VUN")
    ax.set_title("Validation loss vs VUN")

    out_path = OUTPUT_DIR / "phase_plot.pdf"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    logger.success("Saved: {}", out_path)

    if paper_dir is not None:
        import shutil

        paper_dir = Path(paper_dir)
        paper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, paper_dir / "phase_plot.pdf")
        logger.success("Copied to {}", paper_dir / "phase_plot.pdf")


if __name__ == "__main__":
    app()
