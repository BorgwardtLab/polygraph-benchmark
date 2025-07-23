import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from pyprojroot import here

# Load and apply rcparams
rcparams_path = "/fs/pool/pool-mlsb/polygraph/rcparams.json"
with open(rcparams_path, "r") as f:
    rcparams = json.load(f)
plt.rcParams.update(rcparams)

autograph_sbm_proc_small = "/fs/pool/pool-hartout/Documents/Git/AutoGraph/logs/train/polygraph_sbm_procedural_small/llama2-s/0/runs/2025-07-21_12-03-47/csv_logs/version_0/metrics.csv"
autograph_sbm_proc_large = "/fs/pool/pool-hartout/Documents/Git/AutoGraph/logs/train/polygraph_sbm_procedural/llama2-s/0/runs/2025-07-21_12-03-47/csv_logs/version_0/metrics.csv"


def main():
    df_small = pd.read_csv(autograph_sbm_proc_small)
    df_large = pd.read_csv(autograph_sbm_proc_large)

    # Extract the required columns
    x_small = df_small["val/loss"].dropna()
    y_small = df_small["val/valid_unique_novel_mle"].dropna()

    x_large = df_large["val/loss"].dropna()
    y_large = df_large["val/valid_unique_novel_mle"].dropna()

    # Align the data (in case there are different numbers of valid entries)
    min_len = min(len(x_small), len(y_small))
    x_small = x_small.iloc[:min_len]
    y_small = y_small.iloc[:min_len]
    indices_small = np.arange(min_len)

    min_len = min(len(x_large), len(y_large))
    x_large = x_large.iloc[:min_len]
    y_large = y_large.iloc[:min_len]
    indices_large = np.arange(min_len)

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 3))

    # Plot small values with summer colormap
    points_small = np.array([x_small.values, y_small.values]).T.reshape(
        -1, 1, 2
    )
    segments_small = np.concatenate(
        [points_small[:-1], points_small[1:]], axis=1
    )

    norm_small = plt.Normalize(indices_small.min(), indices_small.max())
    lc_small = LineCollection(segments_small, cmap="autumn", norm=norm_small)
    lc_small.set_array(indices_small[:-1])
    lc_small.set_linewidth(2)
    _ = ax.add_collection(lc_small)

    # Add scatter points for small values
    scatter_small = ax.scatter(
        x_small,
        y_small,
        c=indices_small,
        cmap="autumn",
        s=30,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        label="Small",
    )

    # Plot large values with winter colormap
    points_large = np.array([x_large.values, y_large.values]).T.reshape(
        -1, 1, 2
    )
    segments_large = np.concatenate(
        [points_large[:-1], points_large[1:]], axis=1
    )

    norm_large = plt.Normalize(indices_large.min(), indices_large.max())
    lc_large = LineCollection(segments_large, cmap="winter", norm=norm_large)
    lc_large.set_array(indices_large[:-1])
    lc_large.set_linewidth(2)
    _ = ax.add_collection(lc_large)

    # Add scatter points for large values
    scatter_large = ax.scatter(
        x_large,
        y_large,
        c=indices_large,
        cmap="winter",
        s=30,
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5,
        label="Large",
    )

    # Add colorbars
    cbar_small = fig.colorbar(
        scatter_small, ax=ax, shrink=0.6, aspect=30, pad=0.02
    )
    cbar_small.set_label("Small - Time Step", rotation=270, labelpad=15)

    cbar_large = fig.colorbar(
        scatter_large, ax=ax, shrink=0.6, aspect=30, pad=0.08
    )
    cbar_large.set_label("Large - Time Step", rotation=270, labelpad=15)

    # Set axis limits with some padding to accommodate both datasets
    all_x = np.concatenate([x_small, x_large])
    all_y = np.concatenate([y_small, y_large])
    ax.set_xlim(all_x.min() * 0.99, all_x.max() * 1.01)
    ax.set_ylim(all_y.min() * 0.95, all_y.max() * 1.05)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Set labels and title
    ax.set_xlabel("Validation Loss")
    ax.set_ylabel("VUN")
    ax.set_title("Phase Plot of the validation loss vs VUN")

    # Save the plot
    plt.savefig(
        here() / "experiments" / "phase_plot" / "phase_plot.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
