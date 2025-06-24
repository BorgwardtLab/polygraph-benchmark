import itertools
import json
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.ticker import MaxNLocator
from pyprojroot import here


def load_global_rcparams() -> dict:
    """Load global rcparams from JSON file."""
    with open("/fs/pool/pool-mlsb/polygraph/rcparams.json", "r") as f:
        return json.load(f)


def setup_plotting_parameters() -> None:
    """Setup plotting parameters using global rcparams file and seaborn."""
    # Load and apply global rcparams
    global_rcparams = load_global_rcparams()
    plt.rcParams.update(global_rcparams)

    # Set seaborn style
    sns.set_style("white")
    sns.set_palette("colorblind")

    # Log the applied font settings for debugging
    logger.info(
        f"Applied font settings - Family: {plt.rcParams.get('font.family')}, "
        f"UseTeX: {plt.rcParams.get('text.usetex')}, "
        f"MathFont: {plt.rcParams.get('mathtext.fontset')}"
    )


def to_color(element: str, list_of_elements: List[str]) -> str:
    """Map element to colorblind-friendly color."""
    colors = sns.color_palette("colorblind")
    return colors[list_of_elements.index(element)]


def format_dataset(dataset: str) -> str:
    if "lobster" in dataset:
        return "Lobster"
    elif "sbm" in dataset:
        return "SBM"
    elif "planar" in dataset:
        return "Planar"
    else:
        return "Unknown"


def format_descriptor(descriptor: str) -> str:
    """Format a descriptor string by splitting before the first capitalized character in camelCase."""
    if not descriptor or len(descriptor) <= 1:
        return descriptor

    for i in range(1, len(descriptor)):
        if descriptor[i].isupper():
            return descriptor[:i] + " " + descriptor[i:]

    return descriptor


def format_generated_set_name(generated_set_type: str, n_train: int) -> str:
    """Formats generated set name with model and training sample size."""
    if "gran" in generated_set_type:
        model_name = "GRAN"
    elif "autograph" in generated_set_type:
        model_name = "AutoGraph"
    elif "digress" in generated_set_type:
        model_name = "DiGress"
    else:
        raise ValueError(f"Unknown generated set type: {generated_set_type}")
    return f"{model_name} (Trained on {int(n_train)} Samples)"


def format_variant(variant: str) -> str:
    if variant == "umve":
        return "UMVE"
    elif variant == "biased":
        return "Biased"
    else:
        raise ValueError(f"Unknown variant: {variant}")


def format_test_set_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(
        {
            "test_set_type": {
                "test": "Test set",
                "digress_generated_procedural": "DiGress",
                "gran_generated_procedural": "GRAN",
                "autograph_generated_procedural": "AutoGraph",
            }
        }
    )


def plot_subsampling(df: pd.DataFrame, model: str, variant: str) -> None:
    """
    Create a multi-panel line plot for df restricted to a single model.

    Uses a similar setup as analysis.ipynb with subplot creation and
    custom formatting similar to the perturbation analysis plots.
    """
    df_model = df.loc[
        ((df.test_set_type.str.contains(model)) | (df.test_set_type == "test"))
        & (df.variant == variant)
    ]
    df_model = format_test_set_type(df_model)

    # Get unique datasets and descriptors
    datasets = df_model["dataset_type"].unique()
    descriptors = df_model["descriptor"].unique()

    # Create subplot grid similar to analysis.ipynb approach
    fig, ax = plt.subplots(
        nrows=len(datasets),
        ncols=len(descriptors),
        figsize=(4 * len(descriptors), 3 * len(datasets)),
    )

    for i, dataset in enumerate(datasets):
        for j, descriptor in enumerate(descriptors):
            current_ax = ax[i][j] if len(datasets) > 1 else ax[j]

            subset = df_model[
                (df_model["dataset_type"] == dataset)
                & (df_model["descriptor"] == descriptor)
            ]

            if len(subset) == 0:
                current_ax.set_visible(False)
                continue

            for test_type in subset["test_set_type"].unique():
                test_subset = subset[subset["test_set_type"] == test_type]
                test_subset = test_subset.sort_values(by="n_graphs")
                current_ax.plot(
                    test_subset["n_graphs"],
                    test_subset["mmd_results_mean"],
                    label=test_type,
                    marker="o",
                    markersize=4,
                    color=to_color(
                        test_type,
                        sorted(subset["test_set_type"].unique().tolist()),
                    ),
                )
                current_ax.fill_between(
                    test_subset["n_graphs"],
                    test_subset["mmd_results_low"],
                    test_subset["mmd_results_high"],
                    alpha=0.2,
                    color=to_color(
                        test_type,
                        sorted(subset["test_set_type"].unique().tolist()),
                    ),
                )

            current_ax.set_xscale("log")
            current_ax.xaxis.set_major_locator(MaxNLocator(7))

            x_min, x_max = current_ax.get_xlim()
            min_power = int(np.floor(np.log2(x_min)))
            max_power = int(np.ceil(np.log2(x_max)))
            power_of_2_ticks = [2**i for i in range(min_power, max_power + 1)]
            actual_ticks = [
                tick
                for tick in power_of_2_ticks
                if tick >= x_min and tick <= x_max
            ]

            if actual_ticks:
                current_ax.set_xticks(actual_ticks)
                current_ax.set_xticklabels(
                    [str(int(tick)) for tick in actual_ticks]
                )

            current_ax.set_title(
                f"{format_dataset(dataset)} - {format_descriptor(descriptor)}"
            )
            current_ax.set_xlabel("Number of Graphs")
            current_ax.set_ylabel("MMD Score")

            # if i == 0 and j == 0:
            current_ax.legend()

    plt.tight_layout()
    plt.savefig(
        f"experiments/subsampling/figures/{model}_{variant}_subsampling.pdf"
    )
    plt.close()


def main():
    # Use simple setup similar to analysis.ipynb
    setup_plotting_parameters()

    df = pd.read_csv(
        here() / "experiments" / "subsampling" / "results" / "subsampling.csv"
    )
    df = df.drop(columns=["generation_procedure"])

    # models = ["gran", "autograph", "digress"]
    models = ["digress", "gran"]
    variants = ["umve", "biased"]
    parameters = itertools.product(models, variants)
    for model, variant in parameters:
        plot_subsampling(df, model, variant)


if __name__ == "__main__":
    main()
