import itertools
import json

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


def to_color(element: str) -> str:
    """Map element to a consistent colorblind-friendly color."""
    # Define a canonical mapping for all possible elements to ensure color consistency
    color_map = {
        "Test set": sns.color_palette("colorblind")[0],
        "AutoGraph": sns.color_palette("colorblind")[1],
        "DiGress": sns.color_palette("colorblind")[4],
        "GRAN": sns.color_palette("colorblind")[3],
    }
    # Default to the first color if element is not in the map, though it shouldn't happen with current data
    return color_map.get(element, sns.color_palette("colorblind")[0])


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


def add_error_bands(g: sns.FacetGrid, df_model: pd.DataFrame) -> None:
    for (row_val, col_val), ax in g.axes_dict.items():
        subset = df_model[
            (df_model["Dataset"] == row_val)
            & (df_model["Descriptor"] == col_val)
        ]

        if len(subset) == 0:
            continue

        for test_type in subset["test_set_type"].unique():
            test_subset = subset[subset["test_set_type"] == test_type]
            test_subset = test_subset.sort_values(by="n_graphs")

            if len(test_subset) > 0:
                color = to_color(test_type)
                ax.fill_between(
                    test_subset["n_graphs"],
                    test_subset["mmd_results_low"],
                    test_subset["mmd_results_high"],
                    alpha=0.2,
                    color=color,
                )


def apply_log_scale(g: sns.FacetGrid) -> None:
    for ax in g.axes.flat:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(7))

        x_min, x_max = ax.get_xlim()
        min_power = int(np.floor(np.log2(x_min)))
        max_power = int(np.ceil(np.log2(x_max)))
        power_of_2_ticks = [2**i for i in range(min_power, max_power + 1)]
        actual_ticks = [
            tick for tick in power_of_2_ticks if tick >= x_min and tick <= x_max
        ]

        if actual_ticks:
            ax.set_xticks(actual_ticks)
            ax.set_xticklabels([str(int(tick)) for tick in actual_ticks])


def set_titles(g: sns.FacetGrid) -> None:
    for (row_val, col_val), ax in g.axes_dict.items():
        ax.set_title(f"{col_val} on {row_val}")


def plot_individual_subsampling(
    df: pd.DataFrame, model: str, variant: str
) -> None:
    """
    Create individual line plots for each dataset-descriptor combination.
    """
    df_model = df.loc[
        ((df.test_set_type.str.contains(model)) | (df.test_set_type == "test"))
        & (df.variant == variant)
    ]
    df_model = format_test_set_type(df_model)

    df_model["Dataset"] = df_model["dataset_type"].apply(format_dataset)
    df_model["Descriptor"] = df_model["descriptor"].apply(format_descriptor)

    combinations = df_model[["Dataset", "Descriptor"]].drop_duplicates()

    for _, row in combinations.iterrows():
        dataset = row["Dataset"]
        descriptor = row["Descriptor"]

        subset = df_model[
            (df_model["Dataset"] == dataset)
            & (df_model["Descriptor"] == descriptor)
        ]

        if len(subset) == 0:
            continue

        plt.figure(figsize=(4, 3))

        for test_type in subset["test_set_type"].unique():
            test_subset = subset[subset["test_set_type"] == test_type]
            test_subset = test_subset.sort_values(by="n_graphs")

            if len(test_subset) > 0:
                color = to_color(test_type)
                plt.plot(
                    test_subset["n_graphs"],
                    test_subset["mmd_results_mean"],
                    marker="o",
                    markersize=4,
                    color=color,
                    label=test_type,
                )

                plt.fill_between(
                    test_subset["n_graphs"],
                    test_subset["mmd_results_low"],
                    test_subset["mmd_results_high"],
                    alpha=0.2,
                    color=color,
                )

        plt.xscale("log")
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(7))

        x_min, x_max = ax.get_xlim()
        min_power = int(np.floor(np.log2(x_min)))
        max_power = int(np.ceil(np.log2(x_max)))
        power_of_2_ticks = [2**i for i in range(min_power, max_power + 1)]
        actual_ticks = [
            tick for tick in power_of_2_ticks if tick >= x_min and tick <= x_max
        ]

        if actual_ticks:
            ax.set_xticks(actual_ticks)
            ax.set_xticklabels([str(int(tick)) for tick in actual_ticks])

        plt.xlabel("Number of Graphs")
        plt.ylabel("MMD Score")
        plt.title(f"{descriptor} on {dataset}")
        plt.legend(
            title="Graph sources", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.tight_layout()

        dest_dir = (
            here() / "experiments" / "subsampling" / "figures" / "individual"
        )
        dest_dir.mkdir(parents=True, exist_ok=True)
        output_path = f"{dest_dir}/{model}_{variant}_{dataset.lower()}_{descriptor.lower().replace(' ', '_')}_subsampling.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


def plot_subsampling(df: pd.DataFrame, model: str, variant: str) -> None:
    """
    Create a multi-panel line plot for df restricted to a single model using seaborn relplot.

    Uses seaborn relplot for cleaner faceting and styling.
    """
    df_model = df.loc[
        ((df.test_set_type.str.contains(model)) | (df.test_set_type == "test"))
        & (df.variant == variant)
    ]
    df_model = format_test_set_type(df_model)

    df_model["Dataset"] = df_model["dataset_type"].apply(format_dataset)
    df_model["Descriptor"] = df_model["descriptor"].apply(format_descriptor)

    unique_test_types = df_model["test_set_type"].unique()
    color_palette = {
        test_type: to_color(test_type) for test_type in unique_test_types
    }
    # Create relplot with faceting
    g = sns.relplot(
        data=df_model,
        x="n_graphs",
        y="mmd_results_mean",
        hue="test_set_type",
        col="Descriptor",
        row="Dataset",
        kind="line",
        marker="o",
        markersize=4,
        # facet_kws={"margin_titles": True},
        height=3,
        aspect=1.1,
        facet_kws={"sharey": False},
        palette=color_palette,
    )
    sns.move_legend(
        g, "upper right", bbox_to_anchor=(1.0, 0.5), title="Graph sources"
    )
    add_error_bands(g, df_model)
    apply_log_scale(g)
    set_titles(g)

    g.set_axis_labels("Number of Graphs", "MMD Score")

    output_path = (
        f"experiments/subsampling/figures/{model}_{variant}_subsampling.pdf"
    )
    g.fig.savefig(
        output_path,
    )


def main():
    # Use simple setup similar to analysis.ipynb
    setup_plotting_parameters()

    df = pd.read_csv(
        here() / "experiments" / "subsampling" / "results" / "subsampling.csv"
    )
    df = df.drop(columns=["generation_procedure"])

    models = ["gran", "autograph", "digress"]
    # models = ["digress"]
    variants = ["umve", "biased"]
    # variants = ["umve"]
    parameters = itertools.product(models, variants)
    for model, variant in parameters:
        plot_subsampling(df, model, variant)  # Combined plot
        plot_individual_subsampling(df, model, variant)  # Individual plots

if __name__ == "__main__":
    main()
