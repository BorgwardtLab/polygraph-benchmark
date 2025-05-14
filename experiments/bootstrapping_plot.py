from typing import Tuple

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from pyprojroot import here

from polygraph.datasets import (
    LobsterGraphDataset,
    PlanarGraphDataset,
    SBMGraphDataset,
)


def setup_plotting_parameters(
    resolution: int = 600, size: Tuple[float, float] = (6.8, 5)
) -> None:
    plt.rcParams["figure.figsize"] = size
    plt.rcParams["savefig.dpi"] = resolution

    cmfont = font_manager.FontProperties(
        fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf"
    )
    font_name = cmfont.get_name()

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [font_name]
    mpl.rcParams["font.monospace"] = [font_name]
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["axes.unicode_minus"] = False


def format_dataset(dataset: str) -> str:
    if "lobster" in dataset:
        return "Lobster"
    elif "sbm" in dataset:
        return "SBM"
    elif "planar" in dataset:
        return "Planar"


def format_descriptor(descriptor: str) -> str:
    """Format a descriptor string by splitting before the first capitalized character in camelCase.

    Args:
        descriptor: The descriptor string to format

    Returns:
        The formatted descriptor string
    """
    if not descriptor or len(descriptor) <= 1:
        return descriptor

    for i in range(1, len(descriptor)):
        if descriptor[i].isupper():
            return descriptor[:i] + " " + descriptor[i:]

    return descriptor


def add_training_sample_size(df: pd.DataFrame) -> pd.DataFrame:
    n_train_lobster_fixed = len(LobsterGraphDataset(split="train"))
    n_train_sbm_fixed = len(SBMGraphDataset(split="train"))
    n_train_planar_fixed = len(PlanarGraphDataset(split="train"))
    n_val_lobster_fixed = len(LobsterGraphDataset(split="val"))
    n_val_sbm_fixed = len(SBMGraphDataset(split="val"))
    n_val_planar_fixed = len(PlanarGraphDataset(split="val"))
    n_test_lobster_fixed = len(LobsterGraphDataset(split="test"))
    n_test_sbm_fixed = len(SBMGraphDataset(split="test"))
    n_test_planar_fixed = len(PlanarGraphDataset(split="test"))
    n_train_procedural = 8192

    # gran_val_procedural = 256
    # gran_test_procedural = 256
    # gran_train_procedural = 8192

    # digress_val_procedural = 1024
    # digress_test_procedural = 1024
    # digress_train_procedural = 8192

    # autograph_val_procedural = 1024
    # autograph_test_procedural = 1024
    # autograph_train_procedural = 8192

    logger.info("Adding training sample size to dataframe.")
    # ! We're adding the test and val sample sizes from the fixed datasets
    for test_set_type in df.test_set_type.unique():
        for dataset_type in df.dataset_type.unique():
            if "procedural" in test_set_type and "sbm" in dataset_type:
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_train",
                ] = n_train_procedural
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_val",
                ] = n_val_sbm_fixed
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_test",
                ] = n_test_sbm_fixed
            elif "procedural" in test_set_type and "planar" in dataset_type:
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_train",
                ] = n_train_procedural
                df.loc[df.test_set_type == test_set_type, "n_val"] = (
                    n_val_planar_fixed
                )
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_test",
                ] = n_test_planar_fixed
            elif "procedural" in test_set_type and "lobster" in dataset_type:
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_train",
                ] = n_train_procedural
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_val",
                ] = n_val_lobster_fixed
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_test",
                ] = n_test_lobster_fixed

            if "fixed" in test_set_type and "sbm" in dataset_type:
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_train",
                ] = n_train_sbm_fixed
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_val",
                ] = n_val_sbm_fixed
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_test",
                ] = n_test_sbm_fixed
            elif "fixed" in test_set_type and "planar" in dataset_type:
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_train",
                ] = n_train_planar_fixed
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_val",
                ] = n_val_planar_fixed
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_test",
                ] = n_test_planar_fixed
            elif "fixed" in test_set_type and "lobster" in dataset_type:
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_train",
                ] = n_train_lobster_fixed
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_val",
                ] = n_val_lobster_fixed
                df.loc[
                    (df.test_set_type == test_set_type)
                    & (df.dataset_type == dataset_type),
                    "n_test",
                ] = n_test_lobster_fixed
    return df


def format_generated_set_name(generated_set_type: str, n_train: int) -> str:
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


def plot_individual_tests(df):
    plt.tight_layout()
    main_test_color = "#EA5C1F"
    pastel_test_color = "#F4A07C"

    main_generated_color = "#247BA0"
    pastel_generated_color = "#8BCAE5"

    for split in ["procedural"]:
        for descriptor in df.descriptor.unique():
            for dataset in df.dataset_type.unique():
                for variant in df.variant.unique():
                    for generated_set_type in [
                        "gran_generated_procedural",
                        "gran_generated_fixed",
                        "autograph_generated_fixed",
                        "digress_generated_fixed",
                    ]:
                        df_instance = df[
                            (df.generation_procedure == split)
                            & (df.dataset_type == dataset)
                            & (df.descriptor == descriptor)
                            & (df.variant == variant)
                        ]

                        df_test = df_instance[
                            df_instance.test_set_type == "test"
                        ]

                        plt.figure()

                        ax = sns.lineplot(
                            data=df_test,
                            x="n_graphs",
                            y="mmd_results_mean",
                            label="MMD Training vs. Test Set",
                            color=main_test_color,  # Use main test color for the line
                        )
                        ax.fill_between(
                            df_test.n_graphs,
                            df_test.mmd_results_low,
                            df_test.mmd_results_high,
                            alpha=0.5,  # Adjust alpha as needed for pastel effect
                            color=pastel_test_color,  # Use pastel test color for the fill
                        )

                        df_generated = df_instance[
                            (df_instance.test_set_type == generated_set_type)
                            & (df_instance.n_graphs <= 1024)
                        ]
                        if df_generated.mmd_results_mean.isna().all():
                            plt.close()
                            continue

                        n_fixed_train = df_generated.n_train.dropna().unique()
                        assert len(n_fixed_train) == 1, (
                            f"n_fixed_train: {n_fixed_train}"
                        )
                        ax = sns.lineplot(
                            data=df_generated,
                            x="n_graphs",
                            y="mmd_results_mean",
                            label=f"MMD Test vs. {format_generated_set_name(generated_set_type, n_fixed_train[0])}",
                            color=main_generated_color,  # Use main generated color for the line
                        )
                        ax.fill_between(
                            df_generated.n_graphs,
                            df_generated.mmd_results_low,
                            df_generated.mmd_results_high,
                            alpha=0.5,  # Adjust alpha as needed for pastel effect
                            color=pastel_generated_color,  # Use pastel generated color for the fill
                        )

                        plt.xlabel("Number of Bootstrapped Graphs", labelpad=20)
                        plt.ylabel(f"Realized MMD ({format_variant(variant)})")
                        plt.title(
                            f"Dataset: {format_dataset(dataset)}, Descriptor: {format_descriptor(descriptor)}"
                        )
                        plt.xscale("log", base=2)
                        n_train = (
                            df_generated.n_train.dropna().iloc[0]
                            if not df_generated.n_train.dropna().empty
                            else None
                        )
                        n_val = (
                            df_generated.n_val.dropna().iloc[0]
                            if not df_generated.n_val.dropna().empty
                            else None
                        )
                        n_test = (
                            df_generated.n_test.dropna().iloc[0]
                            if not df_generated.n_test.dropna().empty
                            else None
                        )

                        current_tick_locs, current_tick_labels_objs = (
                            plt.xticks()
                        )
                        current_tick_labels = [
                            label.get_text()
                            for label in current_tick_labels_objs
                        ]

                        plt.xticks(current_tick_locs, current_tick_labels)

                        if n_val != n_test:
                            # Define your new ticks and labels
                            new_tick_values = [n_train, n_val, n_test]
                            new_tick_custom_labels = [
                                f"Train ({int(n_train)})",
                                f"Val ({int(n_val)})",
                                f"Test ({int(n_test)})",
                            ]
                        else:
                            new_tick_values = [n_train, n_test]
                            new_tick_custom_labels = [
                                f"Train ({int(n_train)})",
                                f"Test/Val ({int(n_test)})",
                            ]
                        y_min, y_max = ax.get_ylim()
                        text_offset = (y_max - y_min) * 0.05

                        for tick_val, tick_label in zip(
                            new_tick_values, new_tick_custom_labels
                        ):
                            ax.axvline(
                                x=tick_val,
                                color="gray",
                                linestyle="--",
                                linewidth=1,
                                ymax=0.05,
                            )
                            ax.text(
                                tick_val,
                                y_min - text_offset,
                                tick_label,
                                rotation=45,
                                weight="bold",
                                ha="right",
                                va="top",
                            )
                        plt.subplots_adjust(bottom=0.2)

                        plt.legend(frameon=False)
                        sns.despine()
                        plt.tight_layout()
                        min_x_lim = min([32, n_train, n_val, n_test])
                        if min_x_lim <= 32:
                            # to make sure we can see the ticks
                            min_x_lim = min_x_lim - min_x_lim * 0.1
                        plt.xlim(min_x_lim, 2048)
                        plt.savefig(
                            here()
                            / "experiments"
                            / "figures"
                            / f"bootstrapping_{variant}_{descriptor}_{dataset}_{split}_{generated_set_type}.pdf"
                        )
                        plt.close()


def main():
    sns.set_style("white")
    sns.set_palette("bright")
    setup_plotting_parameters()
    df = pd.read_csv(here() / "experiments" / "results" / "bootstrapping.csv")
    df = add_training_sample_size(df)
    df = df.sort_values("n_graphs")  # Sort the dataframe by n_graphs
    plot_individual_tests(df)


if __name__ == "__main__":
    main()
