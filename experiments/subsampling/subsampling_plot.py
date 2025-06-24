import json
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from pyprojroot import here
from tueplots import axes, bundles

from polygraph.datasets import (
    LobsterGraphDataset,
    PlanarGraphDataset,
    SBMGraphDataset,
)


def load_global_rcparams() -> dict:
    """Load global rcparams from JSON file."""
    with open("/fs/pool/pool-mlsb/polygraph/rcparams.json", "r") as f:
        return json.load(f)


def setup_plotting_parameters(
    venue: str = "icml2022",
    use_global_rcparams: bool = True,
    font_source: str = "tueplots",
    resolution: int = 600,
    size: Optional[Tuple[float, float]] = None,
) -> None:
    """Setup plotting parameters using tueplots with optional global rcparams override.

    Args:
        venue: The publication venue (icml2022, neurips2023, etc.)
        use_global_rcparams: Whether to use global rcparams
        font_source: Font source priority - "tueplots", "rcparams", or "hybrid"
        resolution: DPI for saving figures
        size: Custom figure size override
    """

    if venue == "icml2022":
        rcparams = bundles.icml2022()
    elif venue == "neurips2023":
        rcparams = bundles.neurips2023()
    elif venue == "iclr2024":
        rcparams = bundles.iclr2024()
    else:
        rcparams = bundles.icml2022()

    if use_global_rcparams:
        global_rcparams = load_global_rcparams()

        if font_source == "tueplots":
            font_keys = {
                "font.family",
                "font.serif",
                "font.sans-serif",
                "font.monospace",
                "text.usetex",
                "text.latex.preamble",
                "mathtext.fontset",
                "mathtext.rm",
                "mathtext.it",
                "mathtext.bf",
                "mathtext.sf",
                "mathtext.tt",
            }
            non_font_params = {
                k: v for k, v in global_rcparams.items() if k not in font_keys
            }
            rcparams.update(non_font_params)

        elif font_source == "rcparams":
            font_keys = {
                "font.family",
                "font.serif",
                "font.sans-serif",
                "font.monospace",
                "text.usetex",
                "text.latex.preamble",
                "mathtext.fontset",
                "mathtext.rm",
                "mathtext.it",
                "mathtext.bf",
                "mathtext.sf",
                "mathtext.tt",
            }

            rcparams.update(global_rcparams)

            if rcparams.get("font.family") == "DejaVu Sans":
                rcparams.update(
                    {
                        "font.family": "sans-serif",
                        "font.sans-serif": [
                            "DejaVu Sans",
                            "Arial",
                            "Helvetica",
                        ],
                        "mathtext.fontset": "dejavusans",
                        "text.usetex": False,
                    }
                )

        elif font_source == "hybrid":
            rcparams.update(global_rcparams)

            if (
                rcparams.get("font.family") in ["DejaVu Sans"]
                and "font.serif" in rcparams
            ):
                rcparams.update(
                    {
                        "font.family": "serif",
                        "font.serif": [
                            "Times",
                            "Times New Roman",
                            "DejaVu Serif",
                        ],
                        "mathtext.fontset": "stix",
                        "mathtext.rm": "Times",
                        "mathtext.it": "Times:italic",
                        "mathtext.bf": "Times:bold",
                    }
                )

    if size is not None:
        rcparams["figure.figsize"] = size

    rcparams["savefig.dpi"] = resolution

    plt.rcParams.update(rcparams)

    logger.info(
        f"Applied font settings - Family: {rcparams.get('font.family')}, "
        f"UseTeX: {rcparams.get('text.usetex')}, "
        f"MathFont: {rcparams.get('mathtext.fontset')}"
    )


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

    logger.info("Adding training sample size to dataframe.")
    for test_set_type in df.test_set_type.unique():
        for dataset_type in df.dataset_type.unique():
            if "procedural" in test_set_type:
                if "sbm" in dataset_type:
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
                elif "planar" in dataset_type:
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
                elif "lobster" in dataset_type:
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
            elif "fixed" in test_set_type:
                if "sbm" in dataset_type:
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
                elif "planar" in dataset_type:
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
                elif "lobster" in dataset_type:
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


def plot_individual_tests(df):
    with plt.rc_context(axes.lines()):
        plt.tight_layout()

        colors = sns.color_palette("colorblind", n_colors=2)

        main_test_color = colors[0]
        main_generated_color = colors[1]

        for split in ["procedural"]:
            for descriptor in df.descriptor.unique():
                for dataset in df.dataset_type.unique():
                    for variant in df.variant.unique():
                        for generated_set_type in [
                            "gran_generated_procedural",
                            "digress_generated_procedural",
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
                                color=main_test_color,
                            )
                            ax.fill_between(
                                df_test.n_graphs,
                                df_test.mmd_results_low,
                                df_test.mmd_results_high,
                                alpha=0.3,
                                color=main_test_color,
                            )

                            df_generated = df_instance[
                                (
                                    df_instance.test_set_type
                                    == generated_set_type
                                )
                                & (df_instance.n_graphs <= 1024)
                            ]
                            if df_generated.mmd_results_mean.isna().all():
                                plt.close()
                                continue

                            n_fixed_train = (
                                df_generated.n_train.dropna().unique()
                            )
                            assert len(n_fixed_train) == 1, (
                                f"n_fixed_train: {n_fixed_train}"
                            )
                            ax = sns.lineplot(
                                data=df_generated,
                                x="n_graphs",
                                y="mmd_results_mean",
                                label=f"MMD Test vs. {format_generated_set_name(generated_set_type, n_fixed_train[0])}",
                                color=main_generated_color,
                            )
                            ax.fill_between(
                                df_generated.n_graphs,
                                df_generated.mmd_results_low,
                                df_generated.mmd_results_high,
                                alpha=0.3,
                                color=main_generated_color,
                            )

                            plt.xlabel(
                                "Number of Bootstrapped Graphs", labelpad=20
                            )
                            plt.ylabel(
                                f"Realized MMD ({format_variant(variant)})"
                            )
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
                            text_offset = (y_max - y_min) * 0.06

                            min_x_lim = min([32, n_train, n_val, n_test])
                            if min_x_lim <= 32:
                                min_x_lim = min_x_lim - min_x_lim * 0.1
                            max_x_lim = 2048

                            for tick_val, tick_label in zip(
                                new_tick_values, new_tick_custom_labels
                            ):
                                if min_x_lim <= tick_val <= max_x_lim:
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
                            plt.xlim(min_x_lim, max_x_lim)
                            plt.savefig(
                                here()
                                / "experiments"
                                / "figures"
                                / f"bootstrapping_{variant}_{descriptor}_{dataset}_{split}_{generated_set_type}.pdf"
                            )
                            plt.close()


def main():
    sns.set_style("white")
    sns.set_palette("colorblind")

    setup_plotting_parameters(
        venue="icml2022",
        use_global_rcparams=True,
        font_source="tueplots",
    )

    df = pd.read_csv(here() / "experiments" / "results" / "bootstrapping.csv")
    df = add_training_sample_size(df)
    df = df.sort_values("n_graphs")
    plot_individual_tests(df)


if __name__ == "__main__":
    main()
