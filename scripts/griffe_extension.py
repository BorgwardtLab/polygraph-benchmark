import griffe
from jinja2 import Template
from typing import Iterable, Dict, List
from tabulate import tabulate
from polygraph.datasets import *  # noqa
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = griffe.get_logger(__name__)


class JinjaDocstringExtension(griffe.Extension):
    def __init__(self, **kwargs):
        super().__init__()
        # Store context variables that will be available in templates
        self.context = {}

        # Add utility functions to the context
        self.context["summary_md_table"] = summary_md_table
        self.context["plot_first_k_graphs"] = plot_first_k_graphs

        # Add any additional kwargs to the context
        self.context.update(kwargs)

    def on_instance(self, obj: griffe.Object, **kwargs) -> None:
        """Process docstrings as Jinja templates when an object instance is created."""
        if obj.docstring and obj.docstring.value:
            try:
                original_docstring = obj.docstring.value

                # Create a Jinja template from the docstring
                template = Template(original_docstring)

                # Render the template with our context
                rendered_docstring = template.render(**self.context)

                # Update the docstring value
                obj.docstring.value = rendered_docstring

            except Exception:
                logger.warning(
                    f"Failed to render Jinja template in docstring for {obj.path}:\n{traceback.format_exc()}"
                )


def summary_md_table(ds_name: str, splits: Iterable[str], precision: int = 2):
    """Generate a markdown table summarizing dataset statistics with splits as columns.

    Args:
        ds_name: Name of the dataset class to use
        splits: Iterable of split names (e.g., ["train", "val", "test"])
        precision: Number of decimal places to display for floating point values

    Returns:
        String containing a markdown table comparing all splits
    """

    # Define the metrics we want to display
    metrics = [
        "# of Graphs",
        "Min # of Nodes",
        "Max # of Nodes",
        "Avg # of Nodes",
        "Min # of Edges",
        "Max # of Edges",
        "Avg # of Edges",
        "Edge/Node Ratio",
        "Is Undirected",
    ]

    # Collect data for each split
    split_data: Dict[str, List[str]] = {}

    for split in splits:
        ds = globals()[ds_name](split)
        split_data[split] = [
            str(len(ds)),
            str(ds.min_nodes),
            str(ds.max_nodes),
            f"{ds.avg_nodes:.{precision}f}",
            str(ds.min_edges),
            str(ds.max_edges),
            f"{ds.avg_edges:.{precision}f}",
            f"{ds.edge_node_ratio:.{precision}f}",
            str(ds.is_undirected),
        ]

    # Create table data with metrics as rows and splits as columns
    table_data = []
    for i, metric in enumerate(metrics):
        row = [metric]
        for split in splits:
            row.append(split_data[split][i])
        table_data.append(row)

    # Generate headers with split names
    headers = ["Metric"] + [split.capitalize() for split in splits]

    # Generate markdown table using tabulate with disable_numparse=True
    md_table = tabulate(
        table_data, headers=headers, tablefmt="pipe", disable_numparse=True
    )

    print(md_table)

    return md_table + "\n\n"


def plot_first_k_graphs(
    ds_name: str, split: str = "train", k: int = 4, node_size: int = 100
):
    """Plot the first k graphs from a dataset and return markdown image links.

    Args:
        ds_name: Name of the dataset class
        split: Dataset split to use
        k: Number of graphs to plot

    Returns:
        String with markdown image links
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from pathlib import Path

    # Get dataset
    ds = globals()[ds_name](split).to_nx()
    k = min(k, len(ds))

    # Create plots directory
    Path("docs/images").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, k, figsize=(3.3 * k, 3.3))

    # Plot each graph
    for i in range(k):
        graph_data = ds[i]

        nx.draw(graph_data, ax=axes[i], node_size=node_size, with_labels=False)

    plt.tight_layout()

    # Save plot
    filename = f"{ds_name}_{split}_first_{k}.png"
    plt.savefig(f"docs/images/{filename}", dpi=150, bbox_inches="tight")
    plt.close()

    return f"![First {k} graphs](/images/{filename})\n\n"
