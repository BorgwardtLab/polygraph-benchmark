"""
This script converts the digress samples to networkx graphs.
"""

import os
import pickle

import networkx as nx
from loguru import logger
from tqdm import tqdm


def convert_single_graph(data: dict) -> nx.Graph:
    G = nx.Graph()

    n_nodes = len(data[1])

    G.add_nodes_from(range(n_nodes))

    import numpy as np

    adj_matrix = np.array(data[1])
    edges = np.where(adj_matrix == 1)

    edge_list = [(i, j) for i, j in zip(*edges)]
    G.add_edges_from(edge_list)

    return G


def walk_directory(directory_path: str) -> None:
    """Walk through the directory and print its contents.

    Args:
        directory_path: Path to the directory to walk through
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    nx_graphs = []
                    for graphs in tqdm(
                        data, leave=False, desc="Converting graphs"
                    ):
                        G = convert_single_graph(graphs)
                        nx_graphs.append(G)

                    output_path = file_path.replace(".pkl", ".nx.pkl").replace(
                        "/fs/pool/pool-mlsb/polygraph/digress-samples",
                        "/fs/pool/pool-hartout/Documents/Git/polygraph/data/digress/converted",
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "wb") as f:
                        pickle.dump(nx_graphs, f)

                    logger.info(f"Converted {file_path} to {output_path}")


def main() -> None:
    """Main function to walk through the digress samples directory."""
    directory = "/fs/pool/pool-mlsb/polygraph/digress-samples"
    walk_directory(directory)


if __name__ == "__main__":
    main()
