import torch
import torch_geometric
from torch_geometric.utils import to_undirected, coalesce
from polygraph.datasets.base import OnlineGraphDataset
import networkx as nx
from collections import Counter
from typing import Tuple, Optional
import matplotlib.pyplot as plt

NUCLEOTIDE_TO_INDEX = {
    "5'": 0,
    "3'": 1,
    "A": 2,
    "C": 3,
    "G": 4,
    "U": 5,
}

INDEX_TO_NUCLEOTIDE = {v: k for k, v in NUCLEOTIDE_TO_INDEX.items()}

ALLOWED_PAIRINGS = {
    "A": {"U"},
    "U": {"A", "G"},
    "G": {"C", "U"},
    "C": {"G"},
}


def dotbracket_to_graph(rna: str, dotbracket: str) -> torch_geometric.data.Data:
    rna = rna.upper()
    nucleotides = [NUCLEOTIDE_TO_INDEX[n] for n in ("5'", *rna, "3'")]
    backbone_index = torch.stack(
        [
            torch.arange(0, len(nucleotides) - 1),
            torch.arange(1, len(nucleotides)),
        ],
        dim=0,
    )
    base_pairs = [[], []]
    stack = []
    for i, n in enumerate(dotbracket):
        if n == "(":
            stack.append(i)
        elif n == ")":
            base_pairs[0].append(stack.pop() + 1)
            base_pairs[1].append(i + 1)

    base_pairs = torch.tensor(base_pairs, dtype=torch.long)
    edge_index = torch.cat([backbone_index, base_pairs], dim=1)
    edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
    edge_type[backbone_index.shape[1] :] = 1
    edge_index, edge_type = to_undirected(edge_index, edge_type)
    edge_index, edge_type = coalesce(
        edge_index, edge_type, num_nodes=len(nucleotides)
    )

    data = torch_geometric.data.Data(
        x=torch.tensor(nucleotides, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_type,
    )
    return data


def graph_to_dotbracket(graph: nx.Graph) -> Tuple[Optional[str], Optional[str]]:
    """Convert an RNA secondary structure graph to a dotbracket string.

    Args:
        graph: The RNA secondary structure graph.

    Returns:
        Tuple[Optional[str], Optional[str]]: The RNA sequence and dotbracket string. If the graph is not a valid RNA secondary structure, returns `None, None`.
    """
    # Verify that there is one 5' and one 3' end
    node_type_counts = Counter(
        graph.nodes[n]["nucleotides"] for n in graph.nodes
    )
    node_type_counts = {
        INDEX_TO_NUCLEOTIDE[k]: v for k, v in node_type_counts.items()
    }

    if node_type_counts.get("5'", 0) != 1 or node_type_counts.get("3'", 0) != 1:
        return None, None

    start_node = next(
        n
        for n in graph.nodes
        if INDEX_TO_NUCLEOTIDE[graph.nodes[n]["nucleotides"]] == "5'"
    )
    end_node = next(
        n
        for n in graph.nodes
        if INDEX_TO_NUCLEOTIDE[graph.nodes[n]["nucleotides"]] == "3'"
    )

    if graph.degree(start_node) != 1 or graph.degree(end_node) != 1:
        return None, None

    # Verify that there is a unique 5' 3' path
    backbone_edges = [
        e for e in graph.edges if graph.edges[e]["bond_types"] == 0
    ]
    backbone_graph = graph.edge_subgraph(backbone_edges)

    # Verify that every node is connected to the backbone
    if len(backbone_graph.nodes) != len(graph.nodes):
        return None, None

    # Verify that the backbone is a tree
    if (
        not nx.is_connected(backbone_graph)
        or len(backbone_edges) != len(graph.nodes) - 1
    ):
        return None, None

    # Find the 5' 3' path
    path = nx.shortest_path(backbone_graph, source=start_node, target=end_node)
    if len(path) != len(graph.nodes):
        return None, None

    rna = "".join(
        [INDEX_TO_NUCLEOTIDE[graph.nodes[n]["nucleotides"]] for n in path[1:-1]]
    )

    dotbracket = []
    stack = []
    node_to_position = {n: i for i, n in enumerate(path)}

    for n in path[1:-1]:
        incident = list(graph.edges(n, data=True))
        bond_type_counts = Counter(
            [info["bond_types"] for u, v, info in incident]
        )
        assert bond_type_counts[0] == 2
        if bond_type_counts[1] > 1:
            return None, None
        elif bond_type_counts[1] == 1:
            pairing_edge = next(e for e in incident if e[2]["bond_types"] == 1)
            paired_node = pairing_edge[1]
            assert paired_node != n

            if abs(node_to_position[paired_node] - node_to_position[n]) < 4:
                # Loop is too tight
                return None, None

            if (
                rna[node_to_position[n] - 1]
                not in ALLOWED_PAIRINGS[rna[node_to_position[paired_node] - 1]]
            ):
                return None, None

            assert (
                rna[node_to_position[paired_node] - 1]
                in ALLOWED_PAIRINGS[rna[node_to_position[n] - 1]]
            )

            if node_to_position[paired_node] > node_to_position[n]:
                stack.append((n, paired_node))
                dotbracket.append("(")
            else:
                bond = stack.pop()
                if bond[0] != paired_node or bond[1] != n:
                    # Pseudoknot
                    return None, None
                dotbracket.append(")")
        else:
            dotbracket.append(".")

    if len(stack) != 0:
        return None, None

    dotbracket = "".join(dotbracket)
    return rna, dotbracket


class RFAMGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "test": "https://datashare.biochem.mpg.de/s/8H6miRAxfkz7Dyz/download",
        "train": "https://datashare.biochem.mpg.de/s/kQplGPSvbY7zmmy/download",
        "val": "https://datashare.biochem.mpg.de/s/Ve8NV5vmWyhzoqn/download",
    }

    _HASH_FOR_SPLIT = {
        "test": None,
        "train": None,
        "val": None,
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    def hash_for_split(self, split: str) -> str:
        return self._HASH_FOR_SPLIT[split]

    @staticmethod
    def is_valid(graph: nx.Graph) -> bool:
        rna, dotbracket = graph_to_dotbracket(graph)
        return rna is not None and dotbracket is not None

    @staticmethod
    def stability(graph: nx.Graph) -> float:
        import RNA

        rna, dotbracket = graph_to_dotbracket(graph)
        if rna is None or dotbracket is None:
            return None
        optimal_structure, optimal_energy = RNA.fold(rna)
        realized_energy = RNA.energy_of_struct(rna, dotbracket)
        assert optimal_energy == RNA.energy_of_struct(rna, optimal_structure)
        if optimal_energy == 0:
            return None

        stability = realized_energy / optimal_energy
        assert stability <= 1.0
        return stability

    @staticmethod
    def plot_structure(
        graph: nx.Graph, ax: Optional[plt.Axes] = None, **kwargs
    ):
        import forgi.graph.bulge_graph as fgb
        import forgi.visual.mplotlib as fvm

        sequence, structure = graph_to_dotbracket(graph)
        if sequence is None or structure is None:
            return None

        bg = fgb.BulgeGraph.from_dotbracket(structure, seq=sequence)

        # Create a new figure and axis if none are provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the RNA structure
        fvm.plot_rna(bg, ax=ax, **kwargs)
        return ax

    @staticmethod
    def to_sequence_and_structure(
        graph: nx.Graph,
    ) -> Tuple[Optional[str], Optional[str]]:
        rna, dotbracket = graph_to_dotbracket(graph)
        return rna, dotbracket


if __name__ == "__main__":
    ds = RFAMGraphDataset("train").to_nx()
    assert all(RFAMGraphDataset.is_valid(g) for g in ds)
