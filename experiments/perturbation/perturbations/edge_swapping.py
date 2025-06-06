import networkx as nx
import numpy as np
import random

from .base import BasePerturbation


class EdgeSwappingPerturbation(BasePerturbation):
    """Perturbation that swaps edges while preserving node degrees.

    The perturbation randomly selects pairs of edges and swaps their endpoints
    while maintaining the degree of each node and avoiding parallel edges.
    """

    def perturb(self, graph: nx.Graph, noise_level: float) -> nx.Graph:
        """Swap edges in the graph based on the noise level.

        Args:
            graph: The input graph to perturb
            noise_level: Probability of selecting each edge for swapping (0.0 to 1.0)

        Returns:
            A new graph with swapped edges
        """
        if not (0 <= noise_level <= 1):
            raise ValueError("Noise level must be between 0 and 1")

        # Create a copy of the graph to modify
        perturbed = graph.copy()

        # Store original degrees for verification
        original_degrees = dict(perturbed.degree())

        # Get list of edges and randomly select some based on probability
        edges = list(perturbed.edges())
        selected = [
            i for i in range(len(edges)) if np.random.random() < noise_level
        ]

        # Shuffle the selected edges before pairing
        random.shuffle(selected)

        # Pair up selected edges, ignoring last one if odd number
        num_pairs = len(selected) // 2
        for i in range(num_pairs):
            edge1_idx = selected[2 * i]
            edge2_idx = selected[2 * i + 1]

            # Get the nodes for each edge
            a, b = edges[edge1_idx]
            c, d = edges[edge2_idx]

            # Skip if edges share any nodes
            if len(set([a, b, c, d])) < 4:
                continue

            # Check which swap options are valid (don't create parallel edges)
            option1_valid = not perturbed.has_edge(
                a, d
            ) and not perturbed.has_edge(c, b)
            option2_valid = not perturbed.has_edge(
                a, c
            ) and not perturbed.has_edge(b, d)

            # Skip if both options would create parallel edges
            if not option1_valid and not option2_valid:
                continue

            # Remove original edges
            perturbed.remove_edge(a, b)
            perturbed.remove_edge(c, d)

            # Randomly choose between valid configurations
            if option1_valid and option2_valid:
                if random.random() < 0.5:
                    perturbed.add_edge(a, d)
                    perturbed.add_edge(c, b)
                else:
                    perturbed.add_edge(a, c)
                    perturbed.add_edge(b, d)
            elif option1_valid:
                perturbed.add_edge(a, d)
                perturbed.add_edge(c, b)
            else:  # option2_valid
                perturbed.add_edge(a, c)
                perturbed.add_edge(b, d)

        # Verify that node degrees haven't changed
        final_degrees = dict(perturbed.degree())
        for node in perturbed.nodes():
            assert original_degrees[node] == final_degrees[node], (
                f"Degree changed for node {node}: {original_degrees[node]} -> {final_degrees[node]}"
            )

        return perturbed
