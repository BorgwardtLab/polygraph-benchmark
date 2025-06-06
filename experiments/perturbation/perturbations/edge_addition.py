import networkx as nx
import numpy as np

from .base import BasePerturbation


class EdgeAdditionPerturbation(BasePerturbation):
    """Perturbation that randomly adds edges between unconnected nodes.

    The perturbation adds edges with a probability calculated to achieve
    the desired noise level (ratio of new edges to original edges).
    """

    def perturb(self, graph: nx.Graph, noise_level: float) -> nx.Graph:
        """Add edges to the graph based on the noise level.

        Args:
            graph: The input graph to perturb
            noise_level: Target ratio of new edges to original edges (0.0 to 1.0)

        Returns:
            A new graph with added edges
        """
        if not (0 <= noise_level <= 1):
            raise ValueError("Noise level must be between 0 and 1")

        # Create a copy of the graph to modify
        perturbed = graph.copy()

        # Calculate probability needed to achieve desired noise level in expectation
        num_edges = perturbed.number_of_edges()
        num_nodes = perturbed.number_of_nodes()
        max_possible_edges = (num_nodes * (num_nodes - 1)) // 2
        remaining_possible_edges = max_possible_edges - num_edges

        if remaining_possible_edges == 0:
            return perturbed

        # Probability to add each possible edge to achieve noise_level * num_edges new edges
        target_new_edges = noise_level * num_edges
        probability = min(1.0, target_new_edges / remaining_possible_edges)

        # Consider all possible node pairs
        nodes = list(perturbed.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Skip if edge already exists
                if perturbed.has_edge(nodes[i], nodes[j]):
                    continue

                # Add edge with calculated probability
                if np.random.random() < probability:
                    perturbed.add_edge(nodes[i], nodes[j])

        return perturbed
