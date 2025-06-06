import networkx as nx
import numpy as np

from .base import BasePerturbation


class EdgeDeletionPerturbation(BasePerturbation):
    """Perturbation that randomly deletes edges from graphs.

    The perturbation deletes edges with a probability calculated to achieve
    the desired noise level (ratio of deleted edges to original edges).
    """

    def perturb(self, graph: nx.Graph, noise_level: float) -> nx.Graph:
        """Delete edges from the graph based on the noise level.

        Args:
            graph: The input graph to perturb
            noise_level: Target ratio of edges to delete (0.0 to 1.0)

        Returns:
            A new graph with edges deleted
        """
        if not (0 <= noise_level <= 1):
            raise ValueError("Noise level must be between 0 and 1")

        # Create a copy of the graph to modify
        perturbed = graph.copy()

        # Get list of edges
        edges = list(perturbed.edges())
        num_edges = len(edges)

        if num_edges <= 1:
            return perturbed  # Don't delete if only 1 or 0 edges

        # Randomly select edges to delete
        edges_to_delete = []
        for edge in edges:
            if np.random.random() < noise_level:
                edges_to_delete.append(edge)

        # Ensure we keep at least one edge
        if len(edges_to_delete) >= num_edges:
            edge_to_keep = edges[np.random.randint(num_edges)]
            edges_to_delete = [e for e in edges_to_delete if e != edge_to_keep]

        # Remove selected edges
        perturbed.remove_edges_from(edges_to_delete)

        return perturbed
