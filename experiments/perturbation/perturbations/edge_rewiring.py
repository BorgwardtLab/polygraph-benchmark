import networkx as nx
import numpy as np
import random

from .base import BasePerturbation


class EdgeRewiringPerturbation(BasePerturbation):
    """Perturbation that rewires edges by disconnecting and reconnecting endpoints.

    The perturbation randomly selects edges and rewires one of their endpoints
    to a new random node, while avoiding self-loops.
    """

    def perturb(self, graph: nx.Graph, noise_level: float) -> nx.Graph:
        """Rewire edges in the graph based on the noise level.

        Args:
            graph: The input graph to perturb
            noise_level: Probability of selecting each edge for rewiring (0.0 to 1.0)

        Returns:
            A new graph with rewired edges
        """
        if not (0 <= noise_level <= 1):
            raise ValueError("Noise level must be between 0 and 1")

        # Create a copy of the graph to modify
        perturbed = graph.copy()
        edges = list(perturbed.edges())
        nodes = list(perturbed.nodes())

        # Select edges to rewire using binomial distribution
        edges_to_rewire = np.random.binomial(1, noise_level, size=len(edges))
        edge_indices_to_rewire = np.where(edges_to_rewire == 1)[0]

        for edge_index in edge_indices_to_rewire:
            edge = edges[edge_index]
            perturbed.remove_edge(*edge)

            # Randomly choose which endpoint to keep
            if random.random() > 0.5:
                keep_node, detach_node = edge
            else:
                detach_node, keep_node = edge

            # Pick a random node besides detach_node and keep_node
            possible_nodes = [
                n for n in nodes if n not in [keep_node, detach_node]
            ]
            if (
                possible_nodes
            ):  # Only rewire if there are valid nodes to connect to
                attach_node = random.choice(possible_nodes)
                perturbed.add_edge(keep_node, attach_node)
            else:
                # If no valid nodes, restore original edge
                perturbed.add_edge(*edge)

        return perturbed
