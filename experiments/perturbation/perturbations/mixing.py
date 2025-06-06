import random
import networkx as nx

from .base import BasePerturbation


class MixingPerturbation(BasePerturbation):
    """Perturbation that mixes original graphs with randomly generated alternatives.

    The perturbation randomly replaces input graphs with Erdős-Rényi random graphs
    that have similar properties (nodes, density) to the reference set.
    """

    def _make_erdos_reny(self, g: nx.Graph) -> nx.Graph:
        """Generate a random ER graph matching reference set statistics."""
        n = g.number_of_nodes()
        p = nx.density(g)
        return nx.erdos_renyi_graph(n, p)

    def perturb(self, graph: nx.Graph, noise_level: float) -> nx.Graph:
        """Mix original graph with random graph based on noise level.

        Args:
            graph: The input graph to potentially replace
            noise_level: Probability of replacing with random graph (0.0 to 1.0)

        Returns:
            Either the original graph or a newly generated random graph
        """
        if not (0 <= noise_level <= 1):
            raise ValueError("Noise level must be between 0 and 1")

        # Randomly decide whether to replace the graph
        if random.random() < noise_level:
            # Generate and return a new random graph
            return self._make_erdos_reny(graph)

        return graph.copy()
