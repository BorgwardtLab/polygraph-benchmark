from copy import deepcopy

import networkx as nx

from graph_gen_gym.datasets.dataset import OnlineGraphDataset


class LobsterGraphDataset(OnlineGraphDataset):
    _URL_FOR_SPLIT = {
        "train": "https://datashare.biochem.mpg.de/s/mU8mA2GqfssxUFt/download",
        "val": "https://datashare.biochem.mpg.de/s/KTicVKdP6LgTKeV/download",
        "test": "https://datashare.biochem.mpg.de/s/eYS8K0E6IQ7gZ7j/download",
    }

    def url_for_split(self, split: str):
        return self._URL_FOR_SPLIT[split]

    def is_valid(self, graph: nx.Graph) -> bool:
        """Based on https://github.com/lrjconan/GRAN/blob/fc9c04a3f002c55acf892f864c03c6040947bc6b/utils/eval_helper.py#L426C3-L446C17"""
        graph = deepcopy(graph)
        if nx.is_tree(graph):
            leaves = [n for n, d in graph.degree() if d == 1]
            graph.remove_nodes_from(leaves)

            leaves = [n for n, d in graph.degree() if d == 1]
            graph.remove_nodes_from(leaves)

            num_nodes = len(graph.nodes())
            num_degree_one = [d for n, d in graph.degree() if d == 1]
            num_degree_two = [d for n, d in graph.degree() if d == 2]

            if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
                return True
            elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
                return True
            return False
        else:
            return False
