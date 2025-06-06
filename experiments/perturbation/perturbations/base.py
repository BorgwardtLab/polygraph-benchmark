from typing import Collection, Callable, Dict
from abc import ABC, abstractmethod

import networkx as nx
import pandas as pd
from tqdm import tqdm


class BasePerturbation(ABC):
    def __init__(
        self,
        perturb_set: Collection[nx.Graph],
        evaluator: Callable[[Collection[nx.Graph]], Dict[str, float]],
    ):
        self.perturb_set = perturb_set
        self.evaluator = evaluator

    @abstractmethod
    def perturb(self, graph: nx.Graph, noise_level: float) -> nx.Graph: ...

    def evaluate(
        self, noise_levels: Collection[float], progress_bar: bool = True
    ) -> float:
        scores = []
        for noise_level in tqdm(
            noise_levels,
            desc="Evaluating perturbation",
            disable=not progress_bar,
        ):
            perturbed_graphs = [
                self.perturb(graph, noise_level) for graph in self.perturb_set
            ]
            new_scores = self.evaluator(perturbed_graphs)
            new_scores["noise_level"] = noise_level
            scores.append(new_scores)

        df = pd.DataFrame(scores)
        return df
