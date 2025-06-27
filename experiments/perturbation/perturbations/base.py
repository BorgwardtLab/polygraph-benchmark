from typing import Collection, Callable, Dict
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import networkx as nx
import pandas as pd
from tqdm import tqdm
from functools import wraps

from polygraph.utils.graph_descriptors import (
    OrbitCounts,
    DegreeHistogram,
    SparseDegreeHistogram,
    ClusteringHistogram,
    EigenvalueHistogram,
    RandomGIN,
)


def memoize_all_descriptors(verbose: bool = False):
    """Monkey-patch all graph descriptor __call__ methods with memoization"""

    # Dictionary to hold all caches and stats
    caches = {}
    cache_stats = {}

    descriptor_classes = [
        OrbitCounts,
        DegreeHistogram,
        SparseDegreeHistogram,
        ClusteringHistogram,
        EigenvalueHistogram,
        RandomGIN,
    ]

    for desc_class in descriptor_classes:
        class_name = desc_class.__name__

        # Store original method
        original_call = desc_class.__call__

        # Create cache and stats for this class
        caches[class_name] = {}
        cache_stats[class_name] = {"hits": 0, "misses": 0}

        def create_memoized_call(class_name, original_call):
            @wraps(original_call)
            def memoized_call(self, graphs):
                graphs_list = list(graphs)
                cache_key = tuple(id(g) for g in graphs_list)

                if cache_key in caches[class_name]:
                    cache_stats[class_name]["hits"] += 1
                    if verbose:
                        print(
                            f"{class_name} cache HIT ({cache_stats[class_name]['hits']} hits, {cache_stats[class_name]['misses']} misses)"
                        )
                    return caches[class_name][cache_key]

                cache_stats[class_name]["misses"] += 1
                if verbose:
                    print(
                        f"{class_name} cache MISS - computing... ({cache_stats[class_name]['hits']} hits, {cache_stats[class_name]['misses']} misses)"
                    )
                result = original_call(self, graphs_list)
                caches[class_name][cache_key] = result

                return result

            return memoized_call

        # Apply monkey-patch
        desc_class.__call__ = create_memoized_call(class_name, original_call)

    return caches, cache_stats


# Global variable to hold the perturbation instance in each worker process
_worker_perturbation: "BasePerturbation" = None


def init_worker(
    perturbation_instance: "BasePerturbation",
    memoize: bool = False,
    verbose: bool = False,
):
    """
    Initializer for each worker process.
    This function receives the perturbation instance and stores it in a global
    variable, making it accessible to the worker function without needing to
    transfer it for every task.
    """
    global _worker_perturbation
    _worker_perturbation = perturbation_instance

    if memoize:
        memoize_all_descriptors(verbose=verbose)


def run_perturbation_for_noise_level(noise_level: float) -> Dict[str, float]:
    """
    The actual function that runs on the worker process.
    It accesses the globally stored perturbation instance to perform the evaluation.
    """
    return _worker_perturbation._evaluate_single_noise_level(noise_level)


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

    def _evaluate_single_noise_level(
        self, noise_level: float
    ) -> Dict[str, float]:
        """Evaluate a single noise level - used for parallelization."""
        perturbed_graphs = [
            self.perturb(graph, noise_level) for graph in self.perturb_set
        ]
        new_scores = self.evaluator(perturbed_graphs)
        new_scores["noise_level"] = noise_level
        return new_scores

    def evaluate(
        self,
        noise_levels: Collection[float],
        progress_bar: bool = True,
        num_workers: int = 1,
        memoize: bool = False,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate perturbations at different noise levels with optional parallelization.

        Args:
            noise_levels: Collection of noise levels to evaluate
            progress_bar: Whether to show progress bar
            max_workers: Number of worker threads (None for auto-detection)
        """
        scores = []

        pbar = tqdm(
            total=len(noise_levels),
            desc="Evaluating perturbation",
            disable=not progress_bar,
        )

        if num_workers <= 1:
            if memoize:
                memoize_all_descriptors(verbose=verbose)

            for noise_level in noise_levels:
                scores.append(self._evaluate_single_noise_level(noise_level))
                pbar.update(1)
            pbar.close()
            scores.sort(key=lambda x: x["noise_level"])
            df = pd.DataFrame(scores)
            return df

        ctx = multiprocessing.get_context("spawn")

        # Use ProcessPoolExecutor for parallelization, with an initializer
        # to avoid sending the large 'self' object for every task.
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(
                self,
                memoize,
                verbose,
            ),  # Pass the instance to each worker once
        ) as executor:
            # Submit all tasks
            future_to_noise = {
                executor.submit(
                    run_perturbation_for_noise_level, noise_level
                ): noise_level
                for noise_level in noise_levels
            }

            # Collect results as they complete
            for future in as_completed(future_to_noise):
                result = future.result()
                scores.append(result)
                pbar.update(1)

        pbar.close()

        # Sort by noise level to maintain order
        scores.sort(key=lambda x: x["noise_level"])

        df = pd.DataFrame(scores)
        return df
