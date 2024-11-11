from functools import partial
from typing import Callable, Collection

import networkx as nx
import numpy as np

from graph_gen_gym.datasets.abstract_dataset import AbstractDataset
from graph_gen_gym.metrics.mmd.kernels import DescriptorKernel


class ClassifierTest:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
    ):
        self._kernel = kernel
        self._reference_desc = self._kernel.featurize(reference_graphs)

    @staticmethod
    def _compute_labels(
        kernel: DescriptorKernel,
        pos_desc: np.ndarray,
        neg_desc: np.ndarray,
        x: np.ndarray,
    ):
        pos_vs_x = kernel(pos_desc, x)
        neg_vs_x = kernel(neg_desc, x)
        output = pos_vs_x.mean(axis=0) - neg_vs_x.mean(axis=0)
        assert len(output) == len(x)
        return np.sign(output)

    @staticmethod
    def _compute_accuracy(classifier: Callable, pos: np.ndarray, neg: np.ndarray):
        assert pos.shape == neg.shape
        assert pos.ndim == 2
        labels_pos = classifier(x=pos)
        labels_neg = classifier(x=neg)
        assert (
            labels_pos.shape[0] == pos.shape[0] and labels_neg.shape[0] == neg.shape[0]
        )
        return (np.sum(labels_pos > 0, axis=0) + np.sum(labels_neg < 0, axis=0)) / (
            2 * len(labels_pos)
        )

    def compute(self, generated: Collection[nx.Graph]):
        n = len(generated)
        gen_desc = self._kernel.featurize(generated)

        raise NotImplementedError
