from collections import namedtuple
from typing import Collection

import networkx as nx
import numpy as np

from graph_gen_gym.metrics.mmd.kernels import DescriptorKernel
from graph_gen_gym.metrics.mmd.utils import full_gram_from_blocks

AccuracyInterval = namedtuple(
    "AccuracyInterval", ["mean", "std", "median", "low", "high"]
)


class ClassifierTest:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
    ):
        assert kernel.num_kernels > 1
        self._kernel = kernel
        self._reference_desc = self._kernel.featurize(reference_graphs)
        self._ref_vs_ref = self._kernel(self._reference_desc, self._reference_desc)
        self._num_graphs = len(reference_graphs)

    @staticmethod
    def _compute_labels(train_vs_eval: np.ndarray, train_labels: np.ndarray):
        assert train_labels.ndim == 1
        assert train_vs_eval.shape[0] == train_labels.shape[0]
        assert (train_labels == 1).sum() + (
            train_labels == 0
        ).sum() == train_labels.shape[0]
        pos_block = train_vs_eval[train_labels == 1]
        neg_block = train_labels[train_labels == 0]
        result = pos_block.mean(axis=0) - neg_block.mean(axis=0)
        return result > 0

    def compute(self, generated: Collection[nx.Graph], num_samples: int = 100):
        assert len(generated) == self._num_graphs
        gen_desc = self._kernel.featurize(generated)
        gen_vs_gen = self._kernel(gen_desc, gen_desc)
        ref_vs_gen = self._kernel(self._reference_desc, gen_desc)

        full_gram = full_gram_from_blocks(self._ref_vs_ref, ref_vs_gen, gen_vs_gen)

        samples = []
        num_val = self._num_graphs // 2
        for _ in range(num_samples):
            perm = np.random.permutation(2 * self._num_graphs)
            labels = perm < self._num_graphs
            train_indices, test_indices = (
                perm[: self._num_graphs],
                perm[: self._num_graphs],
            )
            train_labels, test_labels = (
                labels[: self._num_graphs],
                labels[: self._num_graphs],
            )
            train_indices, val_indices = (
                train_indices[:num_val],
                train_indices[num_val:],
            )
            train_labels, val_labels = train_labels[:num_val], train_labels[num_val:]

            train_vs_val = full_gram[train_indices][:, val_indices]
            val_pred = self._compute_labels(train_vs_val, train_labels)
            assert val_pred.ndim == 2
            accuracy = np.mean(val_pred == np.expand_dims(val_labels, 1), axis=0)
            optimal_idx = np.argmax(accuracy)

            train_vs_test = full_gram[train_indices][:, test_indices][..., optimal_idx]
            test_pred = self._compute_labels(train_vs_test, train_labels)
            assert test_pred.ndim == 1
            test_acc = np.mean(test_pred == test_labels)
            samples.append(test_acc)

        mean_acc, std_acc = np.mean(samples), np.std(samples)
        low, median, high = (
            np.quantile(samples, 0.05),
            np.quantile(samples, 0.5),
            np.quantile(samples, 0.95),
        )
        return AccuracyInterval(
            mean=mean_acc, std=std_acc, low=low, high=high, median=median
        )
