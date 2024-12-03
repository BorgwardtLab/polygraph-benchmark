from collections import namedtuple
from typing import Collection, Literal

import networkx as nx
import numpy as np
from scipy.stats import binomtest

from graph_gen_gym.metrics.mmd.kernels import DescriptorKernel, StackedKernel
from graph_gen_gym.metrics.mmd.utils import full_gram_from_blocks

AccuracyInterval = namedtuple("AccuracyInterval", ["mean", "low", "high", "pval"])


class ClassifierTest:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        kernel: DescriptorKernel,
    ):
        self._kernel = kernel
        if not isinstance(self._kernel, StackedKernel):
            # We don't need validation to find the optimal kernel in this case
            self._validate = False
        else:
            self._validate = True
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
        pos_block = train_vs_eval[train_labels]
        neg_block = train_vs_eval[~train_labels]
        # Compute mean difference between positive and negative examples
        # This is a simple linear classifier that assigns class based on whether
        # the kernel similarity to positive examples is higher than to negative examples
        result = pos_block.mean(axis=0) - neg_block.mean(axis=0)
        return result > 0

    def compute(
        self,
        generated: Collection[nx.Graph],
        num_samples: int = 100,
        pvalue_method: Literal["binomial", "permutation"] = "binomial",
    ):
        assert len(generated) == self._num_graphs
        gen_desc = self._kernel.featurize(generated)
        gen_vs_gen = self._kernel(gen_desc, gen_desc)
        ref_vs_gen = self._kernel(self._reference_desc, gen_desc)

        full_gram = full_gram_from_blocks(self._ref_vs_ref, ref_vs_gen, gen_vs_gen)

        samples = []
        predicted_label_samples = []
        true_label_samples = []
        num_val = self._num_graphs // 2
        rng = np.random.default_rng(42)
        for _ in range(num_samples):
            perm = rng.permutation(2 * self._num_graphs)
            labels = perm < self._num_graphs
            train_indices, test_indices = (
                perm[: self._num_graphs],
                perm[self._num_graphs :],
            )
            train_labels, test_labels = (
                labels[: self._num_graphs],
                labels[self._num_graphs :],
            )
            if self._validate:
                train_indices, val_indices = (
                    train_indices[:num_val],
                    train_indices[num_val:],
                )
                train_labels, val_labels = (
                    train_labels[:num_val],
                    train_labels[num_val:],
                )

                train_vs_val = full_gram[train_indices][:, val_indices]
                val_pred = self._compute_labels(train_vs_val, train_labels)
                assert val_pred.ndim == 2, f"Kernel must be a {StackedKernel.__name__}"
                val_correct = np.sum(val_pred == np.expand_dims(val_labels, 1), axis=0)
                optimal_idx = np.argmax(val_correct)
                train_vs_test = full_gram[train_indices][:, test_indices][
                    ..., optimal_idx
                ]
            else:
                train_vs_test = full_gram[train_indices][:, test_indices]
            test_pred = self._compute_labels(train_vs_test, train_labels)
            assert len(test_pred) == self._num_graphs
            assert len(test_labels) == self._num_graphs
            assert test_pred.ndim == 1
            test_correct = np.sum(test_pred == test_labels)
            samples.append(test_correct)
            predicted_label_samples.append(test_pred)
            true_label_samples.append(test_labels)

        accuracies = np.array(samples) / self._num_graphs
        mean_acc = np.mean(accuracies)
        low, high = (
            np.quantile(accuracies, 0.05),
            np.quantile(accuracies, 0.95),
        )
        if pvalue_method == "binomial":
            pval = binomtest(
                k=samples[0], n=self._num_graphs, p=0.5, alternative="greater"
            ).pvalue
        elif pvalue_method == "permutation":
            pred, truth = predicted_label_samples[0], true_label_samples[0]
            assert pred.ndim == 1 and truth.ndim == 1
            assert len(pred) == len(truth)
            num_correct = []
            for _ in range(1000):
                perm = rng.permutation(len(pred))
                perm_truth = truth[perm]
                num_correct.append(np.sum(perm_truth == pred))
            num_correct = np.array(num_correct)
            pval = np.sum(num_correct >= samples[0]) / len(num_correct)
        else:
            raise NotImplementedError
        return AccuracyInterval(mean=mean_acc, low=low, high=high, pval=pval)
