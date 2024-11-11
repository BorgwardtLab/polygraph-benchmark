from functools import partial
from typing import Callable, Iterable

import networkx as nx
import numpy as np

from graph_gen_gym.datasets.abstract_dataset import AbstractDataset
from graph_gen_gym.metrics.mmd.kernels import DescriptorKernel
from graph_gen_gym.metrics.mmd.utils import _get_batch_description, _pad_arrays


class ClassifierTest:
    def __init__(
        self,
        reference_graphs: AbstractDataset,
        descriptor_fn: Callable[[nx.Graph], np.ndarray],
        kernel: DescriptorKernel,
        zero_padding: bool = False,
    ):
        self._descriptor_fn = descriptor_fn
        self._kernel = kernel
        self._zero_padding = zero_padding
        self._reference_desc = _get_batch_description(
            reference_graphs.to_nx(), self._descriptor_fn, self._zero_padding
        )

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

    def compute(self, generated: Iterable[nx.Graph]):
        gen_desc = _get_batch_description(
            generated, self._descriptor_fn, self._zero_padding
        )
        assert len(gen_desc) == len(self._reference_desc)
        n = len(gen_desc)
        gen_desc, ref_desc = _pad_arrays(
            gen_desc, self._reference_desc, self._zero_padding
        )
        agg_desc = np.concatenate([gen_desc, ref_desc], axis=0)
        idx = np.random.permutation(len(agg_desc))
        is_generated = idx < len(gen_desc)
        agg_desc = agg_desc[idx]
        train_desc, train_label = agg_desc[:n], is_generated[:n]
        test_desc, test_label = agg_desc[n:], is_generated[:n]

        # Now split training further into training and validation

        assert len(gen_val_desc) == len(self._reference_val)
        assert len(gen_test_desc) == len(self._reference_test), (
            gen_test_desc.shape,
            self._reference_test.shape,
        )

        classifier = partial(
            self._compute_labels,
            kernel=self._kernel,
            pos_desc=gen_train_desc,
            neg_desc=self._reference_train,
        )
        accuracy_val = self._compute_accuracy(
            classifier, gen_val_desc, self._reference_val
        )
        print(accuracy_val)
        assert accuracy_val.ndim == 1

        optimal_kernel_idx = np.argmax(accuracy_val)
        optimal_kernel = self._kernel.get_subkernel(optimal_kernel_idx)
        optimal_classifier = partial(
            self._compute_labels,
            kernel=optimal_kernel,
            pos_desc=gen_train_desc,
            neg_desc=self._reference_train,
        )
        accuracy_test = self._compute_accuracy(
            optimal_classifier, gen_test_desc, self._reference_test
        )
        return accuracy_test
