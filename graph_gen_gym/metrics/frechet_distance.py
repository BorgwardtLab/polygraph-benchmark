from collections import namedtuple
from itertools import islice
from typing import Callable, Collection

import networkx as nx
import numpy as np
import scipy

GaussianParameters = namedtuple("GaussianParameters", ["mean", "covariance"])


def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            return
        yield batch


class FrechetDistance:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptor_fn: Callable[[Collection[nx.Graph]], np.ndarray],
        batch_size: float = None,
    ):
        self._descriptor_fn = descriptor_fn
        self._batch_size = batch_size
        self._dim = None
        self._reference_gaussian = self._fit_gaussian(reference_graphs)

    def compute(self, generated_graphs: Collection[nx.Graph]):
        generated_gaussian = self._fit_gaussian(generated_graphs)
        return self._compute_wasserstein_distance(
            self._reference_gaussian, generated_gaussian
        )

    def _fit_gaussian(self, graphs: Collection[nx.Graph]):
        mean = None
        cov = None
        data_iter = (
            batched(graphs, self._batch_size)
            if self._batch_size is not None
            else (graphs,)
        )
        batch_size = self._batch_size if self._batch_size is not None else len(graphs)
        num_batches = 0

        for graph_batch in data_iter:
            n_graphs = len(graph_batch)
            representations = self._descriptor_fn(graph_batch)
            assert representations.ndim == 2 and representations.shape[0] == n_graphs
            if self._dim is None:
                self._dim = representations.shape[1]
            assert representations.shape[1] == self._dim
            if mean is None:
                mean = np.zeros(self._dim)
                cov = np.zeros((self._dim, self._dim))

            mean += (n_graphs / batch_size) * representations.mean(axis=0)
            cov += (n_graphs / batch_size) * np.einsum(
                "k i, k j -> k i j", representations, representations
            ).mean(axis=0)
            num_batches += 1

        mean = mean / num_batches
        cov = cov / num_batches - np.einsum("i, j -> i j", mean, mean)
        return GaussianParameters(mean=mean, covariance=cov)

    @staticmethod
    def _compute_wasserstein_distance(
        gaussian_a: GaussianParameters,
        gaussian_b: GaussianParameters,
        eps: float = 1e-6,
    ):
        """Based on https://github.com/bioinf-jku/FCD/blob/375216cfb074b0948b5a649210bd66b839df52b4/fcd/utils.py#L158"""
        assert gaussian_a.mean.shape == gaussian_b.mean.shape
        assert gaussian_a.covariance.shape == gaussian_b.covariance.shape

        mean_diff = gaussian_a.mean - gaussian_b.mean

        # product might be almost singular
        covmean, _ = scipy.linalg.sqrtm(
            gaussian_a.covariance @ gaussian_b.covariance, disp=False
        )
        is_real = np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)

        if not np.isfinite(covmean).all() or not is_real:
            offset = np.eye(gaussian_a.covariance.shape[0]) * eps
            covmean = scipy.linalg.sqrtm(
                (gaussian_a.covariance + offset) @ (gaussian_b.covariance + offset)
            )

        assert isinstance(covmean, np.ndarray)
        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return float(
            mean_diff.dot(mean_diff)
            + np.trace(gaussian_a.covariance)
            + np.trace(gaussian_b.covariance)
            - 2 * tr_covmean
        )
