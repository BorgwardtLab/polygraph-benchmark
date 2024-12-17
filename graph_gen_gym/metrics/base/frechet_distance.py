from collections import namedtuple
from itertools import islice
from typing import Callable, Collection

import networkx as nx
import numpy as np
import scipy


__all__ = ["FittedFrechetDistance", "FrechetDistance"]

GaussianParameters = namedtuple("GaussianParameters", ["mean", "covariance"])


def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            return
        yield batch


def compute_wasserstein_distance(
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
            raise ValueError(
                f"Imaginary component {m} for gaussians {gaussian_a}, {gaussian_b}"
            )
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(
        mean_diff.dot(mean_diff)
        + np.trace(gaussian_a.covariance)
        + np.trace(gaussian_b.covariance)
        - 2 * tr_covmean
    )


def fit_gaussian(
    graphs: Collection[nx.Graph],
    descriptor_fn: Callable[[Collection[nx.Graph]], np.ndarray],
):
    representations = descriptor_fn(graphs)
    mean = np.mean(representations, axis=0)
    cov = np.cov(representations, rowvar=False)
    return GaussianParameters(mean=mean, covariance=cov)


class FittedFrechetDistance:
    def __init__(
        self,
        fitted_gaussian: GaussianParameters,
        descriptor_fn: Callable[[Collection[nx.Graph]], np.ndarray],
    ):
        self._reference_gaussian = fitted_gaussian
        self._descriptor_fn = descriptor_fn
        self._dim = None

    def compute(self, generated_graphs: Collection[nx.Graph]):
        generated_gaussian = fit_gaussian(
            generated_graphs,
            self._descriptor_fn,
        )
        return compute_wasserstein_distance(
            self._reference_gaussian, generated_gaussian
        )


class FrechetDistance:
    def __init__(
        self,
        reference_graphs: Collection[nx.Graph],
        descriptor_fn: Callable[[Collection[nx.Graph]], np.ndarray],
    ):
        reference_gaussian = fit_gaussian(
            reference_graphs,
            descriptor_fn,
        )
        self._fd = FittedFrechetDistance(
            reference_gaussian,
            descriptor_fn=descriptor_fn,
        )

    def compute(self, generated_graphs: Collection[nx.Graph]):
        return self._fd.compute(generated_graphs)
