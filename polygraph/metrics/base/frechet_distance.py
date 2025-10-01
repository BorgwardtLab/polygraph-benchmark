from collections import namedtuple
from typing import Callable, Collection, Generic

import numpy as np
import scipy

from polygraph import GraphType
from polygraph.metrics.base.interface import GenerationMetric


__all__ = ["FittedFrechetDistance", "FrechetDistance"]

GaussianParameters = namedtuple("GaussianParameters", ["mean", "covariance"])


def compute_wasserstein_distance(
    gaussian_a: GaussianParameters,
    gaussian_b: GaussianParameters,
    eps: float = 1e-6,
):
    """Computes 2-Wasserstein distance between two multivariate Gaussians.

    Based on the closed-form solution for the 2-Wasserstein distance between
    multivariate normal distributions.

    Implementation adapted from:
    https://github.com/bioinf-jku/FCD/blob/375216cfb074b0948b5a649210bd66b839df52b4/fcd/utils.py#L158

    Args:
        gaussian_a: First Gaussian distribution parameters
        gaussian_b: Second Gaussian distribution parameters
        eps: Small constant added to covariance matrices for numerical stability

    Returns:
        float: 2-Wasserstein distance between the distributions

    Raises:
        ValueError: If the matrix square root has a non-negligible imaginary component
    """
    assert gaussian_a.mean.shape == gaussian_b.mean.shape
    assert gaussian_a.covariance.shape == gaussian_b.covariance.shape

    mean_diff = gaussian_a.mean - gaussian_b.mean

    # product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(
        gaussian_a.covariance @ gaussian_b.covariance, disp=False
    )
    is_real = np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)  # pyright: ignore

    if not np.isfinite(covmean).all() or not is_real:
        offset = np.eye(gaussian_a.covariance.shape[0]) * eps
        covmean = scipy.linalg.sqrtm(
            (gaussian_a.covariance + offset) @ (gaussian_b.covariance + offset)
        )

    assert isinstance(covmean, np.ndarray)
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):  # pyright: ignore
            m = np.max(np.abs(covmean.imag))  # pyright: ignore
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
    graphs: Collection[GraphType],
    descriptor_fn: Callable[[Collection[GraphType]], np.ndarray],
):
    """Fits a multivariate Gaussian to graph descriptors.

    Args:
        graphs: Collection of graphs to fit
        descriptor_fn: Function that computes descriptors for a collection of graphs

    Returns:
        GaussianParameters: Fitted mean and covariance
    """
    representations = descriptor_fn(graphs)
    mean = np.mean(representations, axis=0)
    cov = np.cov(representations, rowvar=False)
    return GaussianParameters(mean=mean, covariance=cov)


class FittedFrechetDistance(GenerationMetric[GraphType], Generic[GraphType]):
    """Frechet distance using pre-computed Gaussian parameters.

    This class accepts pre-computed Gaussian parameters rather than fitting them,
    which can be useful when you want to reuse the same reference distribution
    parameters multiple times.

    Args:
        fitted_gaussian: Pre-computed Gaussian parameters for reference distribution
        descriptor_fn: Function that computes descriptors for a collection of graphs
    """

    def __init__(
        self,
        fitted_gaussian: GaussianParameters,
        descriptor_fn: Callable[[Collection[GraphType]], np.ndarray],
    ):
        self._reference_gaussian = fitted_gaussian
        self._descriptor_fn = descriptor_fn
        self._dim = None

    def compute(self, generated_graphs: Collection[GraphType]) -> float:
        """Computes Frechet distance between reference and generated graphs.

        Args:
            generated_graphs: Collection of graphs to evaluate

        Returns:
            float: Frechet distance between reference and generated graphs
        """
        generated_gaussian = fit_gaussian(
            generated_graphs,
            self._descriptor_fn,
        )
        return compute_wasserstein_distance(
            self._reference_gaussian, generated_gaussian
        )


class FrechetDistance(GenerationMetric[GraphType], Generic[GraphType]):
    """Computes Frechet distance between reference and generated graphs.

    The Frechet distance is computed by fitting Gaussian distributions to graph
    descriptors of reference and generated graphs and computing the 2-Wasserstein
    distance between these two distributions.

    Args:
        reference_graphs: Collection of graphs to compare against
        descriptor_fn: Function that computes descriptors for a collection of graphs
    """

    def __init__(
        self,
        reference_graphs: Collection[GraphType],
        descriptor_fn: Callable[[Collection[GraphType]], np.ndarray],
    ):
        reference_gaussian = fit_gaussian(
            reference_graphs,
            descriptor_fn,
        )
        self._fd = FittedFrechetDistance(
            reference_gaussian,
            descriptor_fn=descriptor_fn,
        )

    def compute(self, generated_graphs: Collection[GraphType]) -> float:
        """Computes Frechet distance between reference and generated graphs.

        Args:
            generated_graphs: Collection of graphs to evaluate

        Returns:
            Frechet distance between reference and generated graphs
        """
        return self._fd.compute(generated_graphs)
