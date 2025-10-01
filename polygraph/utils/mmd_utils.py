"""Utilities for computing Maximum Mean Discrepancy (MMD) from kernel matrices.

This module provides functions for computing MMD statistics from pre-computed kernel
matrices. The MMD measures the distance between two probability distributions based
on their samples, using kernel-based statistics.
"""

from typing import Literal, Union

import numpy as np
import numba as nb


def mmd_from_gram(
    kxx: np.ndarray,
    kyy: np.ndarray,
    kxy: np.ndarray,
    variant: Literal["biased", "umve", "ustat"],
) -> Union[float, np.ndarray]:
    """Computes MMD statistic from kernel matrices.

    Computes the Maximum Mean Discrepancy between two samples using pre-computed
    kernel matrices. Three estimator variants are available:

    - 'biased': Standard biased V-statistic
    - 'umve': Unbiased minimum variance estimator
    - 'ustat': Unbiased U-statistic (requires equal sample sizes)

    Args:
        kxx: Kernel matrix between first sample points (n×n)
        kyy: Kernel matrix between second sample points (m×m)
        kxy: Kernel matrix between first and second samples (n×m)
        variant: Which MMD estimator to use

    Returns:
        MMD value(s). If kernel matrices have an extra dimension for multiple kernels,
        returns one MMD value per kernel.

    Raises:
        RuntimeError: If variant='ustat' but sample sizes are different
        ValueError: If variant is not one of the supported options
    """
    assert kxx.shape[0] == kxx.shape[1] and kyy.shape[0] == kyy.shape[1]
    n, m = kxx.shape[0], kyy.shape[1]
    assert kxy.shape[:2] == (n, m)

    if variant == "biased":
        xvx = kxx.sum(axis=(0, 1)) / (n**2)
        yvy = kyy.sum(axis=(0, 1)) / (m**2)
        xvy = kxy.sum(axis=(0, 1)) / (n * m)
    elif variant in ["umve", "ustat"]:
        xvx = (kxx.sum(axis=(0, 1)) - np.trace(kxx, axis1=0, axis2=1)) / (
            n * (n - 1)
        )
        yvy = (kyy.sum(axis=(0, 1)) - np.trace(kyy, axis1=0, axis2=1)) / (
            m * (m - 1)
        )
        if variant == "ustat":
            if n != m:
                raise RuntimeError(
                    "Sample sizes must be equal for unbiased MMD"
                )
            xvy = (kxy.sum(axis=(0, 1)) - np.trace(kxy, axis1=0, axis2=1)) / (
                n * (n - 1)
            )
        else:
            xvy = kxy.sum(axis=(0, 1)) / (n * m)
    else:
        raise ValueError

    return xvx + yvy - 2 * xvy


@nb.njit(parallel=True, fastmath=True)
def _compute_mmd_biased(gram, x_idx, y_idx):
    """Numba-accelerated implementation of biased MMD computation."""
    n = len(x_idx)
    m = len(y_idx)

    # Compute kxx sum
    kxx_sum = 0.0
    for i in nb.prange(n):
        for j in range(n):
            kxx_sum += gram[x_idx[i], x_idx[j]]

    # Compute kyy sum
    kyy_sum = 0.0
    for i in nb.prange(m):
        for j in range(m):
            kyy_sum += gram[y_idx[i], y_idx[j]]

    # Compute kxy sum
    kxy_sum = 0.0
    for i in nb.prange(n):
        for j in range(m):
            kxy_sum += gram[x_idx[i], y_idx[j]]

    xvx = kxx_sum / (n**2)
    yvy = kyy_sum / (m**2)
    xvy = kxy_sum / (n * m)

    return xvx + yvy - 2 * xvy


@nb.njit(parallel=True, fastmath=True)
def _compute_mmd_umve(gram, x_idx, y_idx):
    """Numba-accelerated implementation of UMVE MMD computation."""
    n = len(x_idx)
    m = len(y_idx)

    # Compute kxx sum and trace
    kxx_sum = 0.0
    kxx_trace = 0.0
    for i in nb.prange(n):
        kxx_trace += gram[x_idx[i], x_idx[i]]
        for j in range(n):
            kxx_sum += gram[x_idx[i], x_idx[j]]

    # Compute kyy sum and trace
    kyy_sum = 0.0
    kyy_trace = 0.0
    for i in nb.prange(m):
        kyy_trace += gram[y_idx[i], y_idx[i]]
        for j in range(m):
            kyy_sum += gram[y_idx[i], y_idx[j]]

    # Compute kxy sum
    kxy_sum = 0.0
    for i in nb.prange(n):
        for j in range(m):
            kxy_sum += gram[x_idx[i], y_idx[j]]

    xvx = (kxx_sum - kxx_trace) / (n * (n - 1))
    yvy = (kyy_sum - kyy_trace) / (m * (m - 1))
    xvy = kxy_sum / (n * m)

    return xvx + yvy - 2 * xvy


@nb.njit(parallel=True, fastmath=True)
def _compute_mmd_ustat(gram, x_idx, y_idx):
    """Numba-accelerated implementation of U-statistic MMD computation."""
    n = len(x_idx)
    m = len(y_idx)

    if n != m:
        # This will be caught by the main function
        return 0.0

    # Compute kxx sum and trace
    kxx_sum = 0.0
    kxx_trace = 0.0
    for i in nb.prange(n):
        kxx_trace += gram[x_idx[i], x_idx[i]]
        for j in range(n):
            kxx_sum += gram[x_idx[i], x_idx[j]]

    # Compute kyy sum and trace
    kyy_sum = 0.0
    kyy_trace = 0.0
    for i in nb.prange(m):
        kyy_trace += gram[y_idx[i], y_idx[i]]
        for j in range(m):
            kyy_sum += gram[y_idx[i], y_idx[j]]

    # Compute kxy sum and trace
    kxy_sum = 0.0
    kxy_trace = 0.0
    for i in nb.prange(n):
        kxy_trace += gram[x_idx[i], y_idx[i]]
        for j in range(m):
            kxy_sum += gram[x_idx[i], y_idx[j]]

    xvx = (kxx_sum - kxx_trace) / (n * (n - 1))
    yvy = (kyy_sum - kyy_trace) / (m * (m - 1))
    xvy = (kxy_sum - kxy_trace) / (n * (n - 1))

    return xvx + yvy - 2 * xvy


def mmd_from_full_gram(
    gram: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    variant: Literal["biased", "umve", "ustat"],
) -> Union[float, np.ndarray]:
    """Computes MMD from a full kernel matrix and indices partitioning the points.

    Args:
        gram: Full kernel matrix (n+m, n+m) or (n+m, n+m, k) for k kernels
        x_idx: Indices of points in first sample
        y_idx: Indices of points in second sample
        variant: Which MMD estimator to use ("biased", "umve", or "ustat")

    Returns:
        MMD value(s). If kernel matrix has an extra dimension for multiple kernels,
        returns one MMD value per kernel.

    Raises:
        RuntimeError: If variant='ustat' but sample sizes are different
        ValueError: If variant is not one of the supported options
    """
    # Convert indices to numpy arrays if they aren't already
    x_idx = np.asarray(x_idx, dtype=np.int64)
    y_idx = np.asarray(y_idx, dtype=np.int64)

    # Check if we have multiple kernels
    if len(gram.shape) > 2:
        # Handle multiple kernels case
        n_kernels = gram.shape[2]
        results = np.zeros(n_kernels)

        for k in range(n_kernels):
            results[k] = mmd_from_full_gram(
                gram[:, :, k], x_idx, y_idx, variant
            )

        return results

    # Single kernel case
    if variant == "biased":
        return _compute_mmd_biased(gram, x_idx, y_idx)
    elif variant == "umve":
        return _compute_mmd_umve(gram, x_idx, y_idx)
    elif variant == "ustat":
        if len(x_idx) != len(y_idx):
            raise RuntimeError("Sample sizes must be equal for unbiased MMD")
        return _compute_mmd_ustat(gram, x_idx, y_idx)
    else:
        raise ValueError(
            f"Unknown variant: {variant}. Must be one of 'biased', 'umve', or 'ustat'"
        )


def full_gram_from_blocks(
    kxx: np.ndarray, kxy: np.ndarray, kyy: np.ndarray
) -> np.ndarray:
    """Combines kernel block matrices into a single kernel matrix.

    Takes separate kernel matrices for within-sample and between-sample comparisons
    and combines them into a single symmetric kernel matrix for all points.
    Useful for computing MMDs in permutation tests.

    Args:
        kxx: Kernel matrix between first sample points (n×n)
        kxy: Kernel matrix between first and second samples (n×m)
        kyy: Kernel matrix between second sample points (m×m)

    Returns:
        Combined kernel matrix of shape ((n+m)×(n+m))

    Note:
        Input matrices (and output matrix) can have an extra dimension for multiple kernels.
    """
    n, _, *residual_shape = kxx.shape
    m = kyy.shape[0]
    assert np.allclose(kxx, np.swapaxes(kxx, 0, 1)) and np.allclose(
        kyy, np.swapaxes(kyy, 0, 1)
    )
    assert kyy.shape == (m, m, *residual_shape), (kxx.shape, kyy.shape)
    assert kxy.shape == (n, m, *residual_shape)

    full_gram_matrix = np.zeros((n + m, n + m, *residual_shape))

    full_gram_matrix[:n, :n] = kxx
    full_gram_matrix[:n, n:] = kxy
    full_gram_matrix[n:, :n] = np.swapaxes(kxy, 0, 1)
    full_gram_matrix[n:, n:] = kyy
    assert np.allclose(full_gram_matrix, np.swapaxes(full_gram_matrix, 0, 1))
    return full_gram_matrix
