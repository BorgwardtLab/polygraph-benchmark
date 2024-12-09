from typing import Literal

import numpy as np


def mmd_from_gram(
    kxx: np.ndarray,
    kyy: np.ndarray,
    kxy: np.ndarray,
    variant: Literal["biased", "umve", "ustat"],
):
    assert kxx.shape[0] == kxx.shape[1] and kyy.shape[0] == kyy.shape[1]
    n, m = kxx.shape[0], kyy.shape[1]
    assert kxy.shape[:2] == (n, m)

    if variant == "biased":
        xvx = kxx.sum(axis=(0, 1)) / (n**2)
        yvy = kyy.sum(axis=(0, 1)) / (m**2)
        xvy = kxy.sum(axis=(0, 1)) / (n * m)
    elif variant in ["umve", "ustat"]:
        xvx = (kxx.sum(axis=(0, 1)) - np.trace(kxx, axis1=0, axis2=1)) / (n * (n - 1))
        yvy = (kyy.sum(axis=(0, 1)) - np.trace(kyy, axis1=0, axis2=1)) / (m * (m - 1))
        if variant == "ustat":
            if n != m:
                raise RuntimeError
            xvy = (kxy.sum(axis=(0, 1)) - np.trace(kxy, axis1=0, axis2=1)) / (
                n * (n - 1)
            )
        else:
            xvy = kxy.sum(axis=(0, 1)) / (n * m)
    else:
        raise ValueError

    return xvx + yvy - 2 * xvy


def full_gram_from_blocks(kxx, kxy, kyy):
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


def _multi_dim_diag(x, axis1=0, axis2=1):
    assert x.shape[axis1] == x.shape[axis2]
    idx = [None for _ in range(x.ndim)]
    idx[axis1] = np.arange(x.shape[axis1])
    idx[axis2] = np.arange(x.shape[axis2])
    idx = tuple(idx)
    result = np.zeros_like(x)
    result[idx] = x[idx]
    return result


def _dot(x, y, axis=0):
    return (x * y).sum(axis=axis)


def fall_fact(n, length):
    """
    Compute the falling factorial of n of length k.
    """
    return np.prod(np.arange(n, n - length, -1))


def mmd_ustat_var(kxx: np.ndarray, kxy: np.ndarray, kyy: np.ndarray) -> np.ndarray:
    """
    Compute the unbiased MMD variance using the second version of the formula.
    Eq 5 of https://arxiv.org/abs/1906.02104

    Args:
        kxx: Kernel matrix between first sample and itself. Shape (m, m).
        kxy: Kernel matrix between first and second sample. Shape (m, n).
        kyy: Kernel matrix between second sample and itself. Shape (n, n).

    Returns:
        float: The unbiased MMD variance estimate.
    """
    m = kxx.shape[0]
    n = kyy.shape[0]

    ones_m = np.ones(m)
    ones_n = np.ones(n)

    m_2 = fall_fact(m, 2)  # (m)₂
    n_4 = fall_fact(n, 4)  # (n)₄
    n_3 = fall_fact(n, 3)  # (n)₃

    # First term
    term1 = (4 * (m * n + m - 2 * n) / (m_2 * n_4)) * (
        np.linalg.norm(kxx @ ones_n) ** 2 + np.linalg.norm(kyy @ ones_n) ** 2
    )

    # Second term
    term2 = -(2 * (2 * m - n) / (m * n * (m - 1) * (n - 2) * (n - 3))) * (
        np.linalg.norm(kxx, "fro") ** 2 + np.linalg.norm(kyy, "fro") ** 2
    )

    # Third term
    term3 = (4 * (m * n + m - 2 * n - 1) / (m_2 * n**2 * (n - 1) ** 2)) * (
        np.linalg.norm(kxy @ ones_n) ** 2 + np.linalg.norm(kxy.T @ ones_m) ** 2
    )

    # Fourth term
    term4 = (
        -(4 * (2 * m - n - 2) / (m_2 * n * (n - 1) ** 2))
        * np.linalg.norm(kxy, "fro") ** 2
    )

    # Fifth term
    term5 = -(2 * (2 * m - 3) / (m_2 * n_4)) * (
        (ones_m.T @ kxx @ ones_n) ** 2 + (ones_m.T @ kyy @ ones_n) ** 2
    )

    # Sixth term
    term6 = (
        -(4 * (2 * m - 3) / (m_2 * n**2 * (n - 1) ** 2))
        * (ones_m.T @ kxy @ ones_n) ** 2
    )

    # Seventh term
    term7 = -(8 / (m * n_3)) * (
        ones_m.T @ kxx @ kxy @ ones_n + ones_m.T @ kyy @ kxy.T @ ones_n
    )

    # Eighth term
    term8 = (8 / (m * n * n_3)) * (
        (ones_m.T @ kxx @ ones_n + ones_m.T @ kyy @ ones_n) * (ones_m.T @ kxy @ ones_n)
    )

    return np.array(term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8)
