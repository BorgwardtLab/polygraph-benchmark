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
