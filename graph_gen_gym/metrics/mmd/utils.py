from typing import Callable, Iterable, Literal, Tuple

import networkx as nx
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


def mmd_ustat_var(kxx: np.ndarray, kyy: np.ndarray, kxy: np.ndarray):
    m = kxx.shape[0]
    n = kyy.shape[0]

    Kxd = kxx - _multi_dim_diag(kxx, axis1=0, axis2=1)
    Kyd = kyy - _multi_dim_diag(kyy, axis1=0, axis2=1)
    if kxx.ndim == 2:
        v = np.zeros(11)
    else:
        assert kxx.ndim == 3
        v = np.zeros((11, kxx.shape[2]))

    Kxd_sum = np.sum(Kxd, axis=(0, 1))
    Kyd_sum = np.sum(Kyd, axis=(0, 1))
    Kxy_sum = np.sum(kxy, axis=(0, 1))
    Kxy2_sum = np.sum(kxy**2, axis=(0, 1))
    Kxd0_red = np.sum(Kxd, 1)
    Kyd0_red = np.sum(Kyd, 1)
    Kxy1 = np.sum(kxy, 1)
    Kyx1 = np.sum(kxy, 0)

    #  varEst = 1/m/(m-1)/(m-2)    * ( sum(Kxd,1)*sum(Kxd,2) - sum(sum(Kxd.^2)))  ...
    v[0] = (
        1.0
        / m
        / (m - 1)
        / (m - 2)
        * (_dot(Kxd0_red, Kxd0_red, axis=0) - np.sum(Kxd**2, axis=(0, 1)))
    )
    #           -  (  1/m/(m-1)   *  sum(sum(Kxd))  )^2 ...
    v[1] = -((1.0 / m / (m - 1) * Kxd_sum) ** 2)
    #           -  2/m/(m-1)/n     *  sum(Kxd,1) * sum(Kxy,2)  ...
    v[2] = -2.0 / m / (m - 1) / n * _dot(Kxd0_red, Kxy1, axis=0)
    #           +  2/m^2/(m-1)/n   * sum(sum(Kxd))*sum(sum(Kxy)) ...
    v[3] = 2.0 / (m**2) / (m - 1) / n * Kxd_sum * Kxy_sum
    #           +  1/(n)/(n-1)/(n-2) * ( sum(Kyd,1)*sum(Kyd,2) - sum(sum(Kyd.^2)))  ...
    v[4] = (
        1.0
        / n
        / (n - 1)
        / (n - 2)
        * (_dot(Kyd0_red, Kyd0_red, axis=0) - np.sum(Kyd**2, axis=(0, 1)))
    )
    #           -  ( 1/n/(n-1)   * sum(sum(Kyd))  )^2	...
    v[5] = -((1.0 / n / (n - 1) * Kyd_sum) ** 2)
    #           -  2/n/(n-1)/m     * sum(Kyd,1) * sum(Kxy',2)  ...
    v[6] = -2.0 / n / (n - 1) / m * _dot(Kyd0_red, Kyx1, axis=0)

    #           +  2/n^2/(n-1)/m  * sum(sum(Kyd))*sum(sum(Kxy)) ...
    v[7] = 2.0 / (n**2) / (n - 1) / m * Kyd_sum * Kxy_sum
    #           +  1/n/(n-1)/m   * ( sum(Kxy',1)*sum(Kxy,2) -sum(sum(Kxy.^2))  ) ...
    v[8] = 1.0 / n / (n - 1) / m * (_dot(Kxy1, Kxy1, axis=0) - Kxy2_sum)
    #           - 2*(1/n/m        * sum(sum(Kxy))  )^2 ...
    v[9] = -2.0 * (1.0 / n / m * Kxy_sum) ** 2
    #           +   1/m/(m-1)/n   *  ( sum(Kxy,1)*sum(Kxy',2) - sum(sum(Kxy.^2)))  ;
    v[10] = 1.0 / m / (m - 1) / n * (_dot(Kyx1, Kyx1, axis=0) - Kxy2_sum)

    # %additional low order correction made to some terms compared with ICLR submission
    # %these corrections are of the same order as the 2nd order term and will
    # %be unimportant far from the null.

    #   %Eq. 13 p. 11 ICLR 2016. This uses ONLY first order term
    #   varEst = 4*(m-2)/m/(m-1) *  varEst  ;
    varEst1st = 4.0 * (m - 2) / m / (m - 1) * np.sum(v, axis=0)

    Kxyd = kxy - _multi_dim_diag(kxy, axis1=0, axis2=1)
    #   %Eq. 13 p. 11 ICLR 2016: correction by adding 2nd order term
    #   varEst2nd = 2/m/(m-1) * 1/n/(n-1) * sum(sum( (Kxd + Kyd - Kxyd - Kxyd').^2 ));
    varEst2nd = (
        2.0
        / m
        / (m - 1)
        * 1
        / n
        / (n - 1)
        * np.sum((Kxd + Kyd - Kxyd - Kxyd.T) ** 2, axis=(0, 1))
    )

    #   varEst = varEst + varEst2nd;
    varEst = varEst1st + varEst2nd

    #   %use only 2nd order term if variance estimate negative
    if varEst < 0:
        varEst = varEst2nd

    return varEst


def _get_batch_description(
    graphs: Iterable[nx.Graph],
    descriptor_fn: Callable[[nx.Graph], np.ndarray],
    zero_padding: bool,
) -> np.ndarray:
    descriptions = [descriptor_fn(graph) for graph in graphs]
    if zero_padding:
        max_length = max(len(descr) for descr in descriptions)
        descriptions = [
            np.concatenate((descr, np.zeros(max_length - len(descr))))
            for descr in descriptions
        ]
    return np.stack(descriptions)


def _pad_arrays(
    x: np.ndarray, y: np.ndarray, zero_padding: bool
) -> Tuple[np.ndarray, np.ndarray]:
    assert x.ndim == 2 and y.ndim == 2
    if x.shape[1] == y.shape[1]:
        return x, y
    if zero_padding:
        max_length = max(x.shape[1], y.shape[1])
        x = np.concatenate(
            (
                x,
                np.zeros((x.shape[0], max_length - x.shape[1])),
            ),
            axis=1,
        )
        y = np.concatenate(
            (
                y,
                np.zeros((y.shape[0], max_length - y.shape[1])),
            ),
            axis=1,
        )
        return x, y
    raise ValueError(
        "Dimensions of descriptors does not match but `zero_padding` was not set to `True`."
    )
