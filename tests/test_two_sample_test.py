import random

import numpy as np
import pytest
from scipy.stats import kstest

from graph_gen_gym.metrics.mmd_two_sample_test import (
    BootStrapMaxMMDTest,
    BootStrapMMDTest,
)


def _ks_test(all_samples, test_function, num_iters=200):
    """Perform Kolmogorov-Smirnov test to assert that two-sample test is valid.

    We assert that we cannot reject F(x) <= x where F is the CDF of p-values under the null hypothesis.
    """
    num_samples = len(all_samples)

    p_val_samples = []

    random.seed(42)

    for _ in range(num_iters):
        random.shuffle(all_samples)
        samples_a = all_samples[: num_samples // 2]
        samples_b = all_samples[num_samples // 2 :]
        pval = test_function(samples_a, samples_b)
        assert 0 <= pval <= 1
        p_val_samples.append(pval)

    res = kstest(p_val_samples, lambda x: np.clip(x, 0, 1), alternative="greater")
    if res.pvalue < 0.05:
        assert False, p_val_samples
    return res.pvalue


def _create_tst_fn(kernel):
    def _bootstrap_tst_function(samples_a, samples_b):
        tst = BootStrapMMDTest(samples_a, kernel)
        res = tst.compute(samples_b)
        return res

    return _bootstrap_tst_function


def test_bootstrap_test(datasets, degree_linear_kernel):
    planar, sbm = datasets
    tst = BootStrapMMDTest(sbm.to_nx(), degree_linear_kernel)
    p_value = tst.compute(planar.to_nx())
    assert 0 <= p_value <= 0.1

    p = _ks_test(list(planar.to_nx()), _create_tst_fn(degree_linear_kernel))
    assert p > 0.05


@pytest.mark.parametrize("kernel", ["degree_rbf_kernel", "degree_adaptive_rbf_kernel"])
def test_multi_bootstrap_test(request, datasets, kernel):
    planar, sbm = datasets
    kernel = request.getfixturevalue(kernel)
    tst = BootStrapMMDTest(sbm.to_nx(), kernel)
    p_value = tst.compute(planar.to_nx())
    assert isinstance(p_value, np.ndarray)
    assert p_value.ndim == 1 and len(p_value) == kernel.num_kernels
    assert (0 <= p_value).all() and (p_value <= 1).all()

    for i in range(max(kernel.num_kernels, 10)):
        subkernel = kernel.get_subkernel(i)
        sub_tst = BootStrapMMDTest(sbm.to_nx(), subkernel)
        sub_p_value = sub_tst.compute(planar.to_nx())
        assert np.isclose(sub_p_value, p_value[i])
        p = _ks_test(list(planar.to_nx()), _create_tst_fn(subkernel))
        assert p > 0.05


@pytest.mark.parametrize("kernel", ["degree_rbf_kernel", "degree_adaptive_rbf_kernel"])
def test_max_bootstrap_test(request, datasets, kernel):
    planar, sbm = datasets
    kernel = request.getfixturevalue(kernel)
    tst = BootStrapMaxMMDTest(sbm.to_nx(), kernel)
    p_value = tst.compute(planar.to_nx())
    assert (0 <= p_value).all() and (p_value <= 0.1).all()

    def _max_mmd_test(samples_a, samples_b):
        tst = BootStrapMaxMMDTest(samples_a, kernel)
        res = tst.compute(samples_b)
        return res

    p = _ks_test(list(planar.to_nx()), _max_mmd_test)
    assert p > 0.05
