import random

import numpy as np
import pytest
from scipy.stats import kstest

from polygraph.two_sample_tests import (
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

    res = kstest(
        p_val_samples, lambda x: np.clip(x, 0, 1), alternative="greater"
    )
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


@pytest.mark.skipif("config.getoption('--skip-slow')")
@pytest.mark.parametrize(
    "kernel", ["degree_rbf_kernel", "degree_adaptive_rbf_kernel"]
)
def test_multi_bootstrap_test(request, datasets, kernel):
    planar, sbm = datasets
    kernel = request.getfixturevalue(kernel)
    if kernel.num_kernels == 1:
        tst = BootStrapMMDTest(sbm.to_nx(), kernel)
        p_value = tst.compute(planar.to_nx())
    else:
        tst = BootStrapMaxMMDTest(sbm.to_nx(), kernel)
        p_value = tst.compute(planar.to_nx())

    assert 0 <= p_value and p_value <= 1


@pytest.mark.skipif("config.getoption('--skip-slow')")
@pytest.mark.parametrize(
    "kernel", ["degree_rbf_kernel", "degree_adaptive_rbf_kernel"]
)
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
