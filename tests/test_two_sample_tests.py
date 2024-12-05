import random

import numpy as np
import pytest
from scipy.stats import kstest

from graph_gen_gym.metrics.two_sample_tests import BootStrapMMDTest, ClassifierTest
from graph_gen_gym.metrics.two_sample_tests.classifier_test import AccuracyInterval


def _is_valid_two_sample_test(
    all_samples, test_function, num_iters=500, threshold=0.05
):
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
    return res.pvalue >= threshold


def test_bootstrap_test(datasets, degree_linear_kernel):
    planar, sbm = datasets
    tst = BootStrapMMDTest(sbm.to_nx(), degree_linear_kernel)
    p_value = tst.compute(planar.to_nx())
    assert 0 <= p_value <= 1

    def _bootstrap_tst_function(samples_a, samples_b):
        tst = BootStrapMMDTest(samples_a, degree_linear_kernel)
        res = tst.compute(samples_b)
        return res

    assert _is_valid_two_sample_test(list(planar.to_nx()), _bootstrap_tst_function)


@pytest.mark.parametrize(
    "kernel", ["degree_linear_kernel", "fast_stacked_kernel", "degree_rbf_kernel"]
)
def test_classifier_test(request, datasets, kernel):
    kernel = request.getfixturevalue(kernel)
    planar, sbm = datasets
    tst = ClassifierTest(sbm.to_nx(), kernel)
    result = tst.compute(planar.to_nx())
    assert isinstance(result, AccuracyInterval)
    assert hasattr(result, "mean")
    assert hasattr(result, "low")
    assert hasattr(result, "high")
    assert hasattr(result, "pval")

    # We expect the classifier test to be able to distinguish planar and SBM
    assert result.pval <= 0.05
    assert result.mean >= 0.6

    def _classifier_tst_function(samples_a, samples_b):
        tst = ClassifierTest(samples_a, kernel)
        res = tst.compute(samples_b, num_samples=10, pvalue_method="permutation")
        return res.pval

    assert _is_valid_two_sample_test(
        list(planar.to_nx()),
        _classifier_tst_function,
    )
