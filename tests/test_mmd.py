import random

import numpy as np
import pytest
import torch
from scipy.stats import kstest
from torch_geometric.data import Batch

from graph_gen_gym.datasets.dataset import GraphDataset
from graph_gen_gym.datasets.graph import Graph
from graph_gen_gym.datasets.spectre import PlanarGraphDataset, SBMGraphDataset
from graph_gen_gym.metrics.graph_descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    OrbitCounts,
)
from graph_gen_gym.metrics.mmd.classifier_test import AccuracyInterval, ClassifierTest
from graph_gen_gym.metrics.mmd.kernels import (
    DescriptorKernel,
    LaplaceKernel,
    LinearKernel,
    RBFKernel,
    StackedKernel,
)
from graph_gen_gym.metrics.mmd.mmd import (
    DescriptorMMD2,
    MaxDescriptorMMD2,
    MMDWithVariance,
)
from graph_gen_gym.metrics.mmd.test import BootStrapMMDTest


def _is_valid_two_sample_test(
    all_samples, test_function, num_iters=500, threshold=0.05
):
    """
    Perform Kolmogorov-Smirnov test to assert that two-sample test is valid.

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


@pytest.fixture
def datasets():
    planar = PlanarGraphDataset("train")
    sbm = SBMGraphDataset("train")
    return planar, sbm


@pytest.fixture
def orbit_rbf_kernel():
    return RBFKernel(OrbitCounts(), bw=np.linspace(0.01, 20, 100))


@pytest.fixture
def degree_linear_kernel():
    return LinearKernel(DegreeHistogram(max_degree=200))


@pytest.fixture
def degree_rbf_kernel():
    return RBFKernel(DegreeHistogram(max_degree=200), bw=np.linspace(0.01, 20, 10))


@pytest.fixture
def clustering_laplace_kernel():
    return LaplaceKernel(ClusteringHistogram(bins=100), lbd=np.linspace(0.01, 20, 100))


@pytest.fixture
def stacked_kernel(orbit_rbf_kernel, degree_linear_kernel, clustering_laplace_kernel):
    return StackedKernel(
        [orbit_rbf_kernel, degree_linear_kernel, clustering_laplace_kernel]
    )


@pytest.fixture
def fast_stacked_kernel(degree_linear_kernel, degree_rbf_kernel):
    return StackedKernel([degree_linear_kernel, degree_rbf_kernel])


def test_dataset_loading(datasets):
    planar, sbm = datasets
    assert len(planar) > 0
    assert len(sbm) > 0


def test_kernel_stacking(stacked_kernel):
    assert stacked_kernel.num_kernels == 201


def test_mmd_computation(datasets, stacked_kernel):
    planar, sbm = datasets
    mmd = DescriptorMMD2(sbm.to_nx(), stacked_kernel, variant="umve")
    result = mmd.compute(planar.to_nx())
    assert isinstance(result, np.ndarray)
    assert len(result) == stacked_kernel.num_kernels


def test_mmd_computation_ustat_var(datasets, degree_linear_kernel):
    planar, sbm = datasets
    mmd = DescriptorMMD2(sbm.to_nx(), degree_linear_kernel, variant="ustat-var")
    result = mmd.compute(planar.to_nx())
    assert isinstance(result, MMDWithVariance)
    assert hasattr(result, "ustat")
    assert hasattr(result, "std")


def test_mmd_computation_ustat_var_multikernel_error(datasets, stacked_kernel):
    planar, sbm = datasets
    mmd = DescriptorMMD2(sbm.to_nx(), stacked_kernel, variant="ustat-var")
    with pytest.raises(AssertionError):
        mmd.compute(planar.to_nx())


def test_max_mmd(datasets, stacked_kernel):
    planar, sbm = datasets
    max_mmd = MaxDescriptorMMD2(sbm.to_nx(), stacked_kernel, "umve")
    metric, kernel = max_mmd.compute(planar.to_nx())
    assert isinstance(metric, float)
    assert isinstance(kernel, DescriptorKernel)


def test_max_mmd_ustat_var_multikernel_error(datasets, stacked_kernel):
    planar, sbm = datasets
    with pytest.raises(AssertionError):
        max_mmd = MaxDescriptorMMD2(sbm.to_nx(), stacked_kernel, "ustat-var")
        max_mmd.compute(planar.to_nx())


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


def test_variance_computation_correctness(datasets, degree_linear_kernel):
    planar, sbm = datasets
    n_samples = 100
    n_bootstrap = 100
    for _ in range(n_bootstrap):
        idx_to_sample = torch.randperm(len(planar))[:n_samples]
        planar_samples = planar[idx_to_sample]
        sbm_samples = sbm[idx_to_sample]
        planar_subset = GraphDataset(
            Graph.from_pyg_batch(Batch.from_data_list(planar_samples))
        )
        sbm_subset = GraphDataset(
            Graph.from_pyg_batch(Batch.from_data_list(sbm_samples))
        )
        mmd = DescriptorMMD2(
            planar_subset.to_nx(), degree_linear_kernel, variant="ustat-var"
        )
        result = mmd.compute(sbm_subset.to_nx())
        assert isinstance(result, MMDWithVariance)
        assert hasattr(result, "std")
