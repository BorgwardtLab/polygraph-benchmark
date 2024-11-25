import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from graph_gen_gym.datasets.dataset import GraphDataset
from graph_gen_gym.datasets.graph import Graph
from graph_gen_gym.datasets.spectre import PlanarGraphDataset, SBMGraphDataset
from graph_gen_gym.metrics.mmd.classifier_test import AccuracyInterval, ClassifierTest
from graph_gen_gym.metrics.mmd.graph_descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    OrbitCounts,
)
from graph_gen_gym.metrics.mmd.kernels import (
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


@pytest.fixture
def datasets():
    planar = PlanarGraphDataset("train")
    sbm = SBMGraphDataset("train")
    return planar, sbm


@pytest.fixture
def rbf_kernel():
    return RBFKernel(OrbitCounts(), bw=np.linspace(0.01, 20, 100))


@pytest.fixture
def linear_kernel():
    return LinearKernel(DegreeHistogram(max_degree=200))


@pytest.fixture
def laplace_kernel():
    return LaplaceKernel(ClusteringHistogram(bins=100), lbd=np.linspace(0.01, 20, 100))


@pytest.fixture
def kernels(rbf_kernel, linear_kernel, laplace_kernel):
    return StackedKernel([rbf_kernel, linear_kernel, laplace_kernel])


def test_dataset_loading(datasets):
    planar, sbm = datasets
    assert len(planar) > 0
    assert len(sbm) > 0


def test_kernel_stacking(kernels):
    assert kernels.num_kernels == 201


def test_mmd_computation(datasets, kernels):
    planar, sbm = datasets
    mmd = DescriptorMMD2(sbm.to_nx(), kernels, variant="umve")
    result = mmd.compute(planar.to_nx())
    assert isinstance(result, np.ndarray)
    assert len(result) == kernels.num_kernels


def test_mmd_computation_ustat_var(datasets, linear_kernel):
    planar, sbm = datasets
    mmd = DescriptorMMD2(sbm.to_nx(), linear_kernel, variant="ustat-var")
    result = mmd.compute(planar.to_nx())
    assert isinstance(result, MMDWithVariance)
    assert hasattr(result, "ustat")
    assert hasattr(result, "std")


def test_mmd_computation_ustat_var_multikernel_error(datasets, kernels):
    planar, sbm = datasets
    mmd = DescriptorMMD2(sbm.to_nx(), kernels, variant="ustat-var")
    with pytest.raises(AssertionError):
        mmd.compute(planar.to_nx())


def test_max_mmd(datasets, kernels):
    planar, sbm = datasets
    max_mmd = MaxDescriptorMMD2(sbm.to_nx(), kernels, "umve")
    metric, kernel = max_mmd.compute(planar.to_nx())
    assert isinstance(metric, float)


def test_max_mmd_ustat_var_multikernel_error(datasets, kernels):
    planar, sbm = datasets
    with pytest.raises(AssertionError):
        max_mmd = MaxDescriptorMMD2(sbm.to_nx(), kernels, "ustat-var")
        max_mmd.compute(planar.to_nx())


def test_bootstrap_test(datasets, linear_kernel):
    planar, sbm = datasets
    tst = BootStrapMMDTest(sbm.to_nx(), linear_kernel)
    p_value = tst.compute(planar.to_nx())
    assert 0 <= p_value <= 1


def test_classifier_test(datasets, linear_kernel):
    planar, sbm = datasets
    tst = ClassifierTest(sbm.to_nx(), linear_kernel)
    result = tst.compute(planar.to_nx())
    assert isinstance(result, AccuracyInterval)
    assert hasattr(result, "mean")
    assert hasattr(result, "low")
    assert hasattr(result, "high")
    assert hasattr(result, "pval")


def test_classifier_test_stacked(datasets, kernels):
    planar, sbm = datasets
    tst = ClassifierTest(sbm.to_nx(), kernels)
    result = tst.compute(planar.to_nx())
    assert isinstance(result, AccuracyInterval)
    assert hasattr(result, "mean")
    assert hasattr(result, "low")
    assert hasattr(result, "high")
    assert hasattr(result, "pval")


def test_variance_computation_correctness(datasets, linear_kernel):
    planar, sbm = datasets
    n_samples = 100
    n_bootstrap = 100
    for _ in range(n_bootstrap):
        idx_to_sample = torch.randperm(len(planar))[:n_samples]
        planar_samples = planar[idx_to_sample]
        sbm_samples = sbm[idx_to_sample]
        planar_subset = GraphDataset(
            Graph.from_pyg_batch(
                Batch.from_data_list(planar_samples), compute_indexing_info=True
            )
        )
        sbm_subset = GraphDataset(
            Graph.from_pyg_batch(
                Batch.from_data_list(sbm_samples), compute_indexing_info=True
            )
        )
        mmd = DescriptorMMD2(planar_subset.to_nx(), linear_kernel, variant="ustat-var")
        result = mmd.compute(sbm_subset.to_nx())
        assert isinstance(result, MMDWithVariance)
        assert hasattr(result, "std")
