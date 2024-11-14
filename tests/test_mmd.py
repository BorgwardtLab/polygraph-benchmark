import numpy as np
from graph_gen_gym.datasets.spectre import PlanarGraphDataset, SBMGraphDataset
from graph_gen_gym.metrics.mmd.mmd import DescriptorMMD2, MaxDescriptorMMD2
from graph_gen_gym.metrics.mmd.graph_descriptors import OrbitCounts, DegreeHistogram, ClusteringHistogram
from graph_gen_gym.metrics.mmd.kernels import RBFKernel, LinearKernel, LaplaceKernel, StackedKernel
from graph_gen_gym.metrics.mmd.mmd_test import BootStrapMMDTest
from graph_gen_gym.metrics.mmd.classifier_test import ClassifierTest, AccuracyInterval


import pytest

@pytest.fixture
def datasets():
    planar = PlanarGraphDataset("train")
    sbm = SBMGraphDataset("train")
    return planar, sbm

@pytest.fixture
def kernel():
    return RBFKernel(OrbitCounts(), bw=np.linspace(0.01, 20, 100))

@pytest.fixture
def kernels():
    kernel1 = RBFKernel(OrbitCounts(), bw=np.linspace(0.01, 20, 100))
    kernel2 = LinearKernel(DegreeHistogram(max_degree=200))
    kernel3 = LaplaceKernel(ClusteringHistogram(bins=100), lbd=np.linspace(0.01, 20, 100))
    return StackedKernel([kernel1, kernel2, kernel3])

def test_dataset_loading(datasets):
    planar, sbm = datasets
    assert len(planar) > 0
    assert len(sbm) > 0

def test_kernel_stacking():
    kernel1 = RBFKernel(OrbitCounts(), bw=np.linspace(0.01, 20, 100))
    kernel2 = LinearKernel(DegreeHistogram(max_degree=200))
    kernel3 = LaplaceKernel(ClusteringHistogram(bins=100), lbd=np.linspace(0.01, 20, 100))
    combined_kernel = StackedKernel([kernel1, kernel2, kernel3])
    assert combined_kernel.num_kernels == 201

def test_mmd_computation(datasets):
    planar, sbm = datasets
    kernel1 = RBFKernel(OrbitCounts(), bw=np.linspace(0.01, 20, 100))
    kernel2 = LinearKernel(DegreeHistogram(max_degree=200))
    combined_kernel = StackedKernel([kernel1, kernel2])
    
    mmd = DescriptorMMD2(sbm.to_nx(), combined_kernel, variant="umve")
    result = mmd.compute(planar.to_nx())
    assert isinstance(result, np.ndarray)
    assert len(result) == combined_kernel.num_kernels

def test_max_mmd(datasets, kernels):
    planar, sbm = datasets
    max_mmd = MaxDescriptorMMD2(sbm.to_nx(), kernels, "umve")
    metric, kernel = max_mmd.compute(planar.to_nx())
    assert isinstance(metric, float)

def test_max_mmd_valueerror(datasets, kernel):
    planar, sbm = datasets
    with pytest.raises(ValueError):
        MaxDescriptorMMD2(sbm.to_nx(), kernel, "umve")

def test_bootstrap_test(datasets):
    planar, sbm = datasets
    kernel = LinearKernel(DegreeHistogram(max_degree=200))
    tst = BootStrapMMDTest(sbm.to_nx(), kernel)
    p_value = tst.compute(planar.to_nx())
    assert 0 <= p_value <= 1

def test_classifier_test(datasets):
    planar, sbm = datasets
    kernel = LinearKernel(DegreeHistogram(max_degree=200))
    tst = ClassifierTest(sbm.to_nx(), kernel)
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
