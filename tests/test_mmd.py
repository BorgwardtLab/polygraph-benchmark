import random

import numpy as np
import pytest
import torch
from gran_mmd_implementation.stats import (
    clustering_stats,
    degree_stats,
    orbit_stats_all,
    spectral_stats,
)
from torch_geometric.data import Batch

from graph_gen_gym.datasets.dataset import GraphDataset
from graph_gen_gym.datasets.graph import Graph
from graph_gen_gym.metrics.mmd import (
    DescriptorMMD2,
    DescriptorMMD2Interval,
    GRANClusteringMMD2,
    GRANDegreeMMD2,
    GRANOrbitMMD2,
    GRANSpectralMMD2,
    MaxDescriptorMMD2,
    MMDInterval,
    MMDWithVariance,
)
from graph_gen_gym.metrics.utils.kernels import DescriptorKernel


def test_dataset_loading(datasets):
    planar, sbm = datasets
    assert len(planar) > 0
    assert len(sbm) > 0


@pytest.mark.parametrize(
    "kernel", ["stacked_kernel", "fast_stacked_kernel", "degree_adaptive_rbf_kernel"]
)
def test_mmd_stacking(request, datasets, kernel):
    planar, sbm = datasets
    kernel = request.getfixturevalue(kernel)
    mmd = DescriptorMMD2(sbm.to_nx(), kernel, variant="umve")
    result = mmd.compute(planar.to_nx())
    assert isinstance(result, np.ndarray)
    assert len(result) == kernel.num_kernels

    separate_kernels = [kernel.get_subkernel(i) for i in range(kernel.num_kernels)]
    separate_mmds = [
        DescriptorMMD2(sbm.to_nx(), subkernel, variant="umve")
        for subkernel in separate_kernels
    ]
    separate_results = [sub_mmd.compute(planar.to_nx()) for sub_mmd in separate_mmds]
    assert np.allclose(result, np.array(separate_results))


@pytest.mark.parametrize(
    "kernel,subsample_size,variant",
    [("degree_linear_kernel", 32, "biased"), ("degree_linear_kernel", 40, "umve")],
)
def test_mmd_uncertainty(request, datasets, kernel, subsample_size, variant):
    planar, sbm = datasets
    kernel = request.getfixturevalue(kernel)
    mmd = DescriptorMMD2Interval(sbm.to_nx(), kernel, variant=variant)
    result = mmd.compute(planar.to_nx(), subsample_size=subsample_size)
    assert isinstance(result, MMDInterval)
    assert result.std > 0

    rng = np.random.default_rng(42)
    planar_idxs = rng.choice(len(planar), size=subsample_size, replace=False)
    sbm_idxs = rng.choice(len(sbm), size=subsample_size, replace=False)
    assert len(np.unique(planar_idxs)) == subsample_size
    assert len(np.unique(sbm_idxs)) == subsample_size
    planar_samples = [planar.to_nx()[int(idx)] for idx in planar_idxs]
    sbm_samples = [sbm.to_nx()[int(idx)] for idx in sbm_idxs]

    single_mmd = DescriptorMMD2(sbm_samples, kernel, variant=variant)
    single_estimate = single_mmd.compute(planar_samples)
    assert result.low <= single_estimate <= result.high


def test_gran_equivalence(datasets):
    """Ensure  that our MMD estimate is equivalent to the one by GRAN implementation."""
    planar, sbm = datasets
    planar, sbm = list(planar.to_nx()), list(sbm.to_nx())

    # Test degree MMD, needs special treatment for max_degree
    deg_mmd = GRANDegreeMMD2(
        planar, max_degree=200
    )  # Max degree must be chosen large enough
    assert np.isclose(deg_mmd.compute(sbm), degree_stats(planar, sbm))
    deg_mmd = GRANDegreeMMD2(planar[:64], 128)
    assert np.isclose(
        deg_mmd.compute(planar[64:]), degree_stats(planar[:64], planar[64:])
    )

    # Test all other MMD classes
    for mmd_cls, baseline_method in zip(
        [GRANSpectralMMD2, GRANOrbitMMD2, GRANClusteringMMD2],
        [spectral_stats, orbit_stats_all, clustering_stats],
    ):
        mmd = mmd_cls(planar)
        assert np.isclose(mmd.compute(sbm), baseline_method(planar, sbm))
        mmd = mmd_cls(planar[:64])
        assert np.isclose(
            mmd.compute(planar[64:]), baseline_method(planar[:64], planar[64:])
        )


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


@pytest.mark.parametrize(
    "kernel,variant",
    [
        ("degree_rbf_kernel", "biased"),
        ("degree_adaptive_rbf_kernel", "umve"),
        ("degree_rbf_kernel", "ustat"),
    ],
)
def test_max_mmd(request, datasets, kernel, variant):
    planar, sbm = datasets
    kernel = request.getfixturevalue(kernel)
    max_mmd = MaxDescriptorMMD2(sbm.to_nx(), kernel, variant)
    metric, kernel = max_mmd.compute(planar.to_nx())
    assert isinstance(metric, float)
    assert isinstance(kernel, DescriptorKernel)

    unpooled_mmd = DescriptorMMD2(sbm.to_nx(), kernel, variant)
    metric_arr = unpooled_mmd.compute(planar.to_nx())
    assert np.isclose(metric, np.max(metric_arr))

    # Assert that we actually get the proper kernel
    mmd = DescriptorMMD2(sbm.to_nx(), kernel, variant)
    metric2 = mmd.compute(planar.to_nx())
    assert np.isclose(metric, metric2)


def test_max_mmd_ustat_var_multikernel_error(datasets, stacked_kernel):
    planar, sbm = datasets
    with pytest.raises(AssertionError):
        max_mmd = MaxDescriptorMMD2(sbm.to_nx(), stacked_kernel, "ustat-var")
        max_mmd.compute(planar.to_nx())


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
