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
from graph_gen_gym.metrics import (
    GRANClusteringMMD2,
    GRANClusteringMMD2Interval,
    GRANDegreeMMD2,
    GRANDegreeMMD2Interval,
    GRANOrbitMMD2,
    GRANOrbitMMD2Interval,
    GRANSpectralMMD2,
    GRANSpectralMMD2Interval,
    RBFClusteringMMD2,
    RBFDegreeMMD2,
    RBFOrbitMMD2,
    RBFSpectralMMD2,
)
from graph_gen_gym.metrics.mmd import (
    DescriptorMMD2,
    DescriptorMMD2Interval,
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
    "kernel,subsample_size,variant",
    [("degree_linear_kernel", 32, "biased"), ("degree_linear_kernel", 40, "umve")],
)
def test_mmd_uncertainty(request, datasets, kernel, subsample_size, variant):
    planar, sbm = datasets
    planar, sbm = list(planar.to_nx()), list(sbm.to_nx())

    kernel = request.getfixturevalue(kernel)
    mmd = DescriptorMMD2Interval(sbm, kernel, variant=variant)
    result = mmd.compute(planar, subsample_size=subsample_size)
    assert isinstance(result, MMDInterval)
    assert result.std > 0

    rng = np.random.default_rng(42)
    planar_idxs = rng.choice(len(planar), size=subsample_size, replace=False)
    sbm_idxs = rng.choice(len(sbm), size=subsample_size, replace=False)
    planar_samples = [planar[int(idx)] for idx in planar_idxs]
    sbm_samples = [sbm[int(idx)] for idx in sbm_idxs]

    single_mmd = DescriptorMMD2(sbm_samples, kernel, variant=variant)
    single_estimate = single_mmd.compute(planar_samples)
    assert result.low <= single_estimate <= result.high


@pytest.mark.parametrize(
    "mmd_cls,baseline_method",
    [
        (GRANSpectralMMD2, spectral_stats),
        (GRANOrbitMMD2, orbit_stats_all),
        (GRANClusteringMMD2, clustering_stats),
        (GRANDegreeMMD2, degree_stats),
    ],
)
def test_gran_equivalence(datasets, orca_executable, mmd_cls, baseline_method):
    """Ensure  that our MMD estimate is equivalent to the one by GRAN implementation."""
    planar, sbm = datasets
    planar, sbm = list(planar.to_nx()), list(sbm.to_nx())

    if baseline_method is orbit_stats_all:
        baseline_method = lambda ref, pred: orbit_stats_all(ref, pred, orca_executable)

    mmd = mmd_cls(planar)
    assert np.isclose(mmd.compute(sbm), baseline_method(planar, sbm)), mmd_cls
    mmd = mmd_cls(planar[:64])
    assert np.isclose(
        mmd.compute(planar[64:]), baseline_method(planar[:64], planar[64:])
    )


@pytest.mark.parametrize("subsample_size", [32, 64, 100, 128])
@pytest.mark.parametrize(
    "single_cls,interval_cls",
    [
        (GRANClusteringMMD2, GRANClusteringMMD2Interval),
        (GRANDegreeMMD2, GRANDegreeMMD2Interval),
        (GRANOrbitMMD2, GRANOrbitMMD2Interval),
        (GRANSpectralMMD2, GRANSpectralMMD2Interval),
    ],
)
def test_gran_uncertainty(datasets, subsample_size, single_cls, interval_cls):
    planar, sbm = datasets
    planar, sbm = list(planar.to_nx()), list(sbm.to_nx())

    assert subsample_size <= len(planar) and subsample_size <= len(sbm)

    rng = np.random.default_rng(1)

    assert issubclass(single_cls, DescriptorMMD2)
    assert issubclass(interval_cls, DescriptorMMD2Interval)

    planar_idxs = rng.choice(len(planar), size=subsample_size, replace=False)
    sbm_idxs = rng.choice(len(sbm), size=subsample_size, replace=False)
    planar_samples = [planar[int(idx)] for idx in planar_idxs]
    sbm_samples = [sbm[int(idx)] for idx in sbm_idxs]

    single_mmd = single_cls(planar_samples)
    interval_mmd = interval_cls(planar)
    interval = interval_mmd.compute(sbm, subsample_size=subsample_size)
    single_estimate = single_mmd.compute(sbm_samples)
    assert isinstance(interval, MMDInterval)
    assert interval.low <= single_estimate <= interval.high


@pytest.mark.parametrize(
    "mmd_cls,stat",
    [
        (RBFClusteringMMD2, "clustering"),
        (RBFDegreeMMD2, "degree"),
        (RBFOrbitMMD2, "orbits"),
        (RBFSpectralMMD2, "spectral"),
    ],
)
def test_rbf_equivalence(datasets, orca_executable, mmd_cls, stat):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent / "ggm_implementation"))
    from ggm_implementation.graph_structure_evaluation import MMDEval

    planar, sbm = datasets
    planar, sbm = list(planar.to_nx()), list(sbm.to_nx())

    baseline_eval = MMDEval(
        statistic=stat, kernel="gaussian_rbf", sigma="range", orca_path=orca_executable
    )
    baseline_results, _ = baseline_eval.evaluate(planar, sbm)
    assert len(baseline_results) == 1
    our_eval = mmd_cls(planar)
    assert np.isclose(our_eval.compute(sbm)[0], list(baseline_results.values())[0])


def test_mmd_computation_ustat_var(datasets, degree_linear_kernel):
    planar, sbm = datasets
    mmd = DescriptorMMD2(sbm.to_nx(), degree_linear_kernel, variant="ustat-var")
    result = mmd.compute(planar.to_nx())
    assert isinstance(result, MMDWithVariance)
    assert hasattr(result, "ustat")
    assert hasattr(result, "std")


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
