import time

import numpy as np
import pytest
from gran_mmd_implementation.stats import (
    clustering_stats,
    degree_stats,
    orbit_stats_all,
    spectral_stats,
)
import networkx as nx

from polygraph.datasets import ProceduralPlanarGraphDataset
from polygraph.metrics.base import (
    DescriptorMMD2,
    DescriptorMMD2Interval,
    MaxDescriptorMMD2,
    MaxDescriptorMMD2Interval,
    MMDInterval,
)
from polygraph.metrics.gran import (
    GRANClusteringMMD2,
    GRANClusteringMMD2Interval,
    GRANDegreeMMD2,
    GRANDegreeMMD2Interval,
    GRANOrbitMMD2,
    GRANOrbitMMD2Interval,
    GRANSpectralMMD2,
    GRANSpectralMMD2Interval,
    RBFClusteringMMD2,
    RBFClusteringMMD2Interval,
    RBFDegreeMMD2,
    RBFDegreeMMD2Interval,
    RBFOrbitMMD2,
    RBFOrbitMMD2Interval,
    RBFSpectralMMD2,
    RBFSpectralMMD2Interval,
)
from polygraph.metrics.gran.linear_mmd import (
    LinearOrbitMMD2,
    LinearClusteringMMD2,
    LinearDegreeMMD2,
    LinearSpectralMMD2,
    LinearOrbitMMD2Interval,
    LinearClusteringMMD2Interval,
    LinearDegreeMMD2Interval,
    LinearSpectralMMD2Interval,
)
from polygraph.utils.kernels import LinearKernel
from polygraph.utils.graph_descriptors import WeisfeilerLehmanDescriptor
from polygraph.utils.mmd_utils import mmd_from_gram
import grakel


class WeisfeilerLehmanMMD2(DescriptorMMD2):
    def __init__(self, reference_graphs, iterations=3):
        super().__init__(
            reference_graphs,
            LinearKernel(
                WeisfeilerLehmanDescriptor(
                    iterations=iterations, use_node_labels=False
                )
            ),
            variant="biased",
        )


def grakel_wl_mmd(
    reference_graphs, test_graphs, is_parallel=False, iterations=3
):
    grakel_kernel = grakel.WeisfeilerLehman(n_iter=iterations)
    all_graphs = reference_graphs + test_graphs
    for g in all_graphs:
        for node in g.nodes():
            g.nodes[node]["degree"] = g.degree(node)

    all_graphs = grakel.graph_from_networkx(
        all_graphs, node_labels_tag="degree"
    )
    gram_matrix = grakel_kernel.fit_transform(all_graphs)
    ref_vs_ref = gram_matrix[: len(reference_graphs), : len(reference_graphs)]
    ref_vs_gen = gram_matrix[: len(reference_graphs), len(reference_graphs) :]
    gen_vs_gen = gram_matrix[len(reference_graphs) :, len(reference_graphs) :]
    return mmd_from_gram(ref_vs_ref, gen_vs_gen, ref_vs_gen, variant="biased")


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
        baseline_method = lambda ref, pred: orbit_stats_all(
            ref, pred, orca_executable
        )  # noqa

    mmd = mmd_cls(planar)
    assert np.isclose(mmd.compute(sbm), baseline_method(planar, sbm)), mmd_cls
    mmd = mmd_cls(planar[:64])
    assert np.isclose(
        mmd.compute(planar[64:]), baseline_method(planar[:64], planar[64:])
    )


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
        statistic=stat,
        kernel="gaussian_rbf",
        sigma="range",
        orca_path=orca_executable,
    )
    baseline_results, _ = baseline_eval.evaluate(planar, sbm)
    assert len(baseline_results) == 1
    our_eval = mmd_cls(planar)
    assert np.isclose(our_eval.compute(sbm), list(baseline_results.values())[0])


@pytest.mark.parametrize(
    "mmd_cls",
    [
        LinearOrbitMMD2,
        LinearClusteringMMD2,
        LinearDegreeMMD2,
        LinearSpectralMMD2,
    ],
)
def test_linear_gran_mmd_smoke(datasets, mmd_cls):
    planar, sbm = datasets
    planar, sbm = list(planar.to_nx()), list(sbm.to_nx())

    mmd = mmd_cls(planar)
    result = mmd.compute(sbm)
    assert isinstance(result, float)


def test_warn_orbit_self_loops():
    g = nx.Graph()
    g.add_node(0)
    g.add_edge(0, 0)
    with pytest.warns(UserWarning):
        mmd = GRANOrbitMMD2([g])
        mmd.compute([g])


def test_rbf_divide_by_zero():
    g = nx.Graph()
    g.add_node(0)

    mmd = RBFClusteringMMD2([g])
    assert np.isclose(mmd.compute([g]), 0.0)


@pytest.mark.parametrize(
    "kernel,subsample_size,variant",
    [
        ("degree_linear_kernel", 32, "biased"),
        ("degree_linear_kernel", 40, "umve"),
    ],
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


@pytest.mark.parametrize("subsample_size", [16, 32, 64, 100, 128])
@pytest.mark.parametrize(
    "single_cls,interval_cls",
    [
        (GRANClusteringMMD2, GRANClusteringMMD2Interval),
        (GRANDegreeMMD2, GRANDegreeMMD2Interval),
        (GRANOrbitMMD2, GRANOrbitMMD2Interval),
        (GRANSpectralMMD2, GRANSpectralMMD2Interval),
        (RBFClusteringMMD2, RBFClusteringMMD2Interval),
        (RBFDegreeMMD2, RBFDegreeMMD2Interval),
        (RBFOrbitMMD2, RBFOrbitMMD2Interval),
        (RBFSpectralMMD2, RBFSpectralMMD2Interval),
        (LinearOrbitMMD2, LinearOrbitMMD2Interval),
        (LinearClusteringMMD2, LinearClusteringMMD2Interval),
        (LinearDegreeMMD2, LinearDegreeMMD2Interval),
        (LinearSpectralMMD2, LinearSpectralMMD2Interval),
    ],
)
def test_concrete_uncertainty(
    datasets, subsample_size, single_cls, interval_cls
):
    planar, sbm = datasets
    planar, sbm = list(planar.to_nx()), list(sbm.to_nx())

    assert subsample_size <= len(planar) and subsample_size <= len(sbm)

    rng = np.random.default_rng(1)

    assert (
        issubclass(single_cls, DescriptorMMD2)
        and issubclass(interval_cls, DescriptorMMD2Interval)
    ) or (
        issubclass(single_cls, MaxDescriptorMMD2)
        and issubclass(interval_cls, MaxDescriptorMMD2Interval)
    )

    interval_mmd = interval_cls(planar)
    interval = interval_mmd.compute(sbm, subsample_size=subsample_size)
    assert isinstance(interval, MMDInterval)

    num_in_bounds = 0
    num_total = 10

    for _ in range(num_total):
        planar_idxs = rng.choice(
            len(planar), size=subsample_size, replace=False
        )
        sbm_idxs = rng.choice(len(sbm), size=subsample_size, replace=False)
        planar_samples = [planar[int(idx)] for idx in planar_idxs]
        sbm_samples = [sbm[int(idx)] for idx in sbm_idxs]

        single_mmd = single_cls(planar_samples)
        single_estimate = single_mmd.compute(sbm_samples)
        if interval.low <= single_estimate <= interval.high:
            num_in_bounds += 1

    assert num_in_bounds / num_total >= 0.7


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
    metric = max_mmd.compute(planar.to_nx())
    assert isinstance(metric, float)

    unpooled_mmd = DescriptorMMD2(sbm.to_nx(), kernel, variant)
    metric_arr = unpooled_mmd.compute(planar.to_nx())
    assert np.isclose(metric, np.max(metric_arr))


@pytest.mark.skip
@pytest.mark.parametrize(
    "mmd_cls,baseline_method",
    [
        (GRANSpectralMMD2, spectral_stats),
        (GRANOrbitMMD2, orbit_stats_all),
        (GRANClusteringMMD2, clustering_stats),
        (GRANDegreeMMD2, degree_stats),
        (WeisfeilerLehmanMMD2, grakel_wl_mmd),
    ],
)
@pytest.mark.parametrize("parallel_baseline", [True, False])
def test_measure_runtime(
    mmd_cls, baseline_method, orca_executable, runtime_stats, parallel_baseline
):
    if parallel_baseline and (
        mmd_cls is GRANOrbitMMD2 or mmd_cls is WeisfeilerLehmanMMD2
    ):
        pytest.skip("Orbit and WL don't have parallel baselines")

    ds1 = ProceduralPlanarGraphDataset("ds1", 1024, seed=42)
    ds2 = ProceduralPlanarGraphDataset("ds2", 1024, seed=42)
    ds1, ds2 = list(ds1.to_nx()), list(ds2.to_nx())

    if baseline_method is orbit_stats_all:
        patched_baseline_method = lambda ref, pred: orbit_stats_all(  # noqa: E731
            ref, pred, orca_executable
        )  # noqa
    else:
        patched_baseline_method = lambda x, y: baseline_method(  # noqa: E731
            x, y, is_parallel=parallel_baseline
        )  # noqa

    # Get JIT compilation out of the way
    mmd = mmd_cls([ds1[0]])
    mmd.compute([ds2[0]])
    del mmd

    for _ in range(1):
        t0 = time.time()
        mmd = mmd_cls(ds1)
        our_estimate = mmd.compute(ds2)
        t1 = time.time()
        runtime_stats[mmd_cls.__name__]["ours"].append(t1 - t0)

        t0 = time.time()
        baseline_estimate = patched_baseline_method(ds1, ds2)
        t1 = time.time()
        runtime_stats[mmd_cls.__name__][
            "baseline_parallel" if parallel_baseline else "baseline"
        ].append(t1 - t0)

        assert np.isclose(our_estimate, baseline_estimate)
