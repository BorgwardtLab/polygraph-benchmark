# -*- coding: utf-8 -*-
"""conftest.py

The conftest.py file is used to define fixtures and other configurations for
pytest at the start of the session.
"""

import graph_tool.all as _  # noqa

import subprocess
import urllib.request
from collections import defaultdict

import numpy as np
import pytest
from loguru import logger
import sys

from graph_gen_gym.datasets import (
    PlanarGraphDataset,
    SBMGraphDataset,
)
from graph_gen_gym.utils.graph_descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    OrbitCounts,
)
from graph_gen_gym.utils.kernels import (
    AdaptiveRBFKernel,
    LaplaceKernel,
    LinearKernel,
    RBFKernel,
)

NO_SKIP_OPTION = "--no-skip"
SAMPLE_SIZE_OPTION = "--sample-size"
LOG_LEVEL_OPTION = "--test-log-level"


def pytest_addoption(parser):
    parser.addoption(
        NO_SKIP_OPTION,
        action="store_true",
        default=False,
        help="also run skipped tests",
    )
    parser.addoption(
        SAMPLE_SIZE_OPTION,
        action="store",
        default=5,
        type=int,
        help="number of samples to use in tests",
    )
    parser.addoption(
        LOG_LEVEL_OPTION,
        action="store",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set the logging level",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption(NO_SKIP_OPTION):
        for test in items:
            test.own_markers = [
                marker
                for marker in test.own_markers
                if marker.name not in ("skip", "skipif")
            ]


@pytest.fixture(scope="session", autouse=True)
def setup_logging(request):
    logger.remove()  # Remove existing handlers
    log_level = request.config.getoption(LOG_LEVEL_OPTION)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
    )

    logger.add(
        "./logs/tests/test_session.log",
        rotation="1 day",
        retention="1 week",
        level=log_level,
        encoding="utf-8",
    )
    logger.success(f"Logging level set to {log_level}")
    logger.success(f"Sample size set to {request.config.getoption(SAMPLE_SIZE_OPTION)}")
    logger.success(
        f"Skipping tests operator set to {request.config.getoption(NO_SKIP_OPTION)}"
    )
    return logger


@pytest.fixture(scope="session", autouse=True)
def sample_size(request):
    return request.config.getoption(SAMPLE_SIZE_OPTION)


@pytest.fixture(scope="session", autouse=True)
def orca_executable(tmpdir_factory):
    orca_path = tmpdir_factory.mktemp("orca")
    cpp_path = orca_path.join("orca.cpp")
    h_path = orca_path.join("orca.h")
    executable_path = orca_path.join("orca")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/cvignac/DiGress/refs/heads/main/src/analysis/orca/orca.cpp",
        cpp_path,
    )
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/cvignac/DiGress/refs/heads/main/src/analysis/orca/orca.h",
        h_path,
    )
    subprocess.run(
        ["g++", "-O2", "-std=c++11", "-o", str(executable_path), str(cpp_path)]
    )
    return executable_path


@pytest.fixture(scope="session", autouse=True)
def datasets():
    planar = PlanarGraphDataset("train")
    sbm = SBMGraphDataset("train")
    return planar, sbm


@pytest.fixture(scope="session", autouse=True)
def orbit_rbf_kernel():
    return RBFKernel(OrbitCounts(), bw=np.linspace(0.01, 20, 100))


@pytest.fixture(scope="session", autouse=True)
def degree_linear_kernel():
    return LinearKernel(DegreeHistogram(max_degree=200))


@pytest.fixture(scope="session", autouse=True)
def degree_rbf_kernel():
    return RBFKernel(DegreeHistogram(max_degree=200), bw=np.linspace(0.01, 20, 10))


@pytest.fixture(scope="session", autouse=True)
def degree_adaptive_rbf_kernel():
    return AdaptiveRBFKernel(
        DegreeHistogram(max_degree=200),
        bw=np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]),
    )


@pytest.fixture(scope="session", autouse=True)
def clustering_laplace_kernel():
    return LaplaceKernel(ClusteringHistogram(bins=100), lbd=np.linspace(0.01, 20, 100))


@pytest.fixture(scope="session")
def runtime_stats(request):
    stats = defaultdict(lambda: {"ours": [], "baseline_parallel": [], "baseline": []})
    yield stats

    # Get capsys through the config
    capsys = request.node.config.pluginmanager.get_plugin("capturemanager")
    with capsys.global_and_fixture_disabled():
        print("\n" + "="*50)
        print("Runtime Comparisons")
        print("="*50)
        for test_name, times in stats.items():
            our_avg = np.mean(times["ours"])
            baseline_parallel_avg = np.mean(times["baseline_parallel"])
            baseline_avg = np.mean(times["baseline"])
            speedup_parallel = baseline_parallel_avg / our_avg
            speedup_sequential = baseline_avg / our_avg
            print(f"\n{test_name}:")
            print(f"  Our implementation: {our_avg:.4f}s (avg)")
            print(f"  Baseline (parallel): {baseline_parallel_avg:.4f}s (avg)")
            print(f"  Baseline (sequential): {baseline_avg:.4f}s (avg)")
            print(f"  Speedup (parallel): {speedup_parallel:.2f}x")
            print(f"  Speedup (sequential): {speedup_sequential:.2f}x")
        print("\n" + "="*50)
