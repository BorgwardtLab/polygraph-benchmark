# -*- coding: utf-8 -*-
"""conftest.py

The conftest.py file is used to define fixtures and other configurations for
pytest at the start of the session.
"""
import graph_tool.all as _   # noqa

import subprocess
import urllib.request

import numpy as np
import pytest

from graph_gen_gym.datasets.spectre import PlanarGraphDataset, SBMGraphDataset
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


def pytest_addoption(parser):
    parser.addoption(
        NO_SKIP_OPTION,
        action="store_true",
        default=False,
        help="also run skipped tests",
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
