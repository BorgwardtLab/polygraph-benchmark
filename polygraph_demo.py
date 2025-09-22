#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""polygraph_demo.py

In this file, we aim to demonstrate some of the features of the polygraph library.

"""

import os
from typing import List

import networkx as nx
from appdirs import user_cache_dir
from loguru import logger

import polygraph
from polygraph.datasets import ProceduralPlanarGraphDataset
from polygraph.metrics import (
    VUN,
    GaussianTVMMD2Benchmark,
    RBFMMD2Benchmark,
    StandardPGS,
)


def _sample_generated_graphs(n: int, num_nodes: int = 64, start_seed: int = 0) -> List[nx.Graph]:
    """Create a small set of Erdos-Renyi graphs as a stand-in for a generator."""
    return [nx.erdos_renyi_graph(num_nodes, 0.1, seed=i + start_seed) for i in range(n)]

def data_location():
    cache_dir = user_cache_dir(f"polygraph-{polygraph.__version__}", "MPIB-MLSB")
    logger.info(f"PolyGraph cache is typically located at: {cache_dir}")
    logger.info("It can be changed by setting the POLYGRAPH_CACHE_DIR environment variable.")
    logger.info("Current value: ", os.environ.get("POLYGRAPH_CACHE_DIR"))

def get_example_datasets():
    """
    Create a small set of Erdos-Renyi graphs as a stand-in for a generator and a reference dataset.
    """

    reference_ds = ProceduralPlanarGraphDataset("val", num_graphs=32).to_nx()
    generated = _sample_generated_graphs(32)
    logger.info(f"Reference graphs: {len(reference_ds)} | Generated graphs: {len(generated)}")
    return reference_ds, generated

def calculate_gtv_mmd(reference, generated):
    """
    Calculate the GTV pseudokernel MMD between a reference dataset and a generated dataset.
    """
    logger.info("GaussianTV MMD² Benchmark")
    gtv = GaussianTVMMD2Benchmark(reference)
    logger.info(f"Computed Gaussian TV pseudokernel MMD²: {gtv.compute(generated)}")

def calculate_rbf_mmd(reference, generated):
    """
    Calculate the RBF MMD between a reference dataset and a generated dataset.
    """
    logger.info("RBF MMD² Benchmark")
    rbf = RBFMMD2Benchmark(reference)
    logger.info(f"Computed RBF MMD²: {rbf.compute(generated)}")


def calculate_pgs(reference, generated):
    """
    Calculate the PolyGraphScore between a reference dataset and a generated dataset.
    """
    logger.info("PolyGraphScore (StandardPGS)")
    pgs = StandardPGS(reference)
    logger.info(f"Computed PolyGraphScore: {pgs.compute(generated)}")

def calculate_vun(reference, generated):
    """
    Calculate the VUN between a reference dataset and a generated dataset.
    """
    ds = ProceduralPlanarGraphDataset("val", num_graphs=1)
    validity_fn = ds.is_valid if reference is not None else None
    logger.info("VUN")
    vun = VUN(reference, validity_fn=validity_fn)
    logger.info(f"Computed VUN: {vun.compute(generated)}")


def main():
    logger.info("=== PolyGraph Demo ===")

    # Data location-related information
    data_location()
    reference, generated = get_example_datasets()

    calculate_gtv_mmd(reference, generated)
    calculate_rbf_mmd(reference, generated)
    calculate_pgs(reference, generated)
    calculate_vun(reference, generated)
    
    logger.success("=== PolyGraph Demo End ===")

if __name__ == "__main__":
    main()
