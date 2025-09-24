#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""polygraph_demo.py

In this file, we aim to demonstrate some of the features of the polygraph library.

"""

import os
from typing import List
import warnings

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


REF_SMILES = [
    "Nc1ncnc2c1ncn2C1OC(CO)CC1F",
    "C=CCc1c(OC(C)=O)c2cccnc2n(-c2ccccc2)c1=O",
    "COc1ccc(Cc2cnc(N)nc2N)cc1OC",
    "COc1cc(O)cc(CCc2ccc(O)c(OC)c2)c1",
    "COc1c(C)cnc(CSc2nccn2C)c1C",
    "O=c1cc(-c2ccncc2)nc(-c2cccnc2)[nH]1",
    "O=c1c2ccccc2oc2nc3n(c(=O)c12)CCCS3",
    "O=c1c2cc(Cl)ccc2oc2nc3n(c(=O)c12)CCCS3",
]
GEN_SMILES = [
    "O=C(NC1CCN(C(=O)C2CC2)CC1)c1ccc(F)cc1",
    "NC(=O)c1cccc2[nH]c(-c3ccc(O)cc3)nc12",
    "CC(C)CCNC(=O)c1c[nH]c2ccccc2c1=O",
    "CCOc1ccc2[nH]cc(C(=O)NCc3cccnc3)c(=O)c2c1",
    "O=C(NCc1ccccc1)c1c[nH]c2c(F)cccc2c1=O",
    "CC(C)c1cccc(C(C)C)c1NCc1ccccn1",
    "CC1CCC(NC(=O)c2cc3ccccc3o2)CC1",
    "COc1ccc2[nH]cc(CCNC(=O)c3ccco3)c2c1",
]

logger.disable("polygraph")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _sample_generated_graphs(
    n: int, num_nodes: int = 64, start_seed: int = 0
) -> List[nx.Graph]:
    """Create a small set of Erdos-Renyi graphs as a stand-in for a generator."""
    return [
        nx.erdos_renyi_graph(num_nodes, 0.1, seed=i + start_seed)
        for i in range(n)
    ]


def data_location():
    cache_dir = user_cache_dir(f"polygraph-{polygraph.__version__}", "ANON_ORG")
    print(f"PolyGraph cache is typically located at: {cache_dir}")
    print(
        "It can be changed by setting the POLYGRAPH_CACHE_DIR environment variable."
    )
    print("Current value: ", os.environ.get("POLYGRAPH_CACHE_DIR"))


def get_example_datasets():
    """
    Create a small set of Erdos-Renyi graphs as a stand-in for a generator and a reference dataset.
    """

    reference_ds = list(
        ProceduralPlanarGraphDataset("val", num_graphs=32).to_nx()
    )
    generated = _sample_generated_graphs(32)
    print(
        f"Reference graphs: {len(reference_ds)} | Generated graphs: {len(generated)}"
    )
    return reference_ds, generated


def calculate_gtv_mmd(reference, generated):
    """
    Calculate the GTV pseudokernel MMD between a reference dataset and a generated dataset.
    """
    print("GaussianTV MMD² Benchmark")
    gtv = GaussianTVMMD2Benchmark(reference)
    result = gtv.compute(generated)
    print("Computed Gaussian TV pseudokernel MMD²:")
    for metric, score in result.items():
        print(f"  {metric.capitalize()}: {score:.6f}")
    print()


def calculate_rbf_mmd(reference, generated):
    """
    Calculate the RBF MMD between a reference dataset and a generated dataset.
    """
    print("RBF MMD² Benchmark")
    rbf = RBFMMD2Benchmark(reference)
    result = rbf.compute(generated)
    print("Computed RBF MMD²:")
    for metric, score in result.items():
        print(f"  {metric.capitalize()}: {score:.6f}")
    print()


def calculate_pgs(reference, generated):
    """
    Calculate the standard PolyGraphScore between a reference dataset and a generated dataset.
    """
    print("PolyGraphScore (StandardPGS)")
    pgs = StandardPGS(reference)
    result = pgs.compute(generated)
    print(f"Overall PGS: {result['polygraphscore']:.6f}")
    print(f"Most powerful descriptor: {result['polygraphscore_descriptor']}")
    print("Subscores:")
    for metric, score in result["subscores"].items():
        print(f"  {metric.capitalize()}: {score:.6f}")
    print()


def calculate_molecule_pgs(ref_smiles, gen_smiles):
    """
    Calculate the PolyGraphScore between a reference dataset of molecules and a generated dataset of molecules.
    """
    from polygraph.metrics.molecule_pgs import MoleculePGS
    import rdkit.Chem

    ref_mols = [rdkit.Chem.MolFromSmiles(smiles) for smiles in ref_smiles]
    gen_mols = [rdkit.Chem.MolFromSmiles(smiles) for smiles in gen_smiles]

    print(
        f"PolyGraphScore (MoleculePGS) between {len(ref_mols)} reference and {len(gen_mols)} generated molecules:"
    )
    pgs = MoleculePGS(ref_mols)
    result = pgs.compute(gen_mols)
    print(f"Overall MoleculePGS: {result['polygraphscore']:.6f}")
    print(f"Most powerful descriptor: {result['polygraphscore_descriptor']}")
    print("Subscores:")
    for metric, score in result["subscores"].items():
        print(f"  {metric.replace('_', ' ').title()}: {score:.6f}")
    print()


def calculate_vun(reference, generated):
    """
    Calculate the VUN between a reference dataset and a generated dataset.
    """
    ds = ProceduralPlanarGraphDataset("val", num_graphs=1)
    validity_fn = ds.is_valid if reference is not None else None
    print("VUN")
    vun = VUN(reference, validity_fn=validity_fn)
    result = vun.compute(generated)
    print("Computed VUN:")
    for metric, score in result.items():
        print(f"  {metric.replace('_', ' ').title()}: {score:.6f}")
    print()


def main():
    print("=== PolyGraph Demo ===")

    # Data location-related information
    data_location()
    reference, generated = get_example_datasets()
    print()

    calculate_gtv_mmd(reference, generated)
    calculate_rbf_mmd(reference, generated)
    calculate_pgs(reference, generated)
    calculate_vun(reference, generated)
    calculate_molecule_pgs(REF_SMILES, GEN_SMILES)

    print("=== PolyGraph Demo End ===")


if __name__ == "__main__":
    main()
