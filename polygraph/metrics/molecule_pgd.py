"""MoleculePGD is a [`PolyGraphDiscrepancy`][polygraph.metrics.base.polygraphdiscrepancy.PolyGraphDiscrepancy] metric based on different molecule descriptors.

- [`TopoChemicalDescriptor`][polygraph.utils.descriptors.molecule_descriptors.TopoChemicalDescriptor]: Topological features based on bond structure
- [`FingerprintDescriptor`][polygraph.utils.descriptors.molecule_descriptors.FingerprintDescriptor]: Molecular fingerprints
- [`LipinskiDescriptor`][polygraph.utils.descriptors.molecule_descriptors.LipinskiDescriptor]: Physico-chemical properties
- [`ChemNetDescriptor`][polygraph.utils.descriptors.molecule_descriptors.ChemNetDescriptor]: Random projection of ChemNet embeddings, based on SMILES strings
- [`MolCLRDescriptor`][polygraph.utils.descriptors.molecule_descriptors.MolCLRDescriptor]: Random projection of MolCLR embeddings from a GNN

By default, we use TabPFN for binary classification and evaluate it by data log-likelihood, obtaining a PolyGraphDiscrepancy that provides an estimated lower bound on the Jensen-Shannon
distance between the generated and true graph distribution.

```python
import rdkit.Chem
from polygraph.metrics.molecule_pgd import MoleculePGD

smiles_a = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC1(C)SC2C(NC(=O)C2=O)C1(C)C(=O)N",
    "C1C(=O)N(C2=CC=CC=C12)C3=CC=C(C=C3)C(F)(F)F",
    "CCCCCCOc1ccc(C(=O)C=Cc2c(C=Cc3ccc(OC)cc3)cc(OC)cc2OC)cc1",
    "O=C(Nc1nc(-c2ccc(Cl)s2)cs1)c1ccncc1",
    "COc1nc(N(C)C)ncc1-n1nc2c(c1C(C)C)C(c1ccc(C#N)c(F)c1)N(c1c[nH]c(=O)c(Cl)c1)C2=O",
]
smiles_b = [
    "CC1=C(C=CC=C1)NC2=NC=CC(=N2)NC3=CC=CC=C3C(=O)NC4=CC=CC=N4",
    "CN1CCN(C2=CC3=C(C=C2)N=CN3C)C4=CC=CC=C14",
    "CN(C)CCCN1C2=CC=CC=C2SC3=CC=CC=C31",
    "CC(C)C(C(=O)NCC(C)C)NC(=O)C1=CC=CC=C1C(C)C(C)NC(=O)C2=CN=CC=C2",
    "CN1C(=O)CN=C(C2=CC=CC=C12)C3=CC=CC=C3Cl",
    "O=C(c1cc(-c2ccc(Cl)cc2Cl)n[nH]1)N1CCCC1",
    "COc1cccc(OC)c1C=CC(=O)NC1CCCCC1",
    "O=C1NC(O)CCN1C1OC(CO)C(O)C1O",
]
mols_a = [rdkit.Chem.MolFromSmiles(smiles) for smiles in smiles_a]
mols_b = [rdkit.Chem.MolFromSmiles(smiles) for smiles in smiles_b]
metric = MoleculePGD(mols_a)
print(metric.compute(mols_b))
```
"""

from typing import Collection

import rdkit.Chem

from polygraph.utils.descriptors.molecule_descriptors import (
    TopoChemicalDescriptor,
    FingerprintDescriptor,
    LipinskiDescriptor,
    ChemNetDescriptor,
    MolCLRDescriptor,
)

from polygraph.metrics.base import (
    PolyGraphDiscrepancy,
    PolyGraphDiscrepancyInterval,
)

__all__ = [
    "MoleculePGD",
    "MoleculePGDInterval",
]


class MoleculePGD(PolyGraphDiscrepancy[rdkit.Chem.Mol]):
    """MoleculePGD to compare molecule distributions, combining different molecule descriptors.

    Args:
        reference_molecules: Reference rdkit molecules
    """

    def __init__(self, reference_molecules: Collection[rdkit.Chem.Mol]):
        super().__init__(
            reference_graphs=reference_molecules,
            descriptors={
                "topochemical": TopoChemicalDescriptor(),
                "morgan_fingerprint": FingerprintDescriptor(
                    algorithm="morgan", dim=128
                ),
                "chemnet": ChemNetDescriptor(dim=128),
                "molclr": MolCLRDescriptor(dim=128),
                "lipinski": LipinskiDescriptor(),
            },
            variant="jsd",
            classifier=None,
        )


class MoleculePGDInterval(PolyGraphDiscrepancyInterval[rdkit.Chem.Mol]):
    """Uncertainty quantification for [`MoleculePGD`][polygraph.metrics.molecule_pgd.MoleculePGD].

    Args:
        reference_molecules: Reference rdkit molecules
        subsample_size: Size of each subsample, should be consistent with the number
            of reference and generated molecules passed to [`MoleculePGD`][polygraph.metrics.molecule_pgd.MoleculePGD]
            for point estimates.
        num_samples: Number of samples to draw for uncertainty quantification.
    """

    def __init__(
        self,
        reference_molecules: Collection[rdkit.Chem.Mol],
        subsample_size: int,
        num_samples: int = 10,
    ):
        super().__init__(
            reference_graphs=reference_molecules,
            descriptors={
                "topochemical": TopoChemicalDescriptor(),
                "morgan_fingerprint": FingerprintDescriptor(
                    algorithm="morgan", dim=128
                ),
                "chemnet": ChemNetDescriptor(dim=128),
                "molclr": MolCLRDescriptor(dim=128),
                "lipinski": LipinskiDescriptor(),
            },
            subsample_size=subsample_size,
            num_samples=num_samples,
            variant="jsd",
            classifier=None,
        )
