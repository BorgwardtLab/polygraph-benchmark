import pytest
import numpy as np

from polygraph.utils.descriptors.molecule_descriptors import (
    TopoChemicalDescriptor,
    FingerprintDescriptor,
    LipinskiDescriptor,
    ChemNetDescriptor,
    MolCLRDescriptor,
)
from polygraph.metrics.base import DescriptorMMD2, PolyGraphDiscrepancy
from polygraph.metrics.molecule_pgd import MoleculePGD, MoleculePGDInterval
from polygraph.utils.kernels import LinearKernel

from rdkit.Chem import AllChem

smiles_a = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC1(C)SC2C(NC(=O)C2=O)C1(C)C(=O)N",
    "C1C(=O)N(C2=CC=CC=C12)C3=CC=C(C=C3)C(F)(F)F",
    "CCCCCCOc1ccc(C(=O)C=Cc2c(C=Cc3ccc(OC)cc3)cc(OC)cc2OC)cc1",
    "O=C(Nc1nc(-c2ccc(Cl)s2)cs1)c1ccncc1",
    "COc1nc(N(C)C)ncc1-n1nc2c(c1C(C)C)C(c1ccc(C#N)c(F)c1)N(c1c[nH]c(=O)c(Cl)c1)C2=O",
    "Cc1ncc([N+](=O)[O-])n1CC(=O)Nc1ccccc1",
    "CCOC(=O)N=C(NC(C)C)c1ccc(-c2ccc(-c3ccc(C(=NC(=O)OCC)NC(C)C)cc3)o2)cc1",
    "COCC(=O)N1CCC(C2CC(C(F)(F)F)n3nc(C)cc3N2)CC1",
    "Fc1ccc(C(OCCN2C3CCC2CC(Cc2ccccc2)C3)c2ccc(F)cc2)cc1",
    "CC(C)C1CCN(C(=O)C2CCC(=O)N(C3CCCCCC3)C2)CC1",
    "CCc1c2c(n(C)c1C)CCCC2=NOC(=O)Nc1ccc(C(C)=O)cc1",
    "Cc1cc2c(-c3ccc(S(=O)(=O)NCCO)cc3)ccnc2[nH]1",
    "CC(C)N1CCN(C(=O)c2ccc(Oc3ccc(F)cc3)nc2)CC1",
    "O=C(Nc1ccc(Cl)c(O)c1)Nc1ccc(Cl)c(Cl)c1",
    "O=C1NC(=NN=CC(O)C(O)C(O)C(O)CO)NC1=Cc1ccfo1",
    "O=C(NCCO)c1c(O)c2ncc(Cc3ccc(F)cc3)cc2[nH]c1=O",
    "NC(=O)C(=O)C(Cc1ccccc1)NC(=O)C1CCN(C(=O)C=Cc2ccncc2)CC1",
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
    "Cc1c2ccnc(C(=O)NCCN(C)C)c2cc2c3cc(OC(=O)CCCCC(=O)O)ccc3n(C)c12",
    "CC1(C)SC2C(NC(=O)C2=O)C1(C)C(=O)N",
    "C1C(=O)N(C2=CC=CC=C12)C3=CC=C(C=C3)C(F)(F)F",
    "CCCCCCOc1ccc(C(=O)C=Cc2c(C=Cc3ccc(OC)cc3)cc(OC)cc2OC)cc1",
    "O=C(Nc1nc(-c2ccc(Cl)s2)cs1)c1ccncc1",
    "COc1nc(N(C)C)ncc1-n1nc2c(c1C(C)C)C(c1ccc(C#N)c(F)c1)N(c1c[nH]c(=O)c(Cl)c1)C2=O",
    "Cc1ncc([N+](=O)[O-])n1CC(=O)Nc1ccccc1",
    "CCOC(=O)N=C(NC(C)C)c1ccc(-c2ccc(-c3ccc(C(=NC(=O)OCC)NC(C)C)cc3)o2)cc1",
    "COCC(=O)N1CCC(C2CC(C(F)(F)F)n3nc(C)cc3N2)CC1",
    "Fc1ccc(C(OCCN2C3CCC2CC(Cc2ccccc2)C3)c2ccc(F)cc2)cc1",
    "O=C(NCC1CCCO1)c1ccc2c(=O)n(-c3ccccc3)c(=S)[nH]c2c1",
    "CCNc1nc(C#N)nc(N2CCCCC2)n1",
]

mols_a = [AllChem.MolFromSmiles(smiles) for smiles in smiles_a]
mols_a = list(filter(lambda x: x is not None, mols_a))
mols_b = [AllChem.MolFromSmiles(smiles) for smiles in smiles_b]
mols_b = list(filter(lambda x: x is not None, mols_b))
num_mols = min(len(mols_a), len(mols_b))
mols_a = mols_a[:num_mols]
mols_b = mols_b[:num_mols]


descriptors = {
    "topo_chemical": TopoChemicalDescriptor(),
    "fingerprint": FingerprintDescriptor(),
    "lipinski": LipinskiDescriptor(),
    "chemnet": ChemNetDescriptor(),
    "molclr": MolCLRDescriptor(),
}


@pytest.mark.parametrize(
    "descriptor",
    [
        TopoChemicalDescriptor(),
        FingerprintDescriptor(),
        LipinskiDescriptor(),
        ChemNetDescriptor(),
        MolCLRDescriptor(),
    ],
)
def test_molecule_descriptors(descriptor):
    mols = mols_a

    features = descriptor(mols)
    assert isinstance(features, np.ndarray)
    assert features.ndim == 2
    assert len(features) == len(mols)


def test_smoke_polygraphscore():
    metric = PolyGraphDiscrepancy(mols_a, descriptors)
    metric.compute(mols_b)


@pytest.mark.parametrize(
    "descriptor",
    [
        TopoChemicalDescriptor(),
        FingerprintDescriptor(),
        LipinskiDescriptor(),
        ChemNetDescriptor(),
        MolCLRDescriptor(),
    ],
)
@pytest.mark.parametrize("kernel", [LinearKernel])
def test_smoke_mmd2(descriptor, kernel):
    metric = DescriptorMMD2(mols_a, kernel=kernel(descriptor))
    metric.compute(mols_b)


def test_smoke_molecule_pgd():
    metric = MoleculePGD(mols_a)
    metric.compute(mols_b)

    metric = MoleculePGDInterval(mols_a, subsample_size=8, num_samples=4)
    metric.compute(mols_b)
