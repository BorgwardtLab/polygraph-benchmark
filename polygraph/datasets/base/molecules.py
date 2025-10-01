import torch
from rdkit import Chem
from torch_geometric.data import Data
from typing import Optional

BOND_TYPES = [
    Chem.rdchem.BondType.UNSPECIFIED,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.QUADRUPLE,
    Chem.rdchem.BondType.QUINTUPLE,
    Chem.rdchem.BondType.HEXTUPLE,
    Chem.rdchem.BondType.ONEANDAHALF,
    Chem.rdchem.BondType.TWOANDAHALF,
    Chem.rdchem.BondType.THREEANDAHALF,
    Chem.rdchem.BondType.FOURANDAHALF,
    Chem.rdchem.BondType.FIVEANDAHALF,
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.IONIC,
    Chem.rdchem.BondType.HYDROGEN,
    Chem.rdchem.BondType.THREECENTER,
    Chem.rdchem.BondType.DATIVEONE,
    Chem.rdchem.BondType.DATIVE,
    Chem.rdchem.BondType.DATIVEL,
    Chem.rdchem.BondType.DATIVER,
    Chem.rdchem.BondType.OTHER,
    Chem.rdchem.BondType.ZERO,
]

BOND_STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOATROPCCW,
    Chem.rdchem.BondStereo.STEREOATROPCW,
]

# Generalized atom vocabulary for all molecules
N_UNIQUE_ATOMS = 119
ATOM_TYPES = {
    i: Chem.GetPeriodicTable().GetElementSymbol(i)
    for i in range(1, N_UNIQUE_ATOMS)
}

# Graph attributes for molecular graphs
NODE_ATTRS = [
    "atom_labels",
    "radical_electrons",
    "charges",
]
EDGE_ATTRS = ["bond_types"]


def are_smiles_equivalent(smiles1, smiles2):
    if smiles1 == smiles2:
        return True

    # Convert SMILES to mol objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Check if either conversion failed
    if mol1 is None or mol2 is None:
        return False

    # Convert to canonical SMILES
    canonical_smiles1 = Chem.MolToSmiles(mol1, canonical=True)
    canonical_smiles2 = Chem.MolToSmiles(mol2, canonical=True)

    return canonical_smiles1 == canonical_smiles2


def mol2smiles(mol, canonical: bool = False):
    try:
        Chem.SanitizeMol(mol)
    except ValueError as e:
        print(e, mol)
        return None
    return Chem.MolToSmiles(mol, canonical=canonical)


def smiles_with_explicit_hydrogens(smiles: str, canonical: bool = True) -> str:
    """Convert a SMILES string to a SMILES string with all hydrogens made explicit.

    Args:
        smiles: Input SMILES string
        canonical: Whether to return canonical SMILES

    Returns:
        SMILES string with all hydrogens made explicit
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    return Chem.MolToSmiles(mol, canonical=canonical, allHsExplicit=True)


def graph2molecule(
    node_labels: torch.Tensor,
    edge_index: torch.Tensor,
    bond_types: torch.Tensor,
    charges: Optional[torch.Tensor] = None,
    num_radical_electrons: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
) -> Chem.RWMol:
    assert edge_index.shape[1] == len(bond_types)
    node_idx_to_atom_idx = {}
    current_atom_idx = 0
    mol = Chem.RWMol()
    for node_idx, atom in enumerate(node_labels):
        a = Chem.Atom(ATOM_TYPES[atom.item()])
        mol.AddAtom(a)
        node_idx_to_atom_idx[node_idx] = current_atom_idx

        atom_obj = mol.GetAtomWithIdx(node_idx_to_atom_idx[node_idx])

        if charges is not None:
            atom_obj.SetFormalCharge(charges[node_idx].item())
        if num_radical_electrons is not None:
            atom_obj.SetNumRadicalElectrons(
                num_radical_electrons[node_idx].item()
            )
        atom_obj.SetNoImplicit(True)
        current_atom_idx += 1

    edges_processed = set()
    for i, (bond, bond_type) in enumerate(zip(edge_index.T, bond_types)):
        a, b = bond[0].item(), bond[1].item()
        if (a, b) in edges_processed or (b, a) in edges_processed:
            continue

        mol.AddBond(
            node_idx_to_atom_idx[a],
            node_idx_to_atom_idx[b],
            BOND_TYPES[bond_type],
        )

        edges_processed.add((a, b))

    if pos is not None:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for node_idx, atom_pos in enumerate(pos):
            conf.SetAtomPosition(
                node_idx_to_atom_idx[node_idx], atom_pos.tolist()
            )
        mol.AddConformer(conf)
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=False)

    Chem.SanitizeMol(mol)
    return mol


def molecule2graph(
    mol: Chem.RWMol,
) -> Data:
    """Convert molecule to graph representation.

    Args:
        mol: Input molecule
    """
    mol = Chem.AddHs(mol, addCoords=True)
    Chem.AssignStereochemistryFrom3D(mol)
    Chem.rdmolops.SetBondStereoFromDirections(mol)

    node_labels = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])

    charge_tensor = torch.tensor(
        [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    )
    radical_tensor = torch.tensor(
        [atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()]
    )

    pos = None
    if mol.GetNumConformers() > 0:
        conformer = mol.GetConformer()
        pos = torch.tensor(
            [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
        )

    bonds = list(mol.GetBonds())
    edge_index = torch.tensor(
        [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in bonds]
        + [(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()) for bond in bonds]
    ).T

    bond_types = torch.tensor(
        [BOND_TYPES.index(bond.GetBondType()) for bond in bonds] * 2
    )

    return Data(
        edge_index=edge_index,
        atom_labels=node_labels,
        bond_types=bond_types,
        charges=charge_tensor,
        radical_electrons=radical_tensor,
        pos=pos,
        num_nodes=len(node_labels),
    )


def add_hydrogens_and_stereochemistry(mol: Chem.RWMol):
    mol = Chem.AddHs(mol, addCoords=True)
    Chem.AssignStereochemistryFrom3D(mol)
    Chem.rdmolops.SetBondStereoFromDirections(mol)
    return mol
