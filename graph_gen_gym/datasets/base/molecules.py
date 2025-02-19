import torch
from rdkit import Chem
from torch_geometric.data import Data

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
    i: Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1, N_UNIQUE_ATOMS)
}

# Graph attributes for molecular graphs
NODE_ATTRS = [
    "atom_labels",
    "radical_electrons",
    "charges",
]
EDGE_ATTRS = ["bond_types", "stereo_types"]


def get_canonical_bond_stereo(bond: Chem.Bond) -> int:
    """Get canonical (order-invariant) stereo representation for a bond.

    This function takes a bond and returns its stereo configuration in a canonical way that is
    independent of atom ordering. For double bonds with E/Z stereochemistry, it ensures that
    the stereo designation (E or Z) is consistent regardless of which end of the bond is considered
    the "beginning". If the begin atom index is greater than the end atom index, it swaps the
    E/Z designation to maintain consistency.

    Args:
        bond: An RDKit bond object

    Returns:
        The canonical bond stereo designation as a Chem.BondStereo enum value
    """
    stereo = bond.GetStereo()
    if stereo == Chem.BondStereo.STEREONONE:
        return Chem.BondStereo.STEREONONE

    # Get the indices in a canonical order
    begin_idx = bond.GetBeginAtomIdx()
    end_idx = bond.GetEndAtomIdx()
    if begin_idx > end_idx:
        # Swap the interpretation if atoms are in reverse order
        if stereo == Chem.BondStereo.STEREOZ:
            return Chem.BondStereo.STEREOE
        elif stereo == Chem.BondStereo.STEREOE:
            return Chem.BondStereo.STEREOZ
    return stereo


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


def graph2molecule(
    node_labels: torch.Tensor,
    edge_index: torch.Tensor,
    bond_types: torch.Tensor,
    stereo_types: torch.Tensor,
    charges: torch.Tensor | None = None,
    num_radical_electrons: torch.Tensor | None = None,
    pos: torch.Tensor | None = None,
) -> Chem.RWMol:
    assert edge_index.shape[1] == len(bond_types)
    assert bond_types.shape[0] == stereo_types.shape[0]
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
            atom_obj.SetNumRadicalElectrons(num_radical_electrons[node_idx].item())
        atom_obj.SetNoImplicit(True)
        current_atom_idx += 1

    edges_processed = set()
    for i, (bond, bond_type, stereo_type) in enumerate(
        zip(edge_index.T, bond_types, stereo_types)
    ):
        a, b = bond[0].item(), bond[1].item()
        if (a, b) in edges_processed or (b, a) in edges_processed:
            continue

        new_bond = (
            mol.AddBond(
                node_idx_to_atom_idx[a],
                node_idx_to_atom_idx[b],
                BOND_TYPES[bond_type],
            )
            - 1
        )
        bond_obj = mol.GetBondWithIdx(new_bond)

        begin_atom = bond_obj.GetBeginAtom()
        end_atom = bond_obj.GetEndAtom()

        begin_neighbors = [
            n.GetIdx()
            for n in begin_atom.GetNeighbors()
            if n.GetIdx() != end_atom.GetIdx()
        ]
        end_neighbors = [
            n.GetIdx()
            for n in end_atom.GetNeighbors()
            if n.GetIdx() != begin_atom.GetIdx()
        ]

        if len(begin_neighbors) > 0 and len(end_neighbors) > 0:
            bond_obj.SetStereoAtoms(begin_neighbors[0], end_neighbors[0])

            if stereo_type in [
                Chem.BondStereo.STEREOCIS,
                Chem.BondStereo.STEREOTRANS,
            ]:
                if len(begin_neighbors) > 0 and len(end_neighbors) > 0:
                    bond_obj.SetStereo(BOND_STEREO_TYPES[stereo_type])
            else:
                bond_obj.SetStereo(BOND_STEREO_TYPES[stereo_type])

        edges_processed.add((a, b))

    if pos is not None:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for node_idx, atom_pos in enumerate(pos):
            conf.SetAtomPosition(node_idx_to_atom_idx[node_idx], atom_pos.tolist())
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

    charge_tensor = torch.tensor([atom.GetFormalCharge() for atom in mol.GetAtoms()])
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

    edge_attrs = torch.tensor(
        [BOND_TYPES.index(bond.GetBondType()) for bond in bonds] * 2
    )

    stereo_attrs = []
    for bond in bonds:
        stereo_attrs.append(BOND_STEREO_TYPES.index(get_canonical_bond_stereo(bond)))

    stereo_attrs = stereo_attrs * 2
    stereo_attrs = torch.tensor(stereo_attrs)

    # Add edge labels to edge_attrs
    edge_attr = torch.cat(
        [edge_attrs.unsqueeze(-1), stereo_attrs.unsqueeze(-1)], dim=-1
    )
    return Data(
        edge_index=edge_index,
        atom_labels=node_labels,
        bond_types=edge_attr[:, 0],
        stereo_types=edge_attr[:, 1],
        charges=charge_tensor,
        radical_electrons=radical_tensor,
        pos=pos,
        num_nodes=len(node_labels),
    )


def add_hydrogens_and_stereochemistry(mol: Chem.RWMol):
    mol = Chem.AddHs(mol, addCoords=True)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol
