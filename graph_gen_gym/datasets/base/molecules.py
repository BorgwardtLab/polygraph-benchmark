from rdkit import Chem

ALLOWED_BONDS = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3, 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
BOND_DICT = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

def are_smiles_equivalent(smiles1, smiles2):
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


def build_molecule(node_labels, edge_index, edge_labels, atom_decoder, explicit_hydrogens=None, charges=None, num_radical_electrons=None):
    assert edge_index.shape[1] == len(edge_labels)
    assert edge_labels.ndim == 1

    node_idx_to_atom_idx = {}
    current_atom_idx = 0
    mol = Chem.RWMol()
    for node_idx, atom in enumerate(node_labels):
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        node_idx_to_atom_idx[node_idx] = current_atom_idx
        if charges is not None:
            mol.GetAtomWithIdx(node_idx_to_atom_idx[node_idx]).SetFormalCharge(charges[node_idx].item())
        if num_radical_electrons is not None:
            mol.GetAtomWithIdx(node_idx_to_atom_idx[node_idx]).SetNumRadicalElectrons(num_radical_electrons[node_idx].item())
        current_atom_idx += 1
        if explicit_hydrogens is not None:
            num_hydrogens = explicit_hydrogens[node_idx].item()
            for _ in range(num_hydrogens):
                mol.AddAtom(Chem.Atom("H"))
                mol.AddBond(current_atom_idx, node_idx_to_atom_idx[node_idx], Chem.rdchem.BondType.SINGLE)
                current_atom_idx += 1


    added_bonds = set()
    for bond, bond_type in zip(edge_index.T, edge_labels):
        a, b = bond[0].item(), bond[1].item()
        if a != b and (a, b) not in added_bonds:
            added_bonds.add((a, b))
            added_bonds.add((b, a))
            #print(f"Attaching {mol.GetAtomWithIdx(node_idx_to_atom_idx[a]).GetSymbol()} to {mol.GetAtomWithIdx(node_idx_to_atom_idx[b]).GetSymbol()} via {BOND_DICT[bond_type.item()]}")
            mol.AddBond(node_idx_to_atom_idx[a], node_idx_to_atom_idx[b], BOND_DICT[bond_type.item()])
    return mol
