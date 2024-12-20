from typing import Literal
import torch
from torch_geometric.data import Data
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
            mol.AddBond(node_idx_to_atom_idx[a], node_idx_to_atom_idx[b], BOND_DICT[bond_type.item()])
    return mol


class AddHydrogenTransform:
    def __init__(self, add_hydrogen: Literal["all", "explicit"] = "all", hydrogen_label: int = 0, single_bond_label: int = 0):
        self._variant = add_hydrogen
        self._hydrogen_label = hydrogen_label
        self._single_bond_label = single_bond_label

    def __call__(self, graph: Data):
        if not (hasattr(graph, "atom_labels") and hasattr(graph, "bond_labels") and hasattr(graph, "explicit_hydrogens") and hasattr(graph, "implicit_hydrogens") and hasattr(graph, "radical_electrons") and hasattr(graph, "charges")):
            raise ValueError("Molecular graph must have attributes: atom_labels, bond_labels, implicit_hydrogens, explicit_hydrogens, radical_electrons, charges.")

        if self._variant == "explicit":
            hydrogens_to_add = graph.explicit_hydrogens
        elif self._variant == "all":
            hydrogens_to_add = graph.explicit_hydrogens + graph.implicit_hydrogens
        else:
            raise NotImplementedError(f"Only valid modes are 'all' and 'explicit', got {self._variant}")

        total_nodes_to_add = hydrogens_to_add.sum().item()
        total_edges_to_add = 2 * total_nodes_to_add

        new_edge_index = torch.zeros((2, graph.edge_index.shape[1] + total_edges_to_add), dtype=graph.edge_index.dtype, device=graph.edge_index.device)
        new_edge_index[:, :graph.edge_index.shape[1]] = graph.edge_index

        new_atom_labels = torch.full((graph.num_nodes + total_nodes_to_add,), self._hydrogen_label, dtype=graph.atom_labels.dtype, device=graph.atom_labels.device)
        new_atom_labels[:len(graph.atom_labels)] = graph.atom_labels

        new_bond_labels = torch.full((len(graph.bond_labels) + total_edges_to_add,), self._single_bond_label, dtype=graph.atom_labels.dtype, device=graph.atom_labels.device)
        new_bond_labels[:len(graph.bond_labels)] = graph.bond_labels

        new_implicit_hydrogens = torch.zeros(len(graph.implicit_hydrogens) + total_nodes_to_add, dtype=graph.implicit_hydrogens.dtype, device=graph.implicit_hydrogens.device)
        if self._variant != "all":
            # only keep implicit hydrogens if we do not add them as nodes
            new_implicit_hydrogens[:len(graph.implicit_hydrogens)] = graph.implicit_hydrogens

        new_explicit_hydrogens = torch.zeros(len(graph.explicit_hydrogens) + total_nodes_to_add, dtype=graph.explicit_hydrogens.dtype, device=graph.explicit_hydrogens.device)
        # All explicit hydrogens are now part of the graph, so we do not keep the old explicit hydrogens

        new_radical_electrons = torch.zeros(len(graph.radical_electrons) + total_nodes_to_add, dtype=graph.radical_electrons.dtype, device=graph.radical_electrons.device)
        new_radical_electrons[:len(graph.radical_electrons)] = graph.radical_electrons

        new_charges = torch.zeros(len(graph.charges) + total_nodes_to_add, dtype=graph.charges.dtype, device=graph.charges.device)
        new_charges[:len(graph.charges)] = graph.charges

        current_edge_idx = graph.edge_index.shape[1]
        current_hydrogen_id = graph.num_nodes
        for node_idx, num_to_add in enumerate(hydrogens_to_add):
            for _ in range(num_to_add):
                new_edge_index[0, current_edge_idx] = node_idx
                new_edge_index[1, current_edge_idx] = current_hydrogen_id
                new_edge_index[1, current_edge_idx + 1] = node_idx
                new_edge_index[0, current_edge_idx + 1] = current_hydrogen_id
                current_edge_idx += 2
                current_hydrogen_id += 1

        assert current_edge_idx == new_edge_index.shape[1]
        return Data(
            edge_index=new_edge_index,
            atom_labels=new_atom_labels,
            bond_labels=new_bond_labels,
            implicit_hydrogens=new_implicit_hydrogens,
            explicit_hydrogens=new_explicit_hydrogens,
            radical_electrons=new_radical_electrons,
            charges=new_charges
        )
