from typing import Literal, Iterable
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import GraphDescriptors, Lipinski
import numpy as np
from pyprojroot import here
from fcd.fcd import get_predictions, load_ref_model
from sklearn.random_projection import SparseRandomProjection
from molclr import GINet, mol_to_graph
from torch_geometric.data import Batch

from polygraph.descriptors import GraphDescriptor


class TopoChemicalDescriptor(GraphDescriptor[Chem.Mol]):
    def __call__(self, mols: Iterable[Chem.Mol]):
        all_fps = []
        for mol in mols:
            fp = [
                GraphDescriptors.AvgIpc(mol),  # pyright: ignore
                GraphDescriptors.BertzCT(mol),  # pyright: ignore
                GraphDescriptors.BalabanJ(mol),  # pyright: ignore
                GraphDescriptors.HallKierAlpha(mol),  # pyright: ignore
                GraphDescriptors.Kappa1(mol),  # pyright: ignore
                GraphDescriptors.Kappa2(mol),  # pyright: ignore
                GraphDescriptors.Kappa3(mol),  # pyright: ignore
                GraphDescriptors.Chi0(mol),  # pyright: ignore
                GraphDescriptors.Chi0n(mol),  # pyright: ignore
                GraphDescriptors.Chi0v(mol),  # pyright: ignore
                GraphDescriptors.Chi1(mol),  # pyright: ignore
                GraphDescriptors.Chi1n(mol),  # pyright: ignore
                GraphDescriptors.Chi1v(mol),  # pyright: ignore
                GraphDescriptors.Chi2n(mol),  # pyright: ignore
                GraphDescriptors.Chi2v(mol),  # pyright: ignore
                GraphDescriptors.Chi3n(mol),  # pyright: ignore
                GraphDescriptors.Chi3v(mol),  # pyright: ignore
                GraphDescriptors.Chi4n(mol),  # pyright: ignore
                GraphDescriptors.Chi4v(mol),  # pyright: ignore
            ]
            fp = np.array(fp)
            all_fps.append(fp)
        return np.stack(all_fps, axis=0)


class FingerprintDescriptor(GraphDescriptor[Chem.Mol]):
    def __init__(
        self, dim: int = 128, algorithm: Literal["rdkit", "morgan"] = "rdkit"
    ):
        self._dim = dim
        if algorithm == "rdkit":
            self._fpgen = AllChem.GetRDKitFPGenerator(fpSize=self._dim)  # pyright: ignore
        elif algorithm == "morgan":
            self._fpgen = AllChem.GetMorganGenerator(fpSize=self._dim)  # pyright: ignore
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

    def __call__(self, mols: Iterable[Chem.Mol]):
        all_fps = []

        for mol in mols:
            fp = self._fpgen.GetCountFingerprint(mol)
            fp = np.array(list(fp))
            all_fps.append(fp)
        return np.stack(all_fps, axis=0)


class LipinskiDescriptor(GraphDescriptor[Chem.Mol]):
    """
    Calculates all Lipinski descriptors available in rdkit.Chem.Lipinski module.
    Based on: https://www.rdkit.org/docs/source/rdkit.Chem.Lipinski.html
    """

    def __call__(self, mols: Iterable[Chem.Mol]):
        all_descriptors = []
        for mol in mols:
            descriptors = [
                # Basic Lipinski descriptors
                Lipinski.HeavyAtomCount(mol),  # pyright: ignore
                Lipinski.NHOHCount(mol),  # pyright: ignore
                Lipinski.NOCount(mol),  # pyright: ignore
                Lipinski.NumHAcceptors(mol),  # pyright: ignore
                Lipinski.NumHDonors(mol),  # pyright: ignore
                Lipinski.NumHeteroatoms(mol),  # pyright: ignore
                Lipinski.NumRotatableBonds(mol),  # pyright: ignore
                Lipinski.RingCount(mol),  # pyright: ignore
                # Ring-related descriptors
                Lipinski.NumAliphaticCarbocycles(mol),  # pyright: ignore
                Lipinski.NumAliphaticHeterocycles(mol),  # pyright: ignore
                Lipinski.NumAliphaticRings(mol),  # pyright: ignore
                Lipinski.NumAromaticCarbocycles(mol),  # pyright: ignore
                Lipinski.NumAromaticHeterocycles(mol),  # pyright: ignore
                Lipinski.NumAromaticRings(mol),  # pyright: ignore
                Lipinski.NumHeterocycles(mol),  # pyright: ignore
                Lipinski.NumSaturatedCarbocycles(mol),  # pyright: ignore
                Lipinski.NumSaturatedHeterocycles(mol),  # pyright: ignore
                Lipinski.NumSaturatedRings(mol),  # pyright: ignore
                # Structural descriptors
                Lipinski.NumAmideBonds(mol),  # pyright: ignore
                Lipinski.NumAtomStereoCenters(mol),  # pyright: ignore
                Lipinski.NumUnspecifiedAtomStereoCenters(mol),  # pyright: ignore
                Lipinski.NumBridgeheadAtoms(mol),  # pyright: ignore
                Lipinski.NumSpiroAtoms(mol),  # pyright: ignore
                # Chemical descriptors
                Lipinski.FractionCSP3(mol),  # pyright: ignore
                Lipinski.Phi(mol),  # pyright: ignore
            ]
            descriptors = np.array(descriptors)
            all_descriptors.append(descriptors)
        return np.stack(all_descriptors, axis=0)


class ChemNetDescriptor(GraphDescriptor[Chem.Mol]):
    def __init__(self, dim: int = 128):
        self._dim = dim
        self._model = load_ref_model(
            here() / ".local/molecule_data/chemnet_model.pt"
        )
        self._proj = SparseRandomProjection(
            n_components=self._dim,
            random_state=42,  # pyright: ignore
        )

    def __call__(self, mols: Iterable[Chem.Mol]):
        smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols]
        return self._proj.fit_transform(get_predictions(self._model, smiles))


class MolCLRDescriptor(GraphDescriptor[Chem.Mol]):
    def __init__(self, dim: int = 128, batch_size: int = 128):
        self._dim = dim
        self._model = GINet(
            num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool="mean"
        )
        self._model.load_state_dict(
            torch.load(
                here() / ".local/molecule_data/molclr_model.pth",
                map_location="cpu",
            )
        )
        self._model.eval()
        self._proj = SparseRandomProjection(
            n_components=self._dim,
            random_state=42,  # pyright: ignore
        )
        self._batch_size = batch_size

    @torch.inference_mode()
    def __call__(self, mols: Iterable[Chem.Mol]):
        graphs = [mol_to_graph(mol) for mol in mols]
        embeddings = []
        for i in range(0, len(graphs), self._batch_size):
            batch = Batch.from_data_list(graphs[i : i + self._batch_size])
            h, _ = self._model(batch)
            embeddings.append(h)
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
        assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}"
        if embeddings.shape[1] != self._dim:
            embeddings = self._proj.fit_transform(embeddings)
        assert embeddings.shape[1] == self._dim
        return embeddings
