from typing import Literal, Iterable
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import GraphDescriptors, Lipinski
import numpy as np
from fcd.fcd import get_predictions, load_ref_model
from sklearn.random_projection import SparseRandomProjection
from torch_geometric.data import Batch

from polygraph.utils.descriptors.molclr import mol_to_graph, load_molclr_model
from polygraph.utils.descriptors import GraphDescriptor


class TopoChemicalDescriptor(GraphDescriptor[Chem.Mol]):
    """Computes topological properties."""

    def __call__(self, mols: Iterable[Chem.Mol]) -> np.ndarray:
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
    """Computes molecular fingerprints.

    Args:
        dim: Dimension of the fingerprint
        algorithm: Algorithm to use for fingerprint generation. Either "rdkit" or "morgan".
    """

    def __init__(
        self, dim: int = 128, algorithm: Literal["rdkit", "morgan"] = "morgan"
    ):
        self._dim = dim
        if algorithm == "rdkit":
            self._fpgen = AllChem.GetRDKitFPGenerator(fpSize=self._dim)  # pyright: ignore
        elif algorithm == "morgan":
            self._fpgen = AllChem.GetMorganGenerator(fpSize=self._dim)  # pyright: ignore
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

    def __call__(self, mols: Iterable[Chem.Mol]) -> np.ndarray:
        all_fps = []

        for mol in mols:
            fp = self._fpgen.GetCountFingerprint(mol)
            fp = np.array(list(fp))
            all_fps.append(fp)
        return np.stack(all_fps, axis=0)


class LipinskiDescriptor(GraphDescriptor[Chem.Mol]):
    """Physico-chemical properties of molecules."""

    def __call__(self, mols: Iterable[Chem.Mol]) -> np.ndarray:
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
    """Random projection of ChemNet embeddings.

    Args:
        dim: Dimension of the projected embedding
    """

    def __init__(self, dim: int = 128):
        self._dim = dim
        self._model = load_ref_model()
        self._proj = SparseRandomProjection(
            n_components=self._dim,  # pyright: ignore
            random_state=42,
        )

    def __call__(self, mols: Iterable[Chem.Mol]) -> np.ndarray:
        smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols]
        return self._proj.fit_transform(get_predictions(self._model, smiles))


class MolCLRDescriptor(GraphDescriptor[Chem.Mol]):
    """Random projection of MolCLR embeddings.

    Args:
        dim: Dimension of the projected embedding
        batch_size: Batch size for the model used during inference
    """

    def __init__(self, dim: int = 128, batch_size: int = 128):
        self._dim = dim
        self._model = load_molclr_model()
        self._proj = SparseRandomProjection(
            n_components=self._dim,  # pyright: ignore
            random_state=42,
        )
        self._batch_size = batch_size

    @torch.inference_mode()
    def __call__(self, mols: Iterable[Chem.Mol]) -> np.ndarray:
        graphs = [mol_to_graph(mol) for mol in mols]
        embeddings = []
        for i in range(0, len(graphs), self._batch_size):
            batch = Batch.from_data_list(graphs[i : i + self._batch_size])  # pyright: ignore
            h, _ = self._model(batch)
            embeddings.append(h)
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
        assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}"
        if embeddings.shape[1] != self._dim:
            embeddings = self._proj.fit_transform(embeddings)
        assert embeddings.shape[1] == self._dim
        return embeddings
