from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


def draw_molecule(
    smiles: str,
    filename: Union[str, Path],
    size: tuple[int, int] = (300, 300),
    legend: Optional[str] = None,
    dpi: int = 100,
) -> bool:
    """Draw a molecule from its SMILES string and save it to a file using matplotlib.

    Args:
        smiles: SMILES string of the molecule
        filename: Path to save the image to
        size: Size of the image in pixels (width, height)
        legend: Optional legend to add to the image
        dpi: Dots per inch for the output image

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Draw molecule first to get its natural aspect ratio
        img = Draw.MolToImage(mol)

        # Create figure with the exact pixel size we want
        figsize = (size[0] / dpi, size[1] / dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Add single subplot that fills the figure
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        # Display the molecule image
        ax.imshow(img, aspect="equal")

        # Remove padding and margins
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if legend:
            ax.set_title(legend, pad=10)

        # Save with exact dimensions
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        plt.close("all")
        return True
    except Exception:
        plt.close("all")
        return False


def draw_molecules(
    smiles_list: List[str],
    output_dir: Union[str, Path],
    filename_pattern: str = "molecule_{}.png",
    size: tuple[int, int] = (300, 300),
    legends: Optional[List[str]] = None,
    dpi: int = 100,
) -> List[bool]:
    """Draw multiple molecules from their SMILES strings and save them to files using matplotlib.

    Args:
        smiles_list: List of SMILES strings
        output_dir: Directory to save the images to
        filename_pattern: Pattern for filenames, must contain {} for index
        size: Size of each image in pixels (width, height)
        legends: Optional list of legends for each molecule
        dpi: Dots per inch for the output images

    Returns:
        List[bool]: List of booleans indicating success/failure for each molecule
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, smiles in enumerate(smiles_list):
        filename = output_dir / filename_pattern.format(i)
        legend = legends[i] if legends is not None else None
        success = draw_molecule(smiles, filename, size=size, legend=legend, dpi=dpi)
        results.append(success)

    plt.close("all")
    return results


def draw_molecules_grid(
    smiles_list: List[str],
    filename: Union[str, Path],
    legends: Optional[List[str]] = None,
    molsPerRow: int = 4,
    subImgSize: tuple[int, int] = (300, 300),
    dpi: int = 100,
) -> bool:
    """Draw multiple molecules in a grid layout and save to a single file using matplotlib.

    Args:
        smiles_list: List of SMILES strings
        filename: Path to save the image to
        legends: Optional list of legends for each molecule
        molsPerRow: Number of molecules per row in the grid
        subImgSize: Size of each molecule image in pixels (width, height)
        dpi: Dots per inch for the output image

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        if any(mol is None for mol in mols):
            return False

        # Validate legends
        if legends is not None and len(legends) != len(smiles_list):
            legends = None

        n_mols = len(mols)
        n_rows = (n_mols + molsPerRow - 1) // molsPerRow

        # Calculate figure size in inches
        figsize = (molsPerRow * subImgSize[0] / dpi, n_rows * subImgSize[1] / dpi)

        # Create figure and axes grid
        fig, axes = plt.subplots(
            n_rows,
            molsPerRow,
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
            constrained_layout=True,
        )

        # Plot each molecule
        for idx, mol in enumerate(mols):
            row = idx // molsPerRow
            col = idx % molsPerRow
            ax = axes[row, col]

            # Remove axes
            ax.set_axis_off()

            # Draw molecule
            img = Draw.MolToImage(mol)
            ax.imshow(img, aspect="equal")

            # Add legend if provided
            if legends is not None:
                ax.set_title(legends[idx], pad=2, wrap=True)

        # Hide empty subplots
        for idx in range(len(mols), n_rows * molsPerRow):
            row = idx // molsPerRow
            col = idx % molsPerRow
            axes[row, col].set_visible(False)

        # Save with exact dimensions
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        plt.close("all")
        return True
    except Exception:
        plt.close("all")
        return False
