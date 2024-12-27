import os

import matplotlib.pyplot as plt
import pytest
from loguru import logger

from graph_gen_gym.utils.molecules import (
    draw_molecule,
    draw_molecules,
    draw_molecules_grid,
)


@pytest.fixture
def test_smiles():
    return [
        "CC(=O)O",  # Acetic acid
        "CCO",  # Ethanol
        "C1=CC=CC=C1",  # Benzene
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]


@pytest.fixture
def test_legends():
    return ["Acetic acid", "Ethanol", "Benzene", "Caffeine"]


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def test_draw_molecule_basic(temp_dir, test_smiles):
    # Test successful drawing
    filename = temp_dir / "test_molecule.png"
    success = draw_molecule(test_smiles[0], filename)
    assert success
    assert filename.exists()


def test_draw_molecule_with_legend(temp_dir, test_smiles):
    # Test with legend
    filename = temp_dir / "test_molecule_legend.png"
    success = draw_molecule(test_smiles[0], filename, legend="Acetic acid")
    assert success
    assert filename.exists()


def test_draw_molecule_custom_size(temp_dir, test_smiles):
    # Test with custom size
    filename = temp_dir / "test_molecule_size.png"
    size = (400, 400)
    dpi = 100
    success = draw_molecule(test_smiles[0], filename, size=size, dpi=dpi)
    assert success
    assert filename.exists()

    # Verify image dimensions (allow for matplotlib's size adjustments)
    img = plt.imread(filename)
    # Allow for a 10% difference in dimensions
    height_diff = abs(img.shape[0] - size[1])
    width_diff = abs(img.shape[1] - size[0])
    assert (
        height_diff <= size[1] * 0.1
    ), f"Height difference too large: {height_diff} pixels"
    assert (
        width_diff <= size[0] * 0.1
    ), f"Width difference too large: {width_diff} pixels"


def test_draw_molecule_invalid_smiles(temp_dir):
    # Test invalid SMILES
    filename = temp_dir / "invalid.png"
    success = draw_molecule("invalid_smiles", filename)
    assert not success
    assert not filename.exists()


def test_draw_molecules_basic(temp_dir, test_smiles):
    # Test basic functionality
    results = draw_molecules(test_smiles[:2], temp_dir)
    assert all(results)
    assert (temp_dir / "molecule_0.png").exists()
    assert (temp_dir / "molecule_1.png").exists()


def test_draw_molecules_custom_filename(temp_dir, test_smiles):
    # Test with custom filename pattern
    results = draw_molecules(
        test_smiles[:2], temp_dir, filename_pattern="custom_{}.png"
    )
    assert all(results)
    assert (temp_dir / "custom_0.png").exists()
    assert (temp_dir / "custom_1.png").exists()


def test_draw_molecules_with_legends(temp_dir, test_smiles, test_legends):
    # Test with legends
    results = draw_molecules(
        test_smiles[:2],
        temp_dir,
        legends=test_legends[:2],
        filename_pattern="legend_{}.png",
    )
    assert all(results)
    assert (temp_dir / "legend_0.png").exists()
    assert (temp_dir / "legend_1.png").exists()


def test_draw_molecules_invalid_smiles(temp_dir, test_smiles):
    # Test with some invalid SMILES
    mixed_smiles = ["invalid_smiles"] + test_smiles[:2]
    results = draw_molecules(mixed_smiles, temp_dir, filename_pattern="mixed_{}.png")
    assert not results[0]  # First should fail
    assert all(results[1:])  # Rest should succeed


def test_draw_molecules_grid_basic(temp_dir, test_smiles):
    # Test basic grid
    filename = temp_dir / "grid.png"
    success = draw_molecules_grid(test_smiles, filename)
    assert success
    assert filename.exists()


def test_draw_molecules_grid_with_legends(temp_dir, test_smiles, test_legends):
    # Test with legends
    filename = temp_dir / "grid_legends.png"
    logger.debug(f"Drawing grid with legends to {filename}")
    success = draw_molecules_grid(test_smiles, filename, legends=test_legends)
    logger.debug(f"Success: {success}")
    assert success
    assert filename.exists()


def test_draw_molecules_grid_custom_rows(temp_dir, test_smiles):
    # Test with custom molsPerRow
    filename = temp_dir / "grid_2per_row.png"
    success = draw_molecules_grid(test_smiles, filename, molsPerRow=2)
    assert success
    assert filename.exists()


def test_draw_molecules_grid_dimensions(temp_dir, test_smiles):
    # Test with custom subImgSize and verify dimensions
    filename = temp_dir / "grid_custom_size.png"
    size = (300, 300)
    dpi = 100
    success = draw_molecules_grid(
        test_smiles, filename, subImgSize=size, molsPerRow=2, dpi=dpi
    )
    assert success
    assert filename.exists()

    # Verify grid dimensions (allow for matplotlib's size adjustments)
    img = plt.imread(filename)
    n_rows = (len(test_smiles) + 1) // 2  # For 2 molecules per row
    expected_height = n_rows * size[1]
    expected_width = 2 * size[0]  # 2 molecules per row

    # Allow for a 30% difference in dimensions due to padding, titles, and matplotlib adjustments
    height_diff = abs(img.shape[0] - expected_height)
    width_diff = abs(img.shape[1] - expected_width)
    assert height_diff <= expected_height * 0.3, (
        f"Height difference too large: got {img.shape[0]}, "
        f"expected {expected_height} ± 30% ({height_diff} pixels off)"
    )
    assert width_diff <= expected_width * 0.3, (
        f"Width difference too large: got {img.shape[1]}, "
        f"expected {expected_width} ± 30% ({width_diff} pixels off)"
    )


def test_draw_molecules_grid_invalid_smiles(temp_dir, test_smiles):
    # Test with invalid SMILES
    filename = temp_dir / "grid_invalid.png"
    invalid_smiles = ["invalid_smiles"] + test_smiles
    success = draw_molecules_grid(invalid_smiles, filename)
    assert not success
    assert not filename.exists()


def test_draw_molecule_with_string_path(temp_dir, test_smiles):
    # Test with string paths
    str_path = os.path.join(temp_dir, "str_path.png")
    success = draw_molecule(test_smiles[0], str_path)
    assert success
    assert os.path.exists(str_path)


def test_draw_molecule_with_path_object(temp_dir, test_smiles):
    # Test with Path objects
    path_obj = temp_dir / "path_obj.png"
    success = draw_molecule(test_smiles[0], path_obj)
    assert success
    assert path_obj.exists()


def test_draw_molecules_creates_directories(temp_dir, test_smiles):
    # Test directory creation in draw_molecules
    nested_dir = temp_dir / "nested" / "dir"
    results = draw_molecules(test_smiles[:2], nested_dir)
    assert all(results)
    assert (nested_dir / "molecule_0.png").exists()
    assert (nested_dir / "molecule_1.png").exists()
