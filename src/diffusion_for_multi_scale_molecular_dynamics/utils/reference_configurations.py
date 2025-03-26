import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def get_silicon_supercell(supercell_factor: int):
    """Get silicon supercell."""
    primitive_cell_a = 3.84
    lattice = Lattice.from_parameters(
        a=primitive_cell_a,
        b=primitive_cell_a,
        c=primitive_cell_a,
        alpha=60.0,
        beta=60.0,
        gamma=60.0,
    )

    species = ["Si", "Si"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])

    primitize_structure = Structure(
        lattice=lattice, species=species, coords=coordinates, coords_are_cartesian=False
    )
    conventional_structure = (
        SpacegroupAnalyzer(primitize_structure)
        .get_symmetrized_structure()
        .to_conventional()
    )

    super_structure = conventional_structure.make_supercell(
        [supercell_factor, supercell_factor, supercell_factor]
    )

    return super_structure.frac_coords


def create_equilibrium_sige_structure():
    """Create the SiGe 1x1x1 equilibrium structure."""
    conventional_cell_a = 5.542
    primitive_cell_a = conventional_cell_a / np.sqrt(2.0)
    lattice = Lattice.from_parameters(
        a=primitive_cell_a,
        b=primitive_cell_a,
        c=primitive_cell_a,
        alpha=60.0,
        beta=60.0,
        gamma=60.0,
    )

    species = ["Si", "Ge"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])

    primitive_structure = Structure(
        lattice=lattice, species=species, coords=coordinates, coords_are_cartesian=False
    )
    conventional_structure = (
        SpacegroupAnalyzer(primitive_structure)
        .get_symmetrized_structure()
        .to_conventional()
    )

    # Shift the relative coordinates a bit for easier visualization
    shift = np.array([0.375, 0.375, 0.375])
    new_coordinates = (conventional_structure.frac_coords + shift) % 1.0

    structure = Structure(
        lattice=conventional_structure.lattice,
        species=conventional_structure.species,
        coords=new_coordinates,
        coords_are_cartesian=False,
    )
    return structure
