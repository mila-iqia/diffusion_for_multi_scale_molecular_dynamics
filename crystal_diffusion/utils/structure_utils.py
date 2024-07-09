from typing import List

import numpy as np
from pymatgen.core import Lattice, Structure


def create_structure(basis_vectors: np.ndarray, relative_coordinates: np.ndarray, species: List[str]) -> Structure:
    """Create structure.

    A utility method to convert various arrays in to a pymatgen structure.

    Args:
        basis_vectors: vectors that define the unit cell.
        relative_coordinates: atomic relative coordinates.
        species : the species. In the same order as relative coordinates.

    Returns:
        structure: a pymatgen structure.
    """
    lattice = Lattice(matrix=basis_vectors, pbc=(True, True, True))

    structure = Structure(lattice=lattice,
                          species=species,
                          coords=relative_coordinates,
                          coords_are_cartesian=False)
    return structure
