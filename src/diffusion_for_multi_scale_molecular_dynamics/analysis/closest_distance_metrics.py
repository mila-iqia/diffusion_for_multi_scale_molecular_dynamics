from typing import List

import numpy as np
from ase import Atoms


def compute_closest_distances(list_of_atoms: List[Atoms]) -> np.ndarray:
    """Compute each atom's distance to its nearest neighbor, across a list of structures.

    Args:
        list_of_atoms: structures to analyse. Each must have at least 2 atoms.

    Returns:
        closest_distances: one nearest-neighbor distance per atom, concatenated across all structures in
            list_of_atoms.
    """
    closest_distances = []
    for atoms in list_of_atoms:
        assert len(atoms) >= 2, "A structure needs at least 2 atoms to have a nearest neighbor."
        distance_matrix = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distance_matrix, np.inf)
        closest_distances.append(distance_matrix.min(axis=1))
    return np.concatenate(closest_distances)
