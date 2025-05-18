from dataclasses import dataclass

from ase import Atoms
import numpy as np


@dataclass(kw_only=True)
class LabelledStructure:
    """A structure and corresponding labels, for training a sparse GP."""
    atoms: Atoms
    forces: np.ndarray
    energy: float
    active_set_indices: np.ndarray
    spatial_dimension: int = 3

    def __post_init__(self):

        number_of_atoms = len(self.atoms)
        assert self.forces.shape == (number_of_atoms, self.spatial_dimension)
        assert len(self.active_set_indices) <= number_of_atoms
        assert len(self.active_set_indices) > 0
        assert np.all(self.active_set_indices >= 0)
        assert np.all(self.active_set_indices < number_of_atoms)

        # No repeated indices
        assert len(set(self.active_set_indices)) == len(self.active_set_indices)