from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from crystal_diffusion import DATA_DIR
from crystal_diffusion.oracle.lammps import get_energy_and_forces_from_lammps


class LAMMPS_for_active_learning:
    def __init__(self):
        pass

    def __call__(self,
                 cartesian_positions: np.ndarray,
                 box: np.ndarray,
                 atom_types: np.ndarray,
                 atom_type_map: Dict[int, str] = {1: 'Si'},
                 tmp_work_dir: str = './',
                 pair_coeff_dir: Path = DATA_DIR) -> Tuple[float, np.ndarray]:
        """Call LAMMPS to get energy and forces for a given set of atoms.

        Args:
            cartesian_positions: atomic positions as a n_atom x 3 array
            box: unit cell definition as a 3x3 array. Assumed to be diagonal.
            atom_types: integers defining each atoms as an array of length n_atom
            atom_type_map: map between indices and atom type. Defaults to {1: 'Si'}
            tmp_work_dir: temporary work directory for LAMMPS. Defaults to ./
            pair_coeff_dir: path to stilinger-weber potential. Defaults to DATA_DIR.

        Returns:
            energy and forces on each atom (n_atom x 3)
        """
        shifted_positions = self.shift_positions(cartesian_positions, box)
        energy, forces = get_energy_and_forces_from_lammps(shifted_positions, box, atom_types, atom_type_map,
                                                           tmp_work_dir, pair_coeff_dir)
        return energy, forces[['fx', 'fy', 'fz']].to_numpy()

    def shift_positions(self, cartesian_positions: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Shift the positions of the atoms so all coordinates are positives.

        This is because LAMMPS will ignore atoms with coordinates outside the [0, a] range (a = size of the unit cell).

        Args:
            cartesian_positions: atomic positions (n_atom x 3 array)
            box: unit cell (3x3 array) - assumed to be diagonal

        Returns:
            array with shifted positions
        """
        for i, cell_size in enumerate(np.diag(box)):
            cartesian_positions[:, i] = cartesian_positions[:, i] % cell_size
        return cartesian_positions
