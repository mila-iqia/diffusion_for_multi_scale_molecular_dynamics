import logging
import tempfile
from typing import AnyStr, Dict

import numpy as np
import torch

from crystal_diffusion.namespace import CARTESIAN_POSITIONS, UNIT_CELL
from crystal_diffusion.oracle.lammps import get_energy_and_forces_from_lammps

logger = logging.getLogger(__name__)


def compute_oracle_energies(samples: Dict[AnyStr, torch.Tensor]) -> torch.Tensor:
    """Compute oracle energies.

    Method to call the oracle for samples expressed in a standardized format.

    Args:
        samples:  a dictionary assumed to contain the fields
                    - CARTESIAN_POSITIONS
                    - UNIT_CELL

    Returns:
        energies: a numpy array with the computed energies.
    """
    assert CARTESIAN_POSITIONS in samples, \
        f"the field '{CARTESIAN_POSITIONS}' must be present in the sample dictionary"

    assert UNIT_CELL in samples, \
        f"the field '{UNIT_CELL}' must be present in the sample dictionary"

    # Dimension [batch_size, space_dimension, space_dimension]
    basis_vectors = samples[UNIT_CELL].detach().cpu().numpy()

    # Dimension [batch_size, number_of_atoms, space_dimension]
    cartesian_positions = samples[CARTESIAN_POSITIONS].detach().cpu().numpy()

    number_of_atoms = cartesian_positions.shape[1]
    atom_types = np.ones(number_of_atoms, dtype=int)

    logger.info("Compute energy from Oracle")

    list_energy = []
    with tempfile.TemporaryDirectory() as tmp_work_dir:
        for positions, box in zip(cartesian_positions, basis_vectors):
            energy, forces = get_energy_and_forces_from_lammps(positions,
                                                               box,
                                                               atom_types,
                                                               tmp_work_dir=tmp_work_dir)
            list_energy.append(energy)
    logger.info("Done computing energies from Oracle")
    return torch.tensor(list_energy)
