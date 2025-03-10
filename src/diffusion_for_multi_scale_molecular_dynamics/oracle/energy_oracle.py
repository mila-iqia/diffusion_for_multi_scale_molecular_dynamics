import logging
from dataclasses import dataclass
from typing import AnyStr, Dict, List, Tuple, Union

import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, AXL_COMPOSITION, CARTESIAN_POSITIONS, UNIT_CELL)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class OracleParameters:
    """Lammps Oracle Parameters."""

    name: str  # what kind of Oracle
    elements: List[str]  # unique elements


class EnergyOracle:
    """Energy oracle base class."""

    def __init__(self, oracle_parameters: OracleParameters, **kwargs):
        """Init method."""
        self._oracle_parameters = oracle_parameters
        self._element_types = ElementTypes(oracle_parameters.elements)

    def _compute_one_configuration_energy_and_forces(
        self,
        cartesian_positions: np.ndarray,
        basis_vectors: np.ndarray,
        atom_types: np.ndarray,
    ) -> Tuple[float, Union[torch.Tensor, np.ndarray]]:
        raise NotImplementedError("This method must be implemented")

    def compute_oracle_energies_and_forces(
        self, samples: Dict[AnyStr, Union[torch.Tensor, np.ndarray]]
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """Compute oracle energies.

        Method to call the oracle for samples expressed in a standardized format.

        Args:
            samples:  a dictionary assumed to contain the fields
                        - CARTESIAN_POSITIONS
                        - UNIT_CELL

        Returns:
            energies: a numpy array with the computed energies.
        """
        assert (
            CARTESIAN_POSITIONS in samples
        ), f"the field '{CARTESIAN_POSITIONS}' must be present in the sample dictionary"

        assert (
            UNIT_CELL in samples
        ), f"the field '{UNIT_CELL}' must be present in the sample dictionary"

        assert (
            AXL_COMPOSITION in samples or ATOM_TYPES in samples
        ), f"the field '{AXL_COMPOSITION}' or '{ATOM_TYPES}' must be present in the sample dictionary"

        if isinstance(samples[UNIT_CELL], torch.Tensor):
            # Dimension [batch_size, space_dimension, space_dimension]
            batched_basis_vectors = (
                samples[UNIT_CELL].detach().cpu().numpy()
            )  # TODO: use the AXL_COMPOSITION
        else:
            batched_basis_vectors = samples[UNIT_CELL]

        # Dimension [batch_size, number_of_atoms, space_dimension]
        if isinstance(samples[CARTESIAN_POSITIONS], torch.Tensor):
            batched_cartesian_positions = (
                samples[CARTESIAN_POSITIONS].detach().cpu().numpy()
            )
            return_type = torch.Tensor
        else:
            batched_cartesian_positions = samples[CARTESIAN_POSITIONS]
            return_type = np.ndarray

        if AXL_COMPOSITION in samples:
            # Dimension [batch_size, number_of_atoms]
            batched_atom_types = samples[AXL_COMPOSITION].A.detach().cpu().numpy()
        else:
            batched_atom_types = samples[ATOM_TYPES]

        logger.info("Compute energy from Oracle")
        list_energy = []
        list_forces = []
        for cartesian_positions, basis_vectors, atom_types in zip(
            batched_cartesian_positions, batched_basis_vectors, batched_atom_types
        ):
            energy, forces = self._compute_one_configuration_energy_and_forces(
                cartesian_positions, basis_vectors, atom_types
            )
            list_energy.append(energy)
            list_forces.append(forces)
        logger.info("Done computing energies from Oracle")

        if return_type == torch.Tensor:
            forces = torch.tensor(np.stack(list_forces))
            energies = torch.tensor(list_energy)
        else:
            forces = np.stack(list_forces)
            energies = np.array(list_energy)

        return energies, forces
