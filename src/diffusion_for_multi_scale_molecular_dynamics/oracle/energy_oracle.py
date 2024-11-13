import logging
from dataclasses import dataclass
from typing import AnyStr, Dict, List

import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, CARTESIAN_POSITIONS, UNIT_CELL)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class OracleParameters:
    """Lammps Oracle Parameters."""
    name: str  # what kind of Oracle
    elements: List[str]  # unique elements


class EnergyOracle:
    """Energy oracle base class."""
    def __init__(
        self, oracle_parameters: OracleParameters, **kwargs
    ):
        """Init method."""
        self._oracle_parameters = oracle_parameters
        self._element_types = ElementTypes(oracle_parameters.elements)

    def _compute_one_configuration_energy(
        self,
        cartesian_positions: np.ndarray,
        basis_vectors: np.ndarray,
        atom_types: np.ndarray,
    ) -> float:
        raise NotImplementedError("This method must be implemented")

    def compute_oracle_energies(
        self, samples: Dict[AnyStr, torch.Tensor]
    ) -> torch.Tensor:
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

        # Dimension [batch_size, space_dimension, space_dimension]
        batched_basis_vectors = samples[UNIT_CELL].detach().cpu().numpy()

        # Dimension [batch_size, number_of_atoms, space_dimension]
        batched_cartesian_positions = (
            samples[CARTESIAN_POSITIONS].detach().cpu().numpy()
        )

        # Dimension [batch_size, number_of_atoms]
        batched_atom_types = samples[ATOM_TYPES].detach().cpu().numpy()

        logger.info("Compute energy from Oracle")
        list_energy = []
        for cartesian_positions, basis_vectors, atom_types in zip(
            batched_cartesian_positions, batched_basis_vectors, batched_atom_types
        ):
            energy = self._compute_one_configuration_energy(
                cartesian_positions, basis_vectors, atom_types
            )
            list_energy.append(energy)
        logger.info("Done computing energies from Oracle")
        return torch.tensor(list_energy)
