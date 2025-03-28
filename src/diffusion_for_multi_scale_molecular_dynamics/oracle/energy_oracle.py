import logging
import warnings
from dataclasses import dataclass
from typing import AnyStr, Dict, List, Tuple, Union

import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, AXL_COMPOSITION, LATTICE_PARAMETERS, RELATIVE_COORDINATES)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates,
    map_lattice_parameters_to_unit_cell_vectors)

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
            samples:  a dictionary assumed to contain the field AXL_COMPOSITION

        Returns:
            energies: a numpy array with the computed energies.
        """
        assert (
            LATTICE_PARAMETERS in samples or AXL_COMPOSITION in samples
        ), f"the field '{LATTICE_PARAMETERS}' or '{AXL_COMPOSITION}' must be present in the sample dictionary"

        assert (
            AXL_COMPOSITION in samples or ATOM_TYPES in samples
        ), f"the field '{AXL_COMPOSITION}' or '{ATOM_TYPES}' must be present in the sample dictionary"

        batched_relative_coordinates = (
            samples[RELATIVE_COORDINATES]
            if RELATIVE_COORDINATES in samples
            else samples[AXL_COMPOSITION].X
        )
        if isinstance(batched_relative_coordinates, torch.Tensor):
            batched_relative_coordinates = batched_relative_coordinates.detach().cpu()
            return_type = torch.Tensor
        else:
            return_type = np.ndarray

        # Dimension [batch_size, space_dimension, space_dimension]
        batched_lattice_parameters = (
            samples[LATTICE_PARAMETERS]
            if LATTICE_PARAMETERS in samples
            else samples[AXL_COMPOSITION].L
        )

        if isinstance(batched_lattice_parameters, torch.Tensor):
            batched_lattice_parameters = batched_lattice_parameters.detach().cpu()

        batched_atom_types = (
            samples[ATOM_TYPES] if ATOM_TYPES in samples else samples[AXL_COMPOSITION].A
        )
        if isinstance(batched_atom_types, torch.Tensor):
            batched_atom_types = batched_atom_types.detach().cpu()

        logger.info("Compute energy from Oracle")
        list_energy = []
        list_forces = []
        spatial_dimension = batched_relative_coordinates.shape[-1]
        for relative_coordinates, lattice_parameters, atom_types in zip(
            batched_relative_coordinates, batched_lattice_parameters, batched_atom_types
        ):
            lattice_parameters[spatial_dimension:] = (
                0  # TODO support non-orthogonal boxes
            )
            if lattice_parameters[:spatial_dimension].min() < 0:
                warnings.warn(
                    "Got a negative lattice parameter. Clipping to 1.0 Angstrom"
                )
                lattice_parameters[:spatial_dimension] = np.clip(
                    lattice_parameters[:spatial_dimension], a_min=1.0, a_max=None
                )
            basis_vectors = map_lattice_parameters_to_unit_cell_vectors(
                lattice_parameters
            )
            cartesian_positions = get_positions_from_coordinates(
                relative_coordinates, basis_vectors
            )

            energy, forces = self._compute_one_configuration_energy_and_forces(
                cartesian_positions.numpy(), basis_vectors.numpy(), atom_types
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
