"""Adapt a batch of AXL samples to a single-point calculator (diffusion-side energy/forces).

The diffusion model evaluates batches of generated structures expressed in the AXL representation. These
functions bridge that batched AXL format to the per-structure ``BaseSinglePointCalculator`` API: they build
an ``ase.Atoms`` per sample and delegate to ``calculate_many``.
"""

import logging
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from ase import Atoms

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    BaseSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates,
    map_lattice_parameters_to_unit_cell_vectors)
from diffusion_for_multi_scale_molecular_dynamics.utils.element_types import \
    ElementTypes

logger = logging.getLogger(__name__)

# A box side shorter than this (in Angstrom) makes LAMMPS crash (its communication cutoff cannot exceed the
# box length); such samples are skipped rather than evaluated. This is a LAMMPS guard, not a physical one.
MINIMUM_BOX_SIDE_LENGTH = 3.0


def _get_atoms_from_axl_configuration(
    relative_coordinates: torch.Tensor,
    lattice_parameters: torch.Tensor,
    atom_types: torch.Tensor,
    element_types: ElementTypes,
) -> Optional[Atoms]:
    """Convert a single AXL configuration to an ase.Atoms, or None if its box is too small for LAMMPS."""
    spatial_dimension = relative_coordinates.shape[-1]

    lattice_parameters = lattice_parameters.clone()
    lattice_parameters[spatial_dimension:] = 0  # TODO support non-orthogonal boxes
    if lattice_parameters[:spatial_dimension].min() < 0:
        warnings.warn("Got a negative lattice parameter. Clipping to 1.0 Angstrom")
        lattice_parameters[:spatial_dimension] = np.clip(
            lattice_parameters[:spatial_dimension], a_min=1.0, a_max=None
        )

    basis_vectors = map_lattice_parameters_to_unit_cell_vectors(lattice_parameters)
    cartesian_positions = get_positions_from_coordinates(
        relative_coordinates, basis_vectors
    ).numpy()
    basis_vectors = basis_vectors.numpy()
    assert np.allclose(
        basis_vectors, np.diag(np.diag(basis_vectors))
    ), "only orthogonal LAMMPS box are valid"

    if np.diag(basis_vectors).min() < MINIMUM_BOX_SIDE_LENGTH:
        warnings.warn(
            f"Got a box with a side length smaller than {MINIMUM_BOX_SIDE_LENGTH} Angstrom in LAMMPS. "
            "Skipping this example."
        )
        return None

    species = [element_types.get_element(int(atom_type)) for atom_type in atom_types]
    return Atoms(
        symbols=species,
        positions=cartesian_positions,
        cell=basis_vectors,
        pbc=True,
    )


def compute_axl_energies_and_forces(
    axl: AXL,
    single_point_calculator: BaseSinglePointCalculator,
    elements: List[str],
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """Compute energies and forces for a batch of AXL samples.

    Each sample is turned into an ``ase.Atoms`` and evaluated by the injected single-point
    calculator (via ``calculate_many``). Samples whose box is too small for LAMMPS are skipped and reported
    as zero energy and forces. The output type mirrors the AXL input (torch or numpy).

    Args:
        axl: a batched AXL (X: relative coordinates, A: atom types, L: lattice parameters).
        single_point_calculator: the calculator that labels each structure.
        elements: the unique elements, mapping atom-type ids to species.

    Returns:
        energies: the computed energies (one per sample).
        forces: the computed forces (one array per sample).
    """
    element_types = ElementTypes(elements)

    batched_relative_coordinates = torch.as_tensor(axl.X).detach().cpu()
    batched_lattice_parameters = torch.as_tensor(axl.L).detach().cpu()
    batched_atom_types = torch.as_tensor(axl.A).detach().cpu()
    return_type = torch.Tensor if isinstance(axl.X, torch.Tensor) else np.ndarray

    number_of_samples, number_of_atoms, spatial_dimension = (
        batched_relative_coordinates.shape
    )
    assert spatial_dimension == 3, (
        "The single-point calculators build ase.Atoms, so labelling AXL samples is only "
        f"supported in 3 spatial dimensions (got {spatial_dimension})."
    )

    # Build one configuration per (non-degenerate) sample, remembering which batch index it came from.
    list_energy: List[float] = [0.0] * number_of_samples
    list_forces: List[np.ndarray] = [
        np.zeros((number_of_atoms, spatial_dimension)) for _ in range(number_of_samples)
    ]
    list_atoms: List[Atoms] = []
    atoms_batch_indices: List[int] = []

    for batch_index, (
        relative_coordinates,
        lattice_parameters,
        atom_types,
    ) in enumerate(
        zip(
            batched_relative_coordinates, batched_lattice_parameters, batched_atom_types
        )
    ):
        atoms = _get_atoms_from_axl_configuration(
            relative_coordinates, lattice_parameters, atom_types, element_types
        )
        if atoms is not None:
            list_atoms.append(atoms)
            atoms_batch_indices.append(batch_index)

    calculations = single_point_calculator.calculate_many(list_atoms)
    for batch_index, calculation in zip(atoms_batch_indices, calculations):
        list_energy[batch_index] = calculation.energy
        list_forces[batch_index] = calculation.forces

    if return_type == torch.Tensor:
        return torch.tensor(list_energy), torch.tensor(np.stack(list_forces))
    return np.array(list_energy), np.stack(list_forces)
