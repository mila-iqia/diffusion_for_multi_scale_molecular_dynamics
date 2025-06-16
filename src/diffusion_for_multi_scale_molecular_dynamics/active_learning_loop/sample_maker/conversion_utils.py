import numpy as np
import torch
from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    map_lattice_parameters_to_unit_cell_vectors,
    map_unit_cell_to_lattice_parameters)
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import \
    create_structure


def convert_structure_to_axl(structure: Structure) -> AXL:
    """Convert structure to AXL.

    Args:
        structure: a pymatgen structure.

    Returns:
        axl_structure: the same data, but represented as an AXL.
    """
    axl_structure = AXL(
        A=np.array([element.symbol for element in structure.species]),
        X=structure.frac_coords,
        L=map_unit_cell_to_lattice_parameters(structure.lattice.matrix, engine="numpy"),
    )
    return axl_structure


def convert_axl_to_structure(axl_structure: AXL) -> Structure:
    """Convert AXL to structure.

    Args:
        axl_structure: an AXL containing the information to build a pymatgen structure. It is assumed that
            the X and L fields contain numpy arrays, and that the A field contains the element symbols.

    Returns:
        structure: The corresponding pymatgen structure.
    """
    species = axl_structure.A
    relative_coordinates = axl_structure.X
    basis_vectors = map_lattice_parameters_to_unit_cell_vectors(
        torch.tensor(axl_structure.L)
    ).numpy()
    structure = create_structure(basis_vectors, relative_coordinates, species)
    return structure
