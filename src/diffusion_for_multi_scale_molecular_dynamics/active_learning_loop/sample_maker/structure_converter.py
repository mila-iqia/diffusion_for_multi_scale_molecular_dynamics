from typing import List

import numpy as np
import torch
from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    map_lattice_parameters_to_unit_cell_vectors,
    map_unit_cell_to_lattice_parameters)
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import \
    create_structure


class StructureConverter:
    """Structure Converter.

    This class is responsible for converting pymatgen Structure objects into AXL objects.
    """
    def __init__(self, list_of_element_symbols: List[str]):
        """Init method.

        Args:
            list_of_element_symbols: list of unique element symbols that will be present. It is assumed that
                these are "real" elements (not arbitrary strings), since this class will interact with
                Pymatgen Structure objects, which only knows about the real elements.
        """
        self._element_type = ElementTypes(elements=list_of_element_symbols)

    def convert_structure_to_axl(self, structure: Structure) -> AXL:
        """Convert Pymatgen structure to AXL.

        Args:
            structure: a Pymatgen structure.

        Returns:
            axl_structure: the same data, but represented as an AXL.
        """
        atom_types = [self._element_type.get_element_id(element.symbol) for element in structure.species]
        axl_structure = AXL(
            A=np.array(atom_types),
            X=structure.frac_coords,
            L=map_unit_cell_to_lattice_parameters(structure.lattice.matrix, engine="numpy"),
        )
        return axl_structure

    def convert_axl_to_structure(self, axl_structure: AXL) -> Structure:
        """Convert AXL to Pymatgen structure.

        Args:
            axl_structure: an AXL containing the information to build a Pymatgen structure. It is assumed that
                the X and L fields contain numpy arrays, and that the A field contains the element id, ie an
                integer corresponding to the element symbol.

        Returns:
            structure: The corresponding Pymatgen structure.
        """
        species = [self._element_type.get_element(element_id) for element_id in axl_structure.A]

        relative_coordinates = axl_structure.X
        basis_vectors = map_lattice_parameters_to_unit_cell_vectors(
            torch.tensor(axl_structure.L)
        ).numpy()
        structure = create_structure(basis_vectors, relative_coordinates, species)
        return structure
