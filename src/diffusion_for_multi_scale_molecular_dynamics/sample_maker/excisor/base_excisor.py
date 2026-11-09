from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@dataclass(kw_only=True)
class BaseEnvironmentExcisionArguments:
    """Parameters controlling the environment excision."""
    algorithm: str


class BaseEnvironmentExcision(ABC):
    """Base class for atomic environment excision."""

    def __init__(self, excision_arguments: BaseEnvironmentExcisionArguments):
        """Init method."""
        self.arguments = excision_arguments

    def excise_environments(
        self, structure: AXL, central_atoms_indices: np.array, center_atoms: bool = True
    ) -> Tuple[List[AXL], List[int]]:
        """Extract all environments around the atoms satisfying the uncertainty constraints.

        This calls the method _excise_one_environment for each atom with a high enough uncertainty.

        Args:
            structure: crystal structure, including atomic species, relative coordinates and lattice parameters
            central_atoms_indices: indices of atoms at the center of environments to be excised from the structure.
                It is assumed that the indices correspond to the atom ordering the input structure.
            center_atoms: if True, apply a translation to all atoms such that the central atom is in the middle of the
                box. Defaults to True.

        Returns:
            excised_environments: list of excised environment around the central atoms identified by their indices.
            excised_central_atoms_indices: list of the indices of the central atom in the excised structures.
        """
        excised_environments = []
        excised_central_atoms_indices = []

        for atom_index in central_atoms_indices:
            excised_environment, excised_atom_index = self._excise_one_environment(structure, atom_index)
            if center_atoms:
                excised_environment = self.center_structure(excised_environment, excised_atom_index)

            excised_environments.append(excised_environment)
            excised_central_atoms_indices.append(excised_atom_index)

        return excised_environments, excised_central_atoms_indices

    @staticmethod
    def center_structure(structure: AXL, atom_index: int) -> AXL:
        """Center the atom around which the excision occurred.

        Args:
            structure: crystal structure, including atomic species, relative coordinates and lattice parameters
            atom_index: index around which the excision occurred. This atom will be translated to the center of the box.

        Returns:
            translated_structure: structure translated so that atom denoted with atom_index is at the center of the box
        """
        central_atom_relative_coordinates = structure.X[atom_index, :]
        translation_to_apply = (
            np.ones_like(central_atom_relative_coordinates) * 0.5
            - central_atom_relative_coordinates
        )
        translated_relative_coordinates = np.mod(structure.X + translation_to_apply, 1)
        translated_structure = AXL(
            A=structure.A, X=translated_relative_coordinates, L=structure.L
        )
        return translated_structure

    @abstractmethod
    def _excise_one_environment(self, structure: AXL, central_atom_idx: int) -> Tuple[AXL, int]:
        """Excise the relevant atomic environment around the central atom.

        Args:
            structure: complete structure to excise from
            central_atom_idx: atom at the center of the excised region

        Returns:
            excised_substructure: all atoms within a distance radial_cutoff of the central atom (including itself).
            excised_central_atom_idx: the index of the central atom in the newly created excised substructure.
        """
        pass
