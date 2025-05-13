from dataclasses import dataclass
from typing import List

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import (
    BaseEnvironmentExcision, BaseEnvironmentExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.utils import \
    get_distances_from_reference_point
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@dataclass(kw_only=True)
class SphericalExcisionArguments(BaseEnvironmentExcisionArguments):
    algorithm: str = "radial_excision"
    radial_cutoff: float = 3.0  # radial cutoff in Angstrom


class SphericalExcision(BaseEnvironmentExcision):
    """Extract all atoms given"""

    def __init__(self, excision_arguments: SphericalExcisionArguments):
        super().__init__(excision_arguments)
        self.radial_cutoff = excision_arguments.radial_cutoff
        assert (
            self.radial_cutoff > 0
        ), f"Radial cutoff is expected to be positive. Got {self.radial_cutoff}"

    def excise_environments(
        self, structure: AXL, uncertainty_per_atom: np.array
    ) -> List[AXL]:
        """

        Args:
            structure: crystal structure, including atomic species, coordinates and lattice parameters
            uncertainty_per_atom: uncertainty associated to each atom. The order is assumed to be the same as those in
                the structure variable.

        Returns:
            environments: list of excised spheres around the highest uncertainties atoms as a list of AXL.
        """
        central_atoms = self.select_central_atoms(uncertainty_per_atom)
        environments = [
            self._excise_one_environment(structure, atom) for atom in central_atoms
        ]
        return environments

    def _excise_one_environment(self, structure: AXL, central_atom_idx: int) -> AXL:
        """Excise the atoms within a distance radial_cutoff from a central atom.

        Args:
            structure: complete structure to excise from
            central_atom_idx: atom at the center of the excised region

        Returns:
            excised_substructure: all atoms within a distance radial_cutoff of the central atom (including itself).
        """
        central_atom_position = structure.X[central_atom_idx, :]
        distances_from_central_atom = get_distances_from_reference_point(
            structure.X, central_atom_position, structure.L
        )
        indices_closer_than_threshold = np.where(
            distances_from_central_atom < self.radial_cutoff
        )[0]
        excised_substructure = AXL(
            A=structure.A[indices_closer_than_threshold],
            X=structure.X[indices_closer_than_threshold, :],
            L=structure.L,
        )
        return excised_substructure
