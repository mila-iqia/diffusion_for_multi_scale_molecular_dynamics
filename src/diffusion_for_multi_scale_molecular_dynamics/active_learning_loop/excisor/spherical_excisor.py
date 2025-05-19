from dataclasses import dataclass

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import (
    BaseEnvironmentExcision, BaseEnvironmentExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.utils import \
    get_distances_from_reference_point
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@dataclass(kw_only=True)
class SphericalExcisionArguments(BaseEnvironmentExcisionArguments):
    """Arguments for a spherical cutoff around a target atom."""

    algorithm: str = "radial_excision"
    radial_cutoff: float = 3.0  # radial cutoff in Angstrom

    def __post_init__(self):
        """Post init."""
        super().__post_init__()
        assert (
            self.radial_cutoff > 0
        ), f"Radial cutoff is expected to be positive. Got {self.radial_cutoff}"


class SphericalExcision(BaseEnvironmentExcision):
    """Extract all atoms within a given distance of the pivot atom."""

    def __init__(self, excision_arguments: SphericalExcisionArguments):
        """Init method."""
        super().__init__(excision_arguments)
        self.radial_cutoff = excision_arguments.radial_cutoff

    def _excise_one_environment(self, structure: AXL, central_atom_idx: int) -> AXL:
        """Excise the atoms within a distance radial_cutoff from a central atom.

        Args:
            structure: complete structure to excise from
            central_atom_idx: atom at the center of the excised region

        Returns:
            excised_substructure: all atoms within a distance radial_cutoff of the central atom (including itself).
        """
        central_atom_relative_coordinates = structure.X[central_atom_idx, :]
        distances_from_central_atom = get_distances_from_reference_point(
            structure.X, central_atom_relative_coordinates, structure.L
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
