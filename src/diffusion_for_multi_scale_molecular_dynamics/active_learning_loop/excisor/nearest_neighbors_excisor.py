from dataclasses import dataclass

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import (
    BaseEnvironmentExcision, BaseEnvironmentExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.utils import \
    get_distances_from_reference_point
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@dataclass(kw_only=True)
class NearestNeighborsExcisionArguments(BaseEnvironmentExcisionArguments):
    """Arguments for a selection of nearest atoms around a target atom."""

    algorithm: str = "nearest_neighbors"
    number_of_neighbors: int = (
        4  # number of nearest neighbors to the pivot atom to keep in the excised region
    )

    def __post_init__(self):
        """Post init."""
        super().__post_init__()
        assert (
            self.number_of_neighbors > 0
        ), f"Number of neighbors to include is expected to be positive. Got {self.number_of_neighbors}"


class NearestNeighborsExcision(BaseEnvironmentExcision):
    """Extract the N nearest neighbors to the pivot atom."""

    def __init__(self, excision_arguments: NearestNeighborsExcisionArguments):
        """Init method."""
        super().__init__(excision_arguments)
        self.number_of_neighbors = excision_arguments.number_of_neighbors

    def _excise_one_environment(self, structure: AXL, central_atom_idx: int) -> AXL:
        """Excise the N nearest atoms within a distance radial_cutoff from a central atom.

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
        # find the indices sorting the distances from closer to most distant
        sorted_indices = np.argsort(distances_from_central_atom)
        # the N nearest are the nearest neighbor. Add 1 to include the central atom itself.
        nearest_neighbor_indices = sorted_indices[: self.number_of_neighbors + 1]

        excised_substructure = AXL(
            A=structure.A[nearest_neighbor_indices],
            X=structure.X[nearest_neighbor_indices, :],
            L=structure.L,
        )
        return excised_substructure
