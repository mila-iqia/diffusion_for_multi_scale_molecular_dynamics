from typing import Tuple

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.lattice_utils import \
    get_relative_coordinates_lattice_vectors


def get_closest_relative_coordinates_and_index(
    reference_relative_coordinates: torch.Tensor,
    relative_coordinates: torch.Tensor,
    avoid_self: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Get closest relative coordinates and index.

    This algorithm works with relative coordinates, assuming Euclidean distance in that space. This is appropriate
    for a cubic unit cell. It accounts for periodicity.

    Args:
        reference_relative_coordinates: a relative coordinate of interest, of dimension [spatial_dimension].
        relative_coordinates: a config of relative_coordinates, of dimension [number_of_atoms, spatial_dimension].
        avoid_self: If reference_relative_coordinates is a member of relative_coordinates, should it avoid finding
            itself as its own closest neighbor.

    Returns:
        shortest_relative_coordinate_distance: shortest near neighbor distance.
        index: index of the relative_coordinates atom which is closest to reference_relative_coordinates.
    """
    assert (
        len(reference_relative_coordinates.shape) == 1
    ), "A single reference coordinate must be given."
    spatial_dimension = reference_relative_coordinates.shape[0]

    assert (
        len(relative_coordinates.shape) == 2
    ), "A single configuration of relative coordinates must be given."
    assert (
        relative_coordinates.shape[1] == spatial_dimension
    ), "Spatial dimensions are inconsistent."
    natoms = relative_coordinates.shape[0]

    lattice_vectors = get_relative_coordinates_lattice_vectors(
        number_of_shells=1, spatial_dimension=spatial_dimension
    )

    repeated_lattice_vectors = einops.repeat(lattice_vectors, "l d -> l n d", n=natoms)
    repeated_reference = einops.repeat(
        reference_relative_coordinates, "d -> l n d", l=len(lattice_vectors), n=natoms
    )
    repeated_relative_coordinates = einops.repeat(
        relative_coordinates, "n d -> l n d", l=len(lattice_vectors)
    )

    # Dimension [number of lattice vectors, number_of_atoms]
    repeated_squared_distances = (
        (repeated_reference + repeated_lattice_vectors - repeated_relative_coordinates)
        ** 2
    ).sum(dim=-1)

    if avoid_self:
        repeated_squared_distances[torch.where(repeated_squared_distances == 0.0)] = (
            torch.inf
        )

    # Dimension [number_of_atoms]
    squared_distances = repeated_squared_distances.min(dim=0).values

    index = squared_distances.argmin()
    minimum_distance = torch.sqrt(squared_distances.min())

    return minimum_distance, index
