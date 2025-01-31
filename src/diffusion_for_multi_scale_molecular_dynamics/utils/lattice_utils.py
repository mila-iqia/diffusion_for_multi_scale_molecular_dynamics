import itertools

import torch


def get_relative_coordinates_lattice_vectors(
    number_of_shells: int = 1, spatial_dimension: int = 3
) -> torch.Tensor:
    """Get relative coordinates lattice vectors.

    Get all the lattice vectors in relative coordinates from -number_of_shells to +number_of_shells,
    in every spatial directions.

    Args:
        number_of_shells: number of shifts along lattice vectors in the positive direction.

    Returns:
        list_relative_lattice_vectors : all the lattice vectors in relative coordinates (ie, integers).
    """
    shifts = range(-number_of_shells, number_of_shells + 1)
    list_relative_lattice_vectors = 1.0 * torch.tensor(
        list(itertools.product(shifts, repeat=spatial_dimension))
    )

    return list_relative_lattice_vectors
