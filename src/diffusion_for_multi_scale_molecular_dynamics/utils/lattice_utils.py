import itertools
from typing import List

import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.geometric_utils import \
    get_cubic_point_group_symmetries


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


def _sort_complete_shell(complete_shell: torch.Tensor) -> torch.Tensor:
    """Sort complete shell.

    Sort the lattice vectors so that the most positive elements appear first.

    Args:
        complete_shell: a tensor of dimension [number of lattice vectors, spatial dimension].

    Returns:
        sorted shell: the complete shell, sorted.
    """
    number_of_lattice_vectors, spatial_dimension = complete_shell.shape

    # Build a scalar quantity that will sort following the desired rules
    ordering_scalar = torch.zeros(number_of_lattice_vectors)

    for d in range(spatial_dimension):
        column = complete_shell[:, d]

        power = spatial_dimension - d - 1
        factor = number_of_lattice_vectors**power

        sorted_unique_values = column.unique().sort().values

        for rank, unique_value in enumerate(sorted_unique_values):
            idx = column == unique_value
            ordering_scalar[idx] += rank * factor

    order = torch.flip(ordering_scalar.argsort(), dims=(0,))

    ordered_shell = complete_shell[order]
    return ordered_shell


def get_cubic_point_group_complete_lattice_shells(
    number_of_complete_shells: int, spatial_dimension: int = 3
) -> List[torch.Tensor]:
    """Get cubic point group complete lattice shells.

    This method creates lattice vectors organized in fully symmetric shells, sorted according to the
    length of a shell member, excluding L=0. If there are length degeneracies, all degenerate shells are included.

    The cubic point group is ASSUMED. This should be modified to deal with other point groups.

    Args:
        number_of_complete_shells: number of complete shells of lattice vectors to create.
        spatial_dimension: the dimension of space.

    Returns:
        list_complete_shells: a list of complete, sorted lattice vector shells.
    """
    number_of_trial_shells = 2 * number_of_complete_shells
    lattice_vectors = get_relative_coordinates_lattice_vectors(
        number_of_trial_shells, spatial_dimension
    )
    # sort by their norm
    squared_norms = (lattice_vectors**2).sum(-1)

    # The vectors should be composed of integers
    sorted_lattice_vectors = lattice_vectors[squared_norms.argsort()].int()

    symmetries = get_cubic_point_group_symmetries(spatial_dimension).int()

    number_of_created_shells = 0
    known_set = set()

    list_complete_shells = []

    previous_shell_squared_norm = 0

    # Exclude zero.
    for lattice_vector in sorted_lattice_vectors[1:]:
        if tuple(lattice_vector.numpy()) in known_set:
            continue

        new_shell_set = set(
            tuple(ell) for ell in torch.matmul(symmetries, lattice_vector).numpy()
        )
        known_set.update(new_shell_set)
        number_of_created_shells += 1

        complete_shell = _sort_complete_shell(torch.tensor(list(new_shell_set)))

        list_complete_shells.append(complete_shell)

        shell_squared_norm = (lattice_vector**2).sum()

        if (
            len(list_complete_shells) >= number_of_complete_shells
            and shell_squared_norm > previous_shell_squared_norm
        ):
            break
        previous_shell_squared_norm = shell_squared_norm

    return list_complete_shells


def get_cubic_point_group_positive_normalized_bloch_wave_vectors(
    number_of_complete_shells: int, spatial_dimension: int = 3
) -> torch.Tensor:
    """Get cubic point group positive normalized bloch wave vectors.

    This method generates reciprocal lattice vectors for the cubic lattice. The vectors are "normalized" in
    the sense that they are made up of integers, such that, for spatial_dimension = 3,

        K[m] = m_1 b_1 + m_2 b_2 + m_3 b_3,  b_i = 2pi/a e_i

    The "normalized" vectors are of the form [m_1, m_2,  m_3].

    We assume that inversion is part of the point group. Thus, we can transform

        {e^[i K r], e^[-i K r] } ->  {cos(K r), sin(K r) }

    and so there is no need to manipulate complex numbers. Thus we wil only keep half of lattice vectors,
    removing the half that is the image under inversion.

    Args:
        number_of_complete_shells: number of complete shells to consider
        spatial_dimension: dimension of space

    Returns:
        positive_bloch_wave_vectors: half of complete shell normalized reciprocal lattice vectors.
    """
    list_complete_shells = get_cubic_point_group_complete_lattice_shells(
        number_of_complete_shells, spatial_dimension
    )
    list_half_shells = []

    inversion = -torch.eye(spatial_dimension).int()

    for shell in list_complete_shells:
        known_set = set()
        half_shell = []
        for lattice_vector in shell:
            if tuple(lattice_vector.numpy()) in known_set:
                continue

            half_shell.append(lattice_vector)

            inverse_lattice_vector = torch.matmul(inversion, lattice_vector)
            s = {tuple(lattice_vector.numpy()), tuple(inverse_lattice_vector.numpy())}
            known_set.update(s)
        list_half_shells.append(torch.stack(half_shell))

    positive_bloch_wave_vectors = torch.vstack(list_half_shells)
    return positive_bloch_wave_vectors
