import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.geometric_utils import \
    get_cubic_point_group_symmetries
from diffusion_for_multi_scale_molecular_dynamics.utils.lattice_utils import (
    _sort_complete_shell, get_cubic_point_group_complete_lattice_shells,
    get_cubic_point_group_positive_normalized_bloch_wave_vectors,
    get_relative_coordinates_lattice_vectors)


@pytest.fixture()
def spatial_dimension():
    return 3


@pytest.mark.parametrize("number_of_shells", [1, 2, 3])
def test_get_relative_coordinates_lattice_vectors_1d(number_of_shells):

    expected_lattice_vectors = []

    for nx in torch.arange(-number_of_shells, number_of_shells + 1):
        lattice_vector = torch.tensor([nx])
        expected_lattice_vectors.append(lattice_vector)

    expected_lattice_vectors = torch.stack(expected_lattice_vectors).to(
        dtype=torch.float32
    )
    computed_lattice_vectors = get_relative_coordinates_lattice_vectors(
        number_of_shells, spatial_dimension=1
    )

    torch.testing.assert_close(expected_lattice_vectors, computed_lattice_vectors)


@pytest.mark.parametrize("number_of_shells", [1, 2, 3])
def test_get_relative_coordinates_lattice_vectors_2d(number_of_shells):

    expected_lattice_vectors = []

    for nx in torch.arange(-number_of_shells, number_of_shells + 1):
        for ny in torch.arange(-number_of_shells, number_of_shells + 1):
            lattice_vector = torch.tensor([nx, ny])
            expected_lattice_vectors.append(lattice_vector)

    expected_lattice_vectors = torch.stack(expected_lattice_vectors).to(
        dtype=torch.float32
    )
    computed_lattice_vectors = get_relative_coordinates_lattice_vectors(
        number_of_shells, spatial_dimension=2
    )

    torch.testing.assert_close(expected_lattice_vectors, computed_lattice_vectors)


@pytest.mark.parametrize("number_of_shells", [1, 2, 3])
def test_get_relative_coordinates_lattice_vectors_3d(number_of_shells):

    expected_lattice_vectors = []

    for nx in torch.arange(-number_of_shells, number_of_shells + 1):
        for ny in torch.arange(-number_of_shells, number_of_shells + 1):
            for nz in torch.arange(-number_of_shells, number_of_shells + 1):
                lattice_vector = torch.tensor([nx, ny, nz])
                expected_lattice_vectors.append(lattice_vector)

    expected_lattice_vectors = torch.stack(expected_lattice_vectors).to(
        dtype=torch.float32
    )
    computed_lattice_vectors = get_relative_coordinates_lattice_vectors(
        number_of_shells, spatial_dimension=3
    )

    torch.testing.assert_close(expected_lattice_vectors, computed_lattice_vectors)


def test_sort_complete_shell():
    shell = torch.tensor(
        [[1, 0, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0], [0, 1, 0], [-1, 0, 0]],
        dtype=torch.int32,
    )

    expected_sorted_shell = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0], [-1, 0, 0]],
        dtype=torch.int32,
    )

    sorted_shell = _sort_complete_shell(shell)

    torch.testing.assert_close(expected_sorted_shell, sorted_shell)


def test_get_cubic_point_group_complete_lattice_shells(spatial_dimension):
    number_of_complete_shells = 3
    list_shells = get_cubic_point_group_complete_lattice_shells(
        number_of_complete_shells=number_of_complete_shells,
        spatial_dimension=spatial_dimension,
    )

    symmetries = get_cubic_point_group_symmetries(spatial_dimension).int()

    assert len(list_shells) >= number_of_complete_shells

    for shell in list_shells:
        computed_set = set(tuple(ell) for ell in shell.numpy())
        expected_set = set(tuple(ell) for ell in torch.matmul(symmetries, shell[0]).numpy())
        assert computed_set == expected_set


def test_get_cubic_point_group_positive_normalized_bloch_wave_vectors(
    spatial_dimension,
):

    number_of_complete_shells = 3
    list_shells = get_cubic_point_group_complete_lattice_shells(
        number_of_complete_shells, spatial_dimension
    )

    all_lattice_vectors = torch.vstack(list_shells)

    bloch_wave_vectors = get_cubic_point_group_positive_normalized_bloch_wave_vectors(
        number_of_complete_shells, spatial_dimension
    )

    assert len(bloch_wave_vectors) == len(all_lattice_vectors) / 2

    expected_set = set(tuple(ell.numpy()) for ell in all_lattice_vectors)
    computed_set = set(tuple(ell.numpy()) for ell in bloch_wave_vectors)
    assert computed_set.issubset(expected_set)

    inversion = -torch.eye(spatial_dimension).int()

    inversion_bloch_wave_vectors = torch.matmul(bloch_wave_vectors, inversion)
    inversion_set = set(tuple(ell.numpy()) for ell in inversion_bloch_wave_vectors)
    computed_set.update(inversion_set)

    assert computed_set == expected_set
