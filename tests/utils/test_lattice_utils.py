import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.lattice_utils import \
    get_relative_coordinates_lattice_vectors


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
