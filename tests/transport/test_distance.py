import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.transport.distance import (
    get_geodesic_displacements, get_squared_geodesic_distance,
    get_squared_geodesic_distance_cost_matrix)


@pytest.fixture
def batch_shape():
    return 3, 4, 5


@pytest.fixture
def spatial_dimension():
    return 3


@pytest.fixture
def number_of_atoms():
    return 14


@pytest.fixture
def batch_x1(batch_shape, spatial_dimension):
    return torch.rand(*batch_shape, spatial_dimension)


@pytest.fixture
def batch_x2(batch_shape, spatial_dimension):
    return torch.rand(*batch_shape, spatial_dimension)


@pytest.fixture
def x1(number_of_atoms, spatial_dimension):
    return torch.rand(number_of_atoms, spatial_dimension)


@pytest.fixture
def x2(number_of_atoms, spatial_dimension):
    return torch.rand(number_of_atoms, spatial_dimension)


@pytest.fixture
def expected_displacements(batch_x1, batch_x2):
    list_d = []
    for v1, v2 in zip(batch_x1.flatten(), batch_x2.flatten()):
        raw_difference = v2 - v1

        if raw_difference < -0.5:
            difference = raw_difference + 1.0
        elif raw_difference < 0.5:
            difference = raw_difference
        else:
            difference = raw_difference - 1.0
        list_d.append(difference)

    return torch.tensor(list_d).reshape(batch_x1.shape)


def test_get_geodesic_displacements(batch_x1, batch_x2, expected_displacements):
    computed_displacements = get_geodesic_displacements(batch_x1, batch_x2)
    torch.testing.assert_allclose(computed_displacements, expected_displacements)


def test_get_squared_geodesic_distance(x1, x2):
    computed_displacements = get_geodesic_displacements(x1, x2)
    expected_squared_distance = torch.sum(computed_displacements**2)
    computed_squared_distance = get_squared_geodesic_distance(x1, x2)
    torch.testing.assert_allclose(computed_squared_distance, expected_squared_distance)


@pytest.fixture()
def expected_cost_matrix(x1, x2, number_of_atoms):

    cost_matrix = torch.zeros(number_of_atoms, number_of_atoms)
    for i, v1 in enumerate(x1):
        for j, v2 in enumerate(x2):
            cost_matrix[i, j] = get_squared_geodesic_distance(v1, v2)

    return cost_matrix


def test_get_squared_geodesic_distance_cost_matrix(x1, x2, expected_cost_matrix):
    computed_cost_matrix = get_squared_geodesic_distance_cost_matrix(x1, x2)
    torch.testing.assert_allclose(computed_cost_matrix, expected_cost_matrix)
