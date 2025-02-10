import itertools

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.closest_neighbors import \
    get_closest_relative_coordinates_and_index


def brute_force_closest_relative_coordinates_and_index(
    reference_relative_coordinates, relative_coordinates, avoid_self
):
    spatial_dimension = reference_relative_coordinates.shape[0]
    list_l = torch.tensor([-1.0, 0.0, 1.0])
    all_lattice_vectors = [
        torch.tensor(ell) for ell in itertools.product(*(spatial_dimension * [list_l]))
    ]

    shortest_distance = torch.inf
    index = None

    for idx, x in enumerate(relative_coordinates):
        list_distances = torch.tensor(
            [
                torch.linalg.norm(x + lattice_vector - reference_relative_coordinates)
                for lattice_vector in all_lattice_vectors
            ]
        )
        if avoid_self:
            list_distances[torch.where(list_distances == 0.0)] = torch.inf

        minimum_distance = list_distances.min()
        if minimum_distance < shortest_distance:
            shortest_distance = minimum_distance
            index = idx

    return shortest_distance, index


@pytest.fixture()
def spatial_dimension():
    return 3


@pytest.fixture()
def number_of_atoms():
    return 8


@pytest.fixture()
def reference_relative_coordinates(spatial_dimension):
    return torch.rand(spatial_dimension)


@pytest.fixture()
def relative_coordinates(number_of_atoms, spatial_dimension):
    return torch.rand(number_of_atoms, spatial_dimension)


def test_get_closest_relative_coordinates_different_configurations(
    reference_relative_coordinates, relative_coordinates
):
    avoid_self = False
    expected_shortest_distance, expected_index = (
        brute_force_closest_relative_coordinates_and_index(
            reference_relative_coordinates, relative_coordinates, avoid_self
        )
    )

    computed_shortest_distance, computed_index = (
        get_closest_relative_coordinates_and_index(
            reference_relative_coordinates, relative_coordinates, avoid_self=avoid_self
        )
    )

    torch.testing.assert_close(computed_shortest_distance, expected_shortest_distance)
    assert expected_index == computed_index


def test_get_closest_relative_coordinates_same_configuration(relative_coordinates):
    avoid_self = True

    for reference_relative_coordinates in relative_coordinates:
        expected_shortest_distance, expected_index = (
            brute_force_closest_relative_coordinates_and_index(
                reference_relative_coordinates, relative_coordinates, avoid_self
            )
        )

        computed_shortest_distance, computed_index = (
            get_closest_relative_coordinates_and_index(
                reference_relative_coordinates,
                relative_coordinates,
                avoid_self=avoid_self,
            )
        )

        torch.testing.assert_close(
            computed_shortest_distance, expected_shortest_distance
        )
        assert expected_index == computed_index
