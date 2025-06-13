from itertools import product

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.utils import (
    find_partition_sizes, get_distances_from_reference_point,
    partition_relative_coordinates_for_voxels, select_occupied_voxels)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_number_of_lattice_parameters


@pytest.fixture
def num_atoms():
    return 8


@pytest.fixture(params=[1, 2, 3])
def spatial_dimension(request):
    return request.param


@pytest.fixture
def relative_coordinates(num_atoms, spatial_dimension):
    return np.random.rand(num_atoms, spatial_dimension)


@pytest.fixture
def lattice_parameters(spatial_dimension):
    num_lattice_parameters = get_number_of_lattice_parameters(spatial_dimension)
    lp = np.zeros(num_lattice_parameters)
    lp[:spatial_dimension] = np.random.rand(spatial_dimension) * 8
    return lp


@pytest.fixture
def box_size(lattice_parameters, spatial_dimension):
    return lattice_parameters[:spatial_dimension]


@pytest.fixture
def cartesian_positions(relative_coordinates, box_size):
    # assume the box is orthogonal
    return relative_coordinates * box_size


@pytest.fixture
def target_point_relative_coordinates(spatial_dimension):
    return np.random.rand(spatial_dimension)


@pytest.fixture
def target_point_cartesian_positions(target_point_relative_coordinates, box_size):
    return target_point_relative_coordinates * box_size


@pytest.fixture
def expected_distances_to_atoms(
    cartesian_positions, target_point_cartesian_positions, box_size, spatial_dimension
):
    all_distances = np.zeros(cartesian_positions.shape[0])
    for atom_idx in range(cartesian_positions.shape[0]):
        distances_to_this_atom = np.zeros(spatial_dimension)
        for d in range(spatial_dimension):
            direct_distance = (
                cartesian_positions[atom_idx, d] - target_point_cartesian_positions[d]
            )
            possible_distance_squared = [
                (direct_distance) ** 2,
                (direct_distance - box_size[d]) ** 2,
                (direct_distance + box_size[d]) ** 2,
            ]
            distances_to_this_atom[d] = min(possible_distance_squared)
        all_distances[atom_idx] = np.sqrt(distances_to_this_atom.sum())
    return all_distances


def test_get_distances_from_reference_point(
    relative_coordinates,
    target_point_relative_coordinates,
    lattice_parameters,
    expected_distances_to_atoms,
):
    calculated_distances_to_atoms = get_distances_from_reference_point(
        relative_coordinates, target_point_relative_coordinates, lattice_parameters
    )
    assert np.allclose(calculated_distances_to_atoms, expected_distances_to_atoms)


@pytest.fixture(params=[1, 8, 129])
def num_voxels(request):
    return request.param


@pytest.fixture
def expected_partition_size(box_size, num_voxels, spatial_dimension):
    voxel_volume = np.prod(box_size) / num_voxels
    voxel_volume_scaled_by_dimension = voxel_volume ** (1 / spatial_dimension)
    num_voxels_per_dimension = []
    for d in range(spatial_dimension):
        best_guess_for_voxel = box_size[d] / voxel_volume_scaled_by_dimension
        num_voxels_per_dimension.append(max(np.round(best_guess_for_voxel), 1))
    return np.array(num_voxels_per_dimension)


def test_find_partition_size(box_size, num_voxels, expected_partition_size):
    calculated_partition_size = find_partition_sizes(box_size, num_voxels)
    assert np.array_equal(calculated_partition_size, expected_partition_size)


@pytest.fixture
def expected_partition_relative_coordinates(expected_partition_size, spatial_dimension):
    relative_coordinates = []
    for d in range(spatial_dimension):
        dimension_coordinates = (
            np.arange(expected_partition_size[d]) / expected_partition_size[d]
        )
        relative_coordinates.append(dimension_coordinates)
    expected_relative_coordinates = np.array(list(product(*relative_coordinates)))
    return expected_relative_coordinates.transpose()


def test_partition_relative_coordinates_for_voxels(
    expected_partition_relative_coordinates, box_size, num_voxels
):
    calculated_partition_coordinates, _ = partition_relative_coordinates_for_voxels(
        box_size, num_voxels
    )
    assert np.allclose(
        calculated_partition_coordinates, expected_partition_relative_coordinates
    )


@pytest.fixture(params=[1, 8, 256])
def num_atoms_for_voxels(request):
    return request.param


def test_select_occupied_voxels(num_voxels, num_atoms):
    # test what happens in num_voxels equal num_atoms
    calculated_full_single_occupancy = select_occupied_voxels(num_voxels, num_voxels)
    _, calculated_full_single_occupancy_counts = np.unique(
        calculated_full_single_occupancy, return_counts=True
    )
    assert calculated_full_single_occupancy_counts.sum() == num_voxels
    assert np.array_equal(
        calculated_full_single_occupancy_counts, np.ones(num_voxels).astype(int)
    )

    # in the real case, we can check that the sum match
    calculated_occupancy = select_occupied_voxels(num_voxels, num_atoms)
    assert np.all(0 <= calculated_occupancy)
    assert np.all(calculated_occupancy < num_voxels)
    _, calculated_occupancy_counts = np.unique(calculated_occupancy, return_counts=True)
    assert calculated_occupancy_counts.sum() == num_atoms
    # the max count should be int(num_atoms / num_voxels) + 1
    max_allowed_occupancy = np.floor(num_atoms / num_voxels) + 1
    # by the same logic, the minimal occupancy is the same - 1
    min_allowed_occupancy = np.floor(num_atoms / num_voxels)
    assert np.all(calculated_occupancy_counts <= max_allowed_occupancy)
    assert np.all(max_allowed_occupancy >= min_allowed_occupancy)
