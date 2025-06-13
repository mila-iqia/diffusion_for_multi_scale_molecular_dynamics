import math
from itertools import product

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.utils import (
    find_partition_sizes, get_distances_from_reference_point,
    partition_relative_coordinates_for_voxels, select_occupied_voxels)


@pytest.fixture(params=[2, 3, 30])
def number_of_atoms(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def spatial_dimension(request):
    return request.param


@pytest.fixture
def basis_vectors(spatial_dimension):
    box_size = np.random.random((spatial_dimension,))
    return np.diag(box_size)


@pytest.fixture
def lattice_parameters(spatial_dimension, basis_vectors):
    lp = np.concatenate(
        (
            np.diag(basis_vectors),
            np.zeros(int(spatial_dimension * (spatial_dimension - 1) / 2)),
        )
    )
    return lp


@pytest.fixture
def box_size(lattice_parameters, spatial_dimension):
    return lattice_parameters[:spatial_dimension]


@pytest.fixture
def atom_relative_coordinates(number_of_atoms, spatial_dimension):
    return np.random.random((number_of_atoms, spatial_dimension))


@pytest.fixture
def atom_cartesian_positions(atom_relative_coordinates, basis_vectors):
    return np.matmul(atom_relative_coordinates, basis_vectors)


@pytest.fixture
def reference_point_relative(spatial_dimension):
    return np.random.random((spatial_dimension,))


@pytest.fixture
def reference_point_cartesian(reference_point_relative, basis_vectors):
    return np.matmul(reference_point_relative, basis_vectors)


def test_get_distances_from_reference_point(
    atom_relative_coordinates,
    reference_point_relative,
    atom_cartesian_positions,
    reference_point_cartesian,
    lattice_parameters,
    basis_vectors,
    spatial_dimension,
):
    expected_distances = []
    box_dimensions = np.diag(basis_vectors)
    for atom in atom_cartesian_positions:
        distance = 0.0
        for d in range(spatial_dimension):
            distance_ = min(
                [
                    (atom[d] - reference_point_cartesian[d]) ** 2,
                    (atom[d] - reference_point_cartesian[d] + box_dimensions[d]) ** 2,
                    (atom[d] - reference_point_cartesian[d] - box_dimensions[d]) ** 2,
                ]
            )
            distance += distance_
        expected_distances.append(math.sqrt(distance))

    calculated_distances = get_distances_from_reference_point(
        atom_relative_coordinates, reference_point_relative, lattice_parameters
    )

    for ed, cd in zip(expected_distances, calculated_distances):
        assert np.isclose(ed, cd)


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


def test_select_occupied_voxels(num_voxels, num_atoms_for_voxels):
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
    calculated_occupancy = select_occupied_voxels(num_voxels, num_atoms_for_voxels)
    assert np.all(0 <= calculated_occupancy)
    assert np.all(calculated_occupancy < num_voxels)
    _, calculated_occupancy_counts = np.unique(calculated_occupancy, return_counts=True)
    assert calculated_occupancy_counts.sum() == num_atoms_for_voxels
    # the max count should be int(num_atoms / num_voxels) + 1
    max_allowed_occupancy = np.floor(num_atoms_for_voxels / num_voxels) + 1
    # by the same logic, the minimal occupancy is the same - 1
    min_allowed_occupancy = np.floor(num_atoms_for_voxels / num_voxels)
    assert np.all(calculated_occupancy_counts <= max_allowed_occupancy)
    assert np.all(max_allowed_occupancy >= min_allowed_occupancy)
